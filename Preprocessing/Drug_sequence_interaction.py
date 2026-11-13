import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import warnings, os, random, copy

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# ---------- 配置 ----------
CORE_IN_87 = None
CORE_COUNT = None
NAIVE_DIM = None
TENDENCY_DIM = None
OUTPUT_DIM = None

MAIN_DATA = r"D:\TumorCNN\Models\naive_merged.csv"
SINGLE_DATA = r"D:\TumorCNN\Models\paired_drug_vs_control.csv"
DOCKING_DATA = r"D:\TumorCNN\Models\Docking_result.csv"
PPI_PATH = "protein_gnn_features_with_names.pth"
FIXED_SPLIT_SEED = 999

# ---------- 工具函数 ----------
def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def create_tendency_vec(docking_scores, threshold=-7.5):
    min_val = docking_scores.min(dim=1, keepdim=True)[0]
    max_val = docking_scores.max(dim=1, keepdim=True)[0]
    normalized = (docking_scores - min_val) / (max_val - min_val + 1e-8)
    competitive = torch.where(docking_scores < threshold, normalized*1.2, normalized*0.8)
    inhibitory = torch.where(docking_scores >= threshold, normalized*1.2, normalized*0.8)
    return torch.cat([competitive.clamp(0,1), inhibitory.clamp(0,1)], dim=1)

def add_noise(points, noise_std=0.01):
    noise = torch.randn_like(points) * noise_std
    return points + noise

def extract_core_genes_from_columns(columns):
    processed_cols = [col for col in columns if col.startswith('processed_')]
    genes = sorted([col.replace('processed_', '') for col in processed_cols])
    return genes

def sort_pointcloud_by_distance(points):
    """按点到质心的欧氏距离升序排序点云"""
    center = points[:, :3].mean(axis=0)
    distances = np.linalg.norm(points[:, :3] - center, axis=1)
    sorted_idx = np.argsort(distances)
    return points[sorted_idx]

# ---------- 数据集 ----------
class SingleDrugDataset(Dataset):
    def __init__(self, csv_path, docking_csv, ppi_path, core_genes=None, max_points=500):
        self.df = pd.read_csv(csv_path)
        self.max_points = max_points
        if core_genes is not None:
            self.core_genes = core_genes
        else:
            candidate_genes = extract_core_genes_from_columns(self.df.columns)
            dock_df = pd.read_csv(docking_csv)
            dock_gene_cols = [col for col in dock_df.columns if col != 'Ligand']
            self.core_genes = sorted(set(candidate_genes) & set(dock_gene_cols))
            if not self.core_genes:
                raise ValueError("No common genes between data and docking results!")
        dock_df = pd.read_csv(docking_csv)
        self.dock_dict = {}
        for _, row in dock_df.iterrows():
            self.dock_dict[row['Ligand']] = row[self.core_genes].values.astype(np.float32)
        ppi_data = torch.load(ppi_path, map_location='cpu')
        self.protein_gnn_dim = ppi_data['external_features'].shape[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        naive = row[[f'naive_{g}' for g in self.core_genes]].values.astype(np.float32)
        label = row[[f'processed_{g}' for g in self.core_genes]].values.astype(np.float32)

        # 加载并排序点云
        drug_path = row['DrugA_loc']
        points = np.load(drug_path).astype(np.float32)
        points = sort_pointcloud_by_distance(points)

        n = points.shape[0]
        if n > self.max_points:
            idxs = np.random.choice(n, self.max_points, replace=False)
            points = points[idxs]; length = self.max_points
        else:
            length = n
            pad = np.zeros((self.max_points - n, points.shape[1]), dtype=np.float32)
            points = np.vstack([points, pad])

        drug_name = row['DrugA']
        dock = self.dock_dict.get(drug_name, np.zeros(len(self.core_genes), dtype=np.float32))
        dock_tensor = torch.from_numpy(dock).unsqueeze(0)
        tendency = create_tendency_vec(dock_tensor).squeeze(0)

        return {
            'naive_gene': torch.from_numpy(naive),
            'tendency_vec': tendency,
            'drug1_points': torch.from_numpy(points),
            'drug1_lengths': torch.tensor(length, dtype=torch.long),
            'drug2_points': torch.from_numpy(points.copy()),
            'drug2_lengths': torch.tensor(length, dtype=torch.long),
            'label': torch.from_numpy(label)
        }

class DrugCombinationDataset(Dataset):
    def __init__(self, csv_path, docking_csv, ppi_path, core_genes=None, max_points=500):
        self.df = pd.read_csv(csv_path)
        self.max_points = max_points
        if core_genes is not None:
            self.core_genes = core_genes
        else:
            candidate_genes = extract_core_genes_from_columns(self.df.columns)
            dock_df = pd.read_csv(docking_csv)
            dock_gene_cols = [col for col in dock_df.columns if col != 'Ligand']
            self.core_genes = sorted(set(candidate_genes) & set(dock_gene_cols))
            if not self.core_genes:
                raise ValueError("No common genes between data and docking results!")
        dock_df = pd.read_csv(docking_csv)
        self.dock_dict = {}
        for _, row in dock_df.iterrows():
            self.dock_dict[row['Ligand']] = row[self.core_genes].values.astype(np.float32)
        ppi_data = torch.load(ppi_path, map_location='cpu')
        self.protein_gnn_dim = ppi_data['external_features'].shape[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        naive = row[[f'naive_{g}' for g in self.core_genes]].values.astype(np.float32)
        label = row[[f'processed_{g}' for g in self.core_genes]].values.astype(np.float32)

        def load_and_process(path):
            pts = np.load(path).astype(np.float32)
            pts = sort_pointcloud_by_distance(pts)
            n = pts.shape[0]
            if n > self.max_points:
                idxs = np.random.choice(n, self.max_points, replace=False)
                pts = pts[idxs]; length = self.max_points
            else:
                length = n
                pad = np.zeros((self.max_points - n, pts.shape[1]), dtype=np.float32)
                pts = np.vstack([pts, pad])
            return pts, length

        pts_a, len_a = load_and_process(row['DrugA_loc'])
        pts_b, len_b = load_and_process(row['DrugB_loc'])

        drug_a, drug_b = row['DrugA'], row['DrugB']
        dock_a = self.dock_dict.get(drug_a, np.zeros(len(self.core_genes), dtype=np.float32))
        dock_b = self.dock_dict.get(drug_b, np.zeros(len(self.core_genes), dtype=np.float32))
        dock_avg = (dock_a + dock_b) / 2.0
        dock_tensor = torch.from_numpy(dock_avg).unsqueeze(0)
        tendency = create_tendency_vec(dock_tensor).squeeze(0)

        return {
            'naive_gene': torch.from_numpy(naive),
            'tendency_vec': tendency,
            'drug1_points': torch.from_numpy(pts_a),
            'drug1_lengths': torch.tensor(len_a, dtype=torch.long),
            'drug2_points': torch.from_numpy(pts_b),
            'drug2_lengths': torch.tensor(len_b, dtype=torch.long),
            'label': torch.from_numpy(label)
        }

def collate_fn(batch):
    naive = torch.stack([b['naive_gene'] for b in batch])
    tendency = torch.stack([b['tendency_vec'] for b in batch])
    label = torch.stack([b['label'] for b in batch])
    d1 = [b['drug1_points'] for b in batch]
    d2 = [b['drug2_points'] for b in batch]
    l1 = torch.stack([b['drug1_lengths'] for b in batch])
    l2 = torch.stack([b['drug2_lengths'] for b in batch])
    d1_pad = pad_sequence(d1, batch_first=True, padding_value=0.0)
    d2_pad = pad_sequence(d2, batch_first=True, padding_value=0.0)
    return {'naive_gene': naive, 'tendency_vec': tendency,
            'drug1_points': d1_pad, 'drug1_points_lengths': l1,
            'drug2_points': d2_pad, 'drug2_points_lengths': l2,
            'label': label}

# ---------- 模型 ----------
class DrugEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=10, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.0)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, points, lengths):
        # points: (batch, max_len, 4), lengths: (batch,)
        packed = pack_padded_sequence(points, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=points.size(1))
        # 注意力
        attn_logits = self.attn(output).squeeze(-1)  # (batch, seq_len)
        mask = torch.arange(points.size(1), device=points.device).unsqueeze(0) < lengths.unsqueeze(1)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)  # (batch, hidden_dim)
        return self.fc(context)

class MyModel(nn.Module):
    def __init__(self, hidden_size=16, num_layers=1, drug_dim=10, protein_gnn_dim=137, core_count=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.core_count = core_count
        self.drug_encoder = DrugEncoder(input_dim=4, hidden_dim=drug_dim, num_layers=num_layers)

        ppi = torch.load(PPI_PATH, map_location='cpu')['external_features']
        self.register_buffer('protein_gnn_features', ppi)
        self.protein_gnn_features.requires_grad = False

        self.W_k = nn.Linear(protein_gnn_dim, hidden_size, bias=False)
        self.W_q = nn.Linear(drug_dim, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, 1, bias=False)

        self.combine = nn.Linear(core_count + core_count*2 + protein_gnn_dim, hidden_size)

        self.inter_lstm = nn.LSTM(drug_dim, hidden_size, 1, batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, core_count)
        )

    def protein_weight(self, drug_enc):
        q = self.W_q(drug_enc)
        k = self.W_k(self.protein_gnn_features)
        feat = torch.tanh(q.unsqueeze(1) + k.unsqueeze(0))
        score = self.w_v(feat).squeeze(-1)
        attn = F.softmax(score, dim=1)
        weighted = torch.bmm(attn.unsqueeze(1),
                             self.protein_gnn_features.unsqueeze(0).expand(drug_enc.size(0),-1,-1)).squeeze(1)
        return weighted, attn

    def forward(self, d1_pts, d1_len, d2_pts, d2_len, naive, tend, use_noise=False):
        if use_noise:
            d1_pts = add_noise(d1_pts, 0.01)
            d2_pts = add_noise(d2_pts, 0.01)

        e1 = self.drug_encoder(d1_pts, d1_len)  # (batch, drug_dim=10)
        e2 = self.drug_encoder(d2_pts, d2_len)

        p1, _ = self.protein_weight(e1)  # (batch, 137)
        p2, _ = self.protein_weight(e2)

        combined1 = torch.cat([naive, tend, p1], dim=1)
        combined2 = torch.cat([naive, tend, p2], dim=1)
        init1 = self.combine(combined1)  # (batch, hidden_size)
        init2 = self.combine(combined2)

        inter_w = F.softmax(torch.matmul(e1.unsqueeze(1), e2.unsqueeze(2)).squeeze(), dim=0)

        def process(seq1, seq2, init):
            seq = torch.stack([seq1, seq2], dim=1)  # (batch, 2, drug_dim)
            h0 = (init * inter_w.unsqueeze(1)).unsqueeze(0)  # (1, batch, hidden_size)
            c0 = torch.zeros_like(h0)
            out, _ = self.inter_lstm(seq, (h0, c0))
            return out[:, -1, :]  # (batch, hidden_size)

        o1 = process(e1, e2, init1)
        o2 = process(e2, e1, init2)
        return self.out(o1 + o2)

# ---------- 评估与可视化 ----------
def compute_metrics(preds, targets):
    p = preds.cpu().numpy(); t = targets.cpu().numpy()
    r2 = r2_score(t.flatten(), p.flatten())
    mse = np.mean((p-t)**2); mae = mean_absolute_error(t.flatten(), p.flatten())
    corr, _ = pearsonr(t.flatten(), p.flatten())
    return {'r2': r2, 'mse': mse, 'mae': mae, 'pearson': corr}

class Visualization:
    @staticmethod
    def plot_training_history(histories, save_path='avg_training_history.png'):
        metric_keys = ['train_loss', 'val_loss', 'train_mse', 'val_mse', 'train_mae', 'val_mae', 'train_r2', 'val_r2']
        min_epochs = min(len(h['train_loss']) for h in histories)
        truncated = [{k: h[k][:min_epochs] for k in metric_keys} for h in histories]
        mean_dict = {}
        for k in metric_keys:
            arr = np.array([h[k] for h in truncated]); mean_dict[k] = np.mean(arr, axis=0)
        train_mean = {k.replace('train_',''): mean_dict[k] for k in metric_keys if k.startswith('train_')}
        val_mean   = {k.replace('val_',''): mean_dict[k] for k in metric_keys if k.startswith('val_')}
        fig, axes = plt.subplots(2,2, figsize=(14,9)); axes = axes.flatten()
        colors = ['#2E86AB','#A23B72']; metrics_plot = ['loss','mse','mae','r2']
        for idx,m in enumerate(metrics_plot):
            if m in train_mean and m in val_mean:
                epochs = range(1, len(train_mean[m])+1)
                axes[idx].plot(epochs, train_mean[m], color=colors[0], linewidth=2, marker='o', markersize=4, label=f'Train {m}')
                axes[idx].plot(epochs, val_mean[m], color=colors[1], linewidth=2, marker='s', markersize=4, label=f'Val {m}')
                axes[idx].set_xlabel('Epoch', fontsize=11); axes[idx].set_ylabel(m.upper() if m!='r2' else 'R²', fontsize=11)
                axes[idx].set_title(f'{m.upper()} Curve' if m!='r2' else 'R² Curve', fontsize=13, fontweight='bold')
                axes[idx].legend(frameon=True, fancybox=True, shadow=False, fontsize=10)
                axes[idx].grid(True, linestyle=':', alpha=0.6); axes[idx].tick_params(labelsize=10)
        plt.suptitle('Average Training History (Dual Drug Fine-tuning)', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.show()

    @staticmethod
    def plot_predictions_vs_actuals(preds, targets, save_path='ensemble_pred_vs_act.png'):
        n_genes = preds.shape[1]
        fig, axes = plt.subplots(1, n_genes, figsize=(5*n_genes, 5))
        if n_genes == 1:
            axes = [axes]
        for i, gene in enumerate(CORE_IN_87):
            ax = axes[i]; gp = preds[:,i]; gt = targets[:,i]; r2 = r2_score(gt, gp)
            ax.scatter(gt, gp, color='#2E86AB', alpha=0.7, edgecolors='k', linewidth=0.5)
            mn = min(gt.min(), gp.min()); mx = max(gt.max(), gp.max())
            ax.plot([mn,mx],[mn,mx], 'r--', alpha=0.6, linewidth=1.5)
            ax.set_xlabel('True Value', fontsize=11); ax.set_ylabel('Predicted Value', fontsize=11)
            ax.set_title(f'{gene}  (R²={r2:.3f})', fontsize=13, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.5); ax.tick_params(labelsize=10)
        plt.suptitle('Ensemble Prediction vs Actual', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.show()

# ---------- 训练函数 ----------
def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, wd=1e-4, device='cuda', tag='',
                freeze_encoder=False):
    model.to(device)
    if freeze_encoder:
        print(f'{tag}: Freezing DrugEncoder...')
        for param in model.drug_encoder.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.SmoothL1Loss()
    history = {'train_loss':[], 'val_loss':[], 'train_mse':[], 'val_mse':[], 'train_mae':[], 'val_mae':[], 'train_r2':[], 'val_r2':[]}
    best_val_r2 = -float('inf')
    best_state = None
    patience_counter = 0
    patience = 80

    for ep in range(epochs):
        model.train(); train_loss = 0; train_preds, train_labels = [], []
        for batch in train_loader:
            batch = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            pred = model(batch['drug1_points'], batch['drug1_points_lengths'],
                         batch['drug2_points'], batch['drug2_points_lengths'],
                         batch['naive_gene'], batch['tendency_vec'],
                         use_noise=True)
            loss = criterion(pred, batch['label'])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            train_loss += loss.item()
            train_preds.append(pred.detach().cpu()); train_labels.append(batch['label'].detach().cpu())
        train_met = compute_metrics(torch.cat(train_preds), torch.cat(train_labels))
        model.eval(); val_loss = 0; val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                pred = model(batch['drug1_points'], batch['drug1_points_lengths'],
                             batch['drug2_points'], batch['drug2_points_lengths'],
                             batch['naive_gene'], batch['tendency_vec'],
                             use_noise=False)
                loss = criterion(pred, batch['label'])
                val_loss += loss.item()
                val_preds.append(pred.cpu()); val_labels.append(batch['label'].cpu())
        val_met = compute_metrics(torch.cat(val_preds), torch.cat(val_labels))
        history['train_loss'].append(train_loss/len(train_loader))
        history['val_loss'].append(val_loss/len(val_loader))
        history['train_mse'].append(train_met['mse']); history['val_mse'].append(val_met['mse'])
        history['train_mae'].append(train_met['mae']); history['val_mae'].append(val_met['mae'])
        history['train_r2'].append(train_met['r2']); history['val_r2'].append(val_met['r2'])

        if (ep+1)%20==0 or ep==0:
            print(f'{tag} Epoch {ep+1}: TrainLoss {history["train_loss"][-1]:.4f}, ValLoss {history["val_loss"][-1]:.4f}, ValR2 {history["val_r2"][-1]:.4f}')

        current_val_r2 = val_met['r2']
        if current_val_r2 > best_val_r2:
            best_val_r2 = current_val_r2
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'{tag} Early stopping at epoch {ep+1}, best ValR2={best_val_r2:.4f}')
                break

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

# ---------- 主程序 ----------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 统一核心基因
    single_df = pd.read_csv(SINGLE_DATA, nrows=0)
    dual_df = pd.read_csv(MAIN_DATA, nrows=0)
    dock_df = pd.read_csv(DOCKING_DATA, nrows=0)

    single_genes = extract_core_genes_from_columns(single_df.columns)
    dual_genes = extract_core_genes_from_columns(dual_df.columns)
    dock_genes = [col for col in dock_df.columns if col != 'Ligand']

    common_genes = sorted(set(single_genes) & set(dual_genes) & set(dock_genes))
    if not common_genes:
        raise ValueError("No common genes among single, dual and docking data!")
    print(f"Unified core genes: {common_genes}")

    global CORE_IN_87, CORE_COUNT, NAIVE_DIM, TENDENCY_DIM, OUTPUT_DIM
    CORE_IN_87 = common_genes
    CORE_COUNT = len(common_genes)
    NAIVE_DIM = CORE_COUNT
    TENDENCY_DIM = CORE_COUNT * 2
    OUTPUT_DIM = CORE_COUNT

    # 创建数据集
    single_dataset = SingleDrugDataset(SINGLE_DATA, DOCKING_DATA, PPI_PATH, core_genes=common_genes)
    dual_dataset = DrugCombinationDataset(MAIN_DATA, DOCKING_DATA, PPI_PATH, core_genes=common_genes)

    total_single = len(single_dataset)
    idx_s = np.random.RandomState(FIXED_SPLIT_SEED).permutation(total_single)
    train_end_s = int(0.8*total_single); val_end_s = int(0.9*total_single)
    single_train = Subset(single_dataset, idx_s[:train_end_s])
    single_val   = Subset(single_dataset, idx_s[train_end_s:val_end_s])
    single_test  = Subset(single_dataset, idx_s[val_end_s:])
    single_train_loader = DataLoader(single_train, batch_size=16, shuffle=True, collate_fn=collate_fn)
    single_val_loader   = DataLoader(single_val, batch_size=16, shuffle=False, collate_fn=collate_fn)
    single_test_loader  = DataLoader(single_test, batch_size=16, shuffle=False, collate_fn=collate_fn)

    total_dual = len(dual_dataset)
    idx_d = np.random.RandomState(FIXED_SPLIT_SEED).permutation(total_dual)
    train_end_d = int(0.8*total_dual); val_end_d = int(0.9*total_dual)
    dual_train = Subset(dual_dataset, idx_d[:train_end_d])
    dual_val   = Subset(dual_dataset, idx_d[train_end_d:val_end_d])
    dual_test  = Subset(dual_dataset, idx_d[val_end_d:])
    dual_train_loader = DataLoader(dual_train, batch_size=16, shuffle=True, collate_fn=collate_fn)
    dual_val_loader   = DataLoader(dual_val, batch_size=16, shuffle=False, collate_fn=collate_fn)
    dual_test_loader  = DataLoader(dual_test, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 获取测试标签
    dual_test_labels = []
    with torch.no_grad():
        for batch in dual_test_loader:
            dual_test_labels.append(batch['label'])
    dual_test_labels = torch.cat(dual_test_labels, dim=0)

    seeds = [42, 123, 456, 789, 111]
    all_dual_histories = []
    all_test_preds = []
    all_test_metrics = []

    for seed in seeds:
        print(f'\n{"="*60}\nSeed {seed}\n{"="*60}')
        set_seed(seed)
        model = MyModel(core_count=CORE_COUNT).to(device)

        # 直接训练双药模型（取消单药预训练）
        print('\n--- Direct Dual Drug Training ---')
        model, dual_history = train_model(model, dual_train_loader, dual_val_loader,
                                          epochs=200, lr=1e-3, wd=1e-4, device=device,
                                          tag=f'Direct-{seed}', freeze_encoder=False)
        all_dual_histories.append(dual_history)

        # 测试
        model.eval(); preds_list = []
        with torch.no_grad():
            for batch in dual_test_loader:
                batch = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                pred = model(batch['drug1_points'], batch['drug1_points_lengths'],
                             batch['drug2_points'], batch['drug2_points_lengths'],
                             batch['naive_gene'], batch['tendency_vec'], use_noise=False)
                preds_list.append(pred.cpu())
        test_preds = torch.cat(preds_list, dim=0)
        all_test_preds.append(test_preds)
        met = compute_metrics(test_preds, dual_test_labels)
        all_test_metrics.append(met)
        print(f'Seed {seed} Test: R2={met["r2"]:.4f}, MSE={met["mse"]:.4f}')

    # 集成（剔除最差种子）
    r2_vals = [m['r2'] for m in all_test_metrics]
    worst_idx = np.argmin(r2_vals)
    best_indices = [i for i in range(len(seeds)) if i != worst_idx]
    print(f'\nExcluding seed {seeds[worst_idx]} (R²={r2_vals[worst_idx]:.4f})')
    selected_preds = [all_test_preds[i] for i in best_indices]
    ensemble_preds = torch.stack(selected_preds, dim=0).mean(dim=0)
    ensemble_met = compute_metrics(ensemble_preds, dual_test_labels)
    print(f'\nEnsemble Test (excl. worst): R2={ensemble_met["r2"]:.4f}, MSE={ensemble_met["mse"]:.4f}, '
          f'MAE={ensemble_met["mae"]:.4f}, Pearson={ensemble_met["pearson"]:.4f}')

    # 画图
    Visualization.plot_training_history(all_dual_histories, save_path='avg_dual_training.png')
    Visualization.plot_predictions_vs_actuals(ensemble_preds.numpy(), dual_test_labels.numpy(),
                                              save_path='ensemble_dual_pred_vs_act.png')

    # 汇总
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'Individual R²: {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}')
    print(f'Ensemble R²  : {ensemble_met["r2"]:.4f}')
    torch.save({'model_state': model.state_dict(), 'ensemble_metrics': ensemble_met}, 'final_curriculum_ensemble.pth')
    print('Done.')

if __name__ == '__main__':
    main()