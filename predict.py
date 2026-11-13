import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings, os, random, itertools, joblib

warnings.filterwarnings('ignore')

# ---------- 配置（请根据实际情况修改） ----------
RNN_CHECKPOINT = r"D:\TumorCNN\Models\final_curriculum_ensemble.pth"
ENSEMBLE_MODEL_PATH = r"D:\TumorCNN\Models\ensemble_classifier.pkl"
DRUG_INFO_FILE = r"D:\TumorCNN\Models\ALL_drug_info_merged.csv"
CELL_LINE_FILE = r"D:\TumorCNN\Models\Naive_gene_from_depmap_and_geo.csv"
TRAINING_DATA = r"D:\TumorCNN\Models\CNN_integrated_result.csv"  # 用于重新拟合PCA
DOCKING_FILE = r"D:\TumorCNN\Models\CNN_merged_MYL9_CYP1B1_PCK2.csv"  # 用于构建tendency
PPI_PATH = r"D:\TumorCNN\Models\protein_gnn_features_with_names.pth"
GENE_SUBSET = ['MYL9', 'CYP1B1', 'PCK2']
THRESHOLD = 0.8  # 协同概率阈值
MAX_POINTS = 500
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------- 工具函数（与训练时一致） ----------
def create_tendency_vec(docking_scores, threshold=-7.5):
    min_val = docking_scores.min(dim=1, keepdim=True)[0]
    max_val = docking_scores.max(dim=1, keepdim=True)[0]
    normalized = (docking_scores - min_val) / (max_val - min_val + 1e-8)
    competitive = torch.where(docking_scores < threshold, normalized * 1.2, normalized * 0.8)
    inhibitory = torch.where(docking_scores >= threshold, normalized * 1.2, normalized * 0.8)
    return torch.cat([competitive.clamp(0, 1), inhibitory.clamp(0, 1)], dim=1)


def sort_pointcloud_by_distance(points):
    center = points[:, :3].mean(axis=0)
    distances = np.linalg.norm(points[:, :3] - center, axis=1)
    sorted_idx = np.argsort(distances)
    return points[sorted_idx]


def set_seed(seed):
    np.random.seed(seed);
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed);
    random.seed(seed)
    torch.backends.cudnn.deterministic = True;
    torch.backends.cudnn.benchmark = False


# ---------- RNN 模型定义（与训练时完全相同） ----------
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
        packed = pack_padded_sequence(points, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=points.size(1))
        attn_logits = self.attn(output).squeeze(-1)
        mask = torch.arange(points.size(1), device=points.device).unsqueeze(0) < lengths.unsqueeze(1)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)
        return self.fc(context)


class MyModel(nn.Module):
    def __init__(self, hidden_size=16, num_layers=1, drug_dim=10, protein_gnn_dim=137, core_count=3):
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

        self.combine = nn.Linear(core_count + core_count * 2 + protein_gnn_dim, hidden_size)
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
                             self.protein_gnn_features.unsqueeze(0).expand(drug_enc.size(0), -1, -1)).squeeze(1)
        return weighted, attn

    def forward(self, d1_pts, d1_len, d2_pts, d2_len, naive, tend, use_noise=False):
        if use_noise:
            d1_pts = d1_pts + torch.randn_like(d1_pts) * 0.01
            d2_pts = d2_pts + torch.randn_like(d2_pts) * 0.01

        e1 = self.drug_encoder(d1_pts, d1_len)
        e2 = self.drug_encoder(d2_pts, d2_len)

        p1, _ = self.protein_weight(e1)
        p2, _ = self.protein_weight(e2)

        combined1 = torch.cat([naive, tend, p1], dim=1)
        combined2 = torch.cat([naive, tend, p2], dim=1)
        init1 = self.combine(combined1)
        init2 = self.combine(combined2)

        inter_w = torch.ones(e1.size(0), device=e1.device)

        def process(seq1, seq2, init):
            seq = torch.stack([seq1, seq2], dim=1)
            h0 = (init * inter_w.unsqueeze(1)).unsqueeze(0)
            c0 = torch.zeros_like(h0)
            out, _ = self.inter_lstm(seq, (h0, c0))
            return out[:, -1, :]

        o1 = process(e1, e2, init1)
        o2 = process(e2, e1, init2)
        return self.out(o1 + o2)


# ---------- 加载模型 ----------
def load_rnn_model(checkpoint_path, device):
    model = MyModel(core_count=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state'].items()
                       if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Loaded RNN model from {checkpoint_path}")
    return model


def load_ensemble_model(model_path):
    """加载软投票集成模型（包含models, weights, scaler）"""
    data = joblib.load(model_path)
    print(f"Loaded ensemble model from {model_path}")
    return data['models'], data['weights'], data['scaler']


# ---------- 重新拟合PCA（与训练时一致） ----------
def fit_pca_from_training_data(training_data_path, docking_file, ppi_path, device):
    """从训练数据中重新拟合PCA，返回pca对象和特征列名"""
    # 这部分逻辑与scoring_CNN.py中的FeatureExtractor类似，但只用于PCA
    # 为了简化，我们直接读取训练数据文件并提取protein_raw
    # 注意：这里需要用到RNN模型来提取特征，但我们只需要protein_raw，而protein_raw是从RNN的protein_weight得到的
    # 实际上，PCA是在protein_raw上拟合的，而protein_raw依赖于RNN模型。因此我们需要加载RNN模型并提取所有样本的protein_raw。
    # 这是一个循环依赖：我们需要RNN来提取protein_raw，而RNN已经加载了。
    # 因此，我们在这里加载RNN模型，然后提取所有样本的protein_raw，再拟合PCA。
    rnn_model = load_rnn_model(RNN_CHECKPOINT, device)

    # 读取训练数据
    df = pd.read_csv(training_data_path)
    # 假设格式与之前一致
    COL_NAMES = ['Drug1', 'Drug1_loc', 'Drug2', 'Drug2_loc', 'Cell_line', 'ZIP', 'naive_MYL9', 'naive_CYP1B1',
                 'naive_PCK2']
    df.columns = COL_NAMES

    # 读取对接数据
    dock_df = pd.read_csv(docking_file)
    dock_dict = {}
    for _, row in dock_df.iterrows():
        dock_dict[row['Ligand']] = row[GENE_SUBSET].values.astype(np.float32)

    # 标准化naive
    naive_cols = [f'naive_{g}' for g in GENE_SUBSET]
    naive_scaler = StandardScaler()
    naive_values = naive_scaler.fit_transform(df[naive_cols].values)

    protein_raw_list = []
    for idx, row in df.iterrows():
        naive = naive_values[idx]
        drug_a, drug_b = row['Drug1'], row['Drug2']
        dock_a = dock_dict.get(drug_a, np.zeros(len(GENE_SUBSET), dtype=np.float32))
        dock_b = dock_dict.get(drug_b, np.zeros(len(GENE_SUBSET), dtype=np.float32))
        dock_avg = (dock_a + dock_b) / 2.0
        dock_tensor = torch.from_numpy(dock_avg).unsqueeze(0).to(device).float()
        tendency = create_tendency_vec(dock_tensor).squeeze(0).cpu().numpy()

        def load_points(path):
            pts = np.load(path).astype(np.float32)
            pts = sort_pointcloud_by_distance(pts)
            n = pts.shape[0]
            if n > MAX_POINTS:
                idxs = np.random.choice(n, MAX_POINTS, replace=False)
                pts = pts[idxs];
                length = MAX_POINTS
            else:
                length = n
                pad = np.zeros((MAX_POINTS - n, pts.shape[1]), dtype=np.float32)
                pts = np.vstack([pts, pad])
            return torch.from_numpy(pts).unsqueeze(0).to(device).float(), torch.tensor([length]).to(device)

        pts_a, len_a = load_points(row['Drug1_loc'])
        pts_b, len_b = load_points(row['Drug2_loc'])

        with torch.no_grad():
            naive_tensor = torch.from_numpy(naive).unsqueeze(0).to(device).float()
            tend_tensor = torch.from_numpy(tendency).unsqueeze(0).to(device).float()
            _ = rnn_model(pts_a, len_a, pts_b, len_b, naive_tensor, tend_tensor, use_noise=False)
            e1 = rnn_model.drug_encoder(pts_a, len_a)
            e2 = rnn_model.drug_encoder(pts_b, len_b)
            p1, _ = rnn_model.protein_weight(e1)
            p2, _ = rnn_model.protein_weight(e2)
            p_concat = torch.cat([p1, p2], dim=1).cpu().numpy().flatten()
        protein_raw_list.append(p_concat)

    protein_raw_arr = np.array(protein_raw_list)
    pca = PCA(n_components=10)
    pca.fit(protein_raw_arr)
    print(f"Fitted PCA on {len(protein_raw_arr)} samples from training data.")
    return pca, naive_scaler


# ---------- 对单个药物组合和单个细胞系提取特征 ----------
def extract_features_for_pair_cell(rnn_model, drug_a_name, drug_b_name, cell_naive,
                                   dock_dict, pca, naive_scaler, device):
    """
    返回22维特征向量（已标准化）
    """
    # 获取对接分数
    dock_a = dock_dict.get(drug_a_name, np.zeros(len(GENE_SUBSET), dtype=np.float32))
    dock_b = dock_dict.get(drug_b_name, np.zeros(len(GENE_SUBSET), dtype=np.float32))
    dock_avg = (dock_a + dock_b) / 2.0
    dock_tensor = torch.from_numpy(dock_avg).unsqueeze(0).to(device).float()
    tendency = create_tendency_vec(dock_tensor).squeeze(0).cpu().numpy()

    # 标准化naive
    naive_scaled = naive_scaler.transform(cell_naive.reshape(1, -1)).flatten()

    # 加载点云（假设npy路径可以从药物信息中获取，这里需要外部传入）
    # 实际上，我们需要从drug_info文件中获取npy_path。下面假设在调用时已经传入。
    # 为了通用，我们让调用者传入点云路径。
    # 但为了接口简洁，我们在这里不处理点云加载，而是由上层提供。
    # 我们修改函数签名，直接接收点云tensor。
    pass


# 由于特征提取涉及点云加载，我们将其放在主循环中处理。
# 下面直接编写主预测逻辑。

def main():
    print(f"Using device: {DEVICE}")

    # 1. 加载RNN模型
    rnn_model = load_rnn_model(RNN_CHECKPOINT, DEVICE)

    # 2. 加载软投票集成模型
    ensemble_models, ensemble_weights, scaler = load_ensemble_model(ENSEMBLE_MODEL_PATH)

    # 3. 重新拟合PCA（从训练数据）
    print("Re-fitting PCA from training data...")
    pca, naive_scaler = fit_pca_from_training_data(TRAINING_DATA, DOCKING_FILE, PPI_PATH, DEVICE)

    # 4. 读取药物信息
    drug_df = pd.read_csv(DRUG_INFO_FILE)
    drug_names = drug_df['drug_name'].tolist()
    drug_npy_paths = drug_df['npy_path'].tolist()
    drug_docking = drug_df[['MYL9', 'CYP1B1', 'PCK2']].values.astype(np.float32)
    drug_info = {name: {'npy': path, 'dock': dock}
                 for name, path, dock in zip(drug_names, drug_npy_paths, drug_docking)}

    # 5. 读取细胞系基线
    cell_df = pd.read_csv(CELL_LINE_FILE)
    cell_lines = cell_df['CellLine'].tolist()
    cell_naive = cell_df[['MYL9', 'CYP1B1', 'PCK2']].values.astype(np.float32)

    # 6. 构建药物组合列表
    drug_combinations = list(itertools.combinations(drug_names, 2))
    print(f"Total drug combinations: {len(drug_combinations)}")
    print(f"Total cell lines: {len(cell_lines)}")

    # 7. 准备结果存储
    results = []

    # 对每个药物组合
    for combo_idx, (drug_a, drug_b) in enumerate(drug_combinations):
        if combo_idx % 10 == 0:
            print(f"Processing combination {combo_idx + 1}/{len(drug_combinations)}...")

        # 获取药物信息
        info_a = drug_info[drug_a]
        info_b = drug_info[drug_b]

        # 加载点云（只加载一次，因为同一药物的点云在不同细胞系中相同）
        pts_a = np.load(info_a['npy']).astype(np.float32)
        pts_a = sort_pointcloud_by_distance(pts_a)
        n_a = pts_a.shape[0]
        if n_a > MAX_POINTS:
            idxs = np.random.choice(n_a, MAX_POINTS, replace=False)
            pts_a = pts_a[idxs];
            len_a = MAX_POINTS
        else:
            len_a = n_a
            pad = np.zeros((MAX_POINTS - n_a, pts_a.shape[1]), dtype=np.float32)
            pts_a = np.vstack([pts_a, pad])
        pts_a_tensor = torch.from_numpy(pts_a).unsqueeze(0).to(DEVICE).float()
        len_a_tensor = torch.tensor([len_a]).to(DEVICE)

        pts_b = np.load(info_b['npy']).astype(np.float32)
        pts_b = sort_pointcloud_by_distance(pts_b)
        n_b = pts_b.shape[0]
        if n_b > MAX_POINTS:
            idxs = np.random.choice(n_b, MAX_POINTS, replace=False)
            pts_b = pts_b[idxs];
            len_b = MAX_POINTS
        else:
            len_b = n_b
            pad = np.zeros((MAX_POINTS - n_b, pts_b.shape[1]), dtype=np.float32)
            pts_b = np.vstack([pts_b, pad])
        pts_b_tensor = torch.from_numpy(pts_b).unsqueeze(0).to(DEVICE).float()
        len_b_tensor = torch.tensor([len_b]).to(DEVICE)

        # 对接分数
        dock_a = info_a['dock']
        dock_b = info_b['dock']
        dock_avg = (dock_a + dock_b) / 2.0
        dock_tensor = torch.from_numpy(dock_avg).unsqueeze(0).to(DEVICE).float()
        tendency = create_tendency_vec(dock_tensor).squeeze(0).cpu().numpy()  # 6维

        # 对每个细胞系提取特征并预测
        cell_probs = []
        for cell_idx, cell_name in enumerate(cell_lines):
            naive = cell_naive[cell_idx]  # 3维
            naive_scaled = naive_scaler.transform(naive.reshape(1, -1)).flatten()

            # 通过RNN提取change和protein_raw
            with torch.no_grad():
                naive_tensor = torch.from_numpy(naive_scaled).unsqueeze(0).to(DEVICE).float()
                tend_tensor = torch.from_numpy(tendency).unsqueeze(0).to(DEVICE).float()
                pred = rnn_model(pts_a_tensor, len_a_tensor, pts_b_tensor, len_b_tensor,
                                 naive_tensor, tend_tensor, use_noise=False)
                change = (pred - naive_tensor).cpu().numpy().flatten()  # 3维

                e1 = rnn_model.drug_encoder(pts_a_tensor, len_a_tensor)
                e2 = rnn_model.drug_encoder(pts_b_tensor, len_b_tensor)
                p1, _ = rnn_model.protein_weight(e1)
                p2, _ = rnn_model.protein_weight(e2)
                protein_raw = torch.cat([p1, p2], dim=1).cpu().numpy().flatten()  # 274维

            # PCA降维
            protein_pca = pca.transform(protein_raw.reshape(1, -1)).flatten()  # 10维

            # 拼接22维特征
            feature = np.concatenate([naive_scaled, tendency, change, protein_pca])
            feature_scaled = scaler.transform(feature.reshape(1, -1)).flatten()

            # 软投票预测概率
            probs = np.zeros(1)
            for (name, model), w in zip(ensemble_models, ensemble_weights):
                prob = model.predict_proba(feature_scaled.reshape(1, -1))[:, 1]
                probs += w * prob
            synergy_prob = probs[0]
            cell_probs.append(synergy_prob)

        # 统计该药物组合在所有细胞系中的表现
        cell_probs = np.array(cell_probs)
        above_threshold = np.sum(cell_probs >= THRESHOLD)
        avg_prob = np.mean(cell_probs)
        max_prob = np.max(cell_probs)

        results.append({
            'drug_A': drug_a,
            'drug_B': drug_b,
            'cells_above_threshold': above_threshold,
            'total_cells': len(cell_lines),
            'ratio': above_threshold / len(cell_lines),
            'avg_prob': avg_prob,
            'max_prob': max_prob,
            'prob_list': cell_probs.tolist()
        })

    # 8. 排序：先按through细胞系数降序，再按平均概率降序
    results.sort(key=lambda x: (-x['cells_above_threshold'], -x['avg_prob']))

    # 9. 输出结果
    print("\n========== Prediction Results (Ranked) ==========")
    print(
        f"{'Rank':<5} {'Drug A':<20} {'Drug B':<20} {'Cells Above':<12} {'Ratio':<8} {'Avg Prob':<10} {'Max Prob':<10}")
    print("-" * 85)
    for rank, res in enumerate(results, 1):
        print(f"{rank:<5} {res['drug_A']:<20} {res['drug_B']:<20} {res['cells_above_threshold']:<12} "
              f"{res['ratio']:<8.3f} {res['avg_prob']:<10.4f} {res['max_prob']:<10.4f}")

    # 10. 保存详细结果到CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv('drug_combination_predictions.csv', index=False)
    print("\nDetailed results saved to 'drug_combination_predictions.csv'")

    # 11. 输出通过细胞系数最多的前20个组合的详细细胞系概率（可选）
    top_n = min(20, len(results))
    print(f"\nTop {top_n} combinations per-cell probabilities:")
    for res in results[:top_n]:
        print(f"\n{res['drug_A']} + {res['drug_B']} (cells above: {res['cells_above_threshold']}/{res['total_cells']})")
        for ci, cell_name in enumerate(cell_lines):
            prob = res['prob_list'][ci]
            flag = "✓" if prob >= THRESHOLD else " "
            print(f"  {cell_name:<15} {prob:.4f} {flag}")


if __name__ == '__main__':
    main()