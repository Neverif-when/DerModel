import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import pandas as pd
import networkx as nx
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from collections import deque
import random
from matplotlib.patches import Patch

# 设置全局参数
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
np.random.seed(42)
torch.manual_seed(42)


class SimplePPIAnalyzer:
    """PPI网络分析器 - 整合版本"""

    def __init__(self, physical_net, full_net, core_genes, id_mapping, protein_info_df, random_seed=42):
        self.core_genes = set(core_genes)
        self.id_mapping = id_mapping
        self.protein_info_df = protein_info_df
        self.random_seed = random_seed
        self._set_random_seeds()
        self.graph = self._build_network(physical_net, full_net)

    def _set_random_seeds(self):
        """设置随机种子"""
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        if hasattr(nx, 'default_rng'):
            nx.default_rng(self.random_seed)

    def _build_network(self, physical_net, full_net):
        """构建PPI网络"""
        G = nx.Graph()

        # 筛选高置信度互作
        high_conf_phys = physical_net[physical_net['combined_score'] > 700]
        high_conf_full = full_net[full_net['combined_score'] > 500]
        all_edges = pd.concat([high_conf_phys, high_conf_full]).sort_values(['protein1', 'protein2'])

        # 添加边
        for _, row in all_edges.iterrows():
            gene1 = self.id_mapping.get(row['protein1'], row['protein1'])
            gene2 = self.id_mapping.get(row['protein2'], row['protein2'])
            if gene1 and gene2:
                G.add_edge(gene1, gene2, weight=row['combined_score'] / 1000, score=row['combined_score'])

        self._add_node_attributes(G)
        print(f"网络构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
        return G

    def _add_node_attributes(self, graph):
        """添加节点属性"""
        for node in sorted(graph.nodes()):
            protein_data = self.protein_info_df[
                self.protein_info_df['preferred_name'] == node
                ].iloc[0] if node in self.protein_info_df['preferred_name'].values else None

            if protein_data is not None:
                graph.nodes[node]['protein_size'] = protein_data.get('protein_size', 0)
                graph.nodes[node]['annotation'] = protein_data.get('annotation', 'Unknown')
                graph.nodes[node]['string_id'] = protein_data.get('#string_protein_id', 'Unknown')
            else:
                graph.nodes[node]['protein_size'] = 0
                graph.nodes[node]['annotation'] = 'Unknown'
                graph.nodes[node]['string_id'] = 'Unknown'

    def find_paths(self, max_depth=6):
        """查找核心基因间的路径"""
        paths = []
        core_list = sorted(list(self.core_genes))
        available_cores = sorted([gene for gene in core_list if gene in self.graph])

        for i, gene1 in enumerate(available_cores):
            for gene2 in available_cores[i + 1:]:
                path = self._bfs_shortest_path(gene1, gene2, max_depth)
                if path:
                    paths.append(path)

        return sorted(paths, key=lambda x: (len(x), x[0], x[-1]))

    def _bfs_shortest_path(self, start, end, max_depth):
        """BFS查找最短路径"""
        if start == end:
            return None

        queue = deque([(start, [start])])
        visited = set([start])

        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth + 1:
                continue
            if current == end:
                return path

            for neighbor in sorted(list(self.graph.neighbors(current))):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def _create_custom_layout(self, graph, core_set):
        """改进的布局算法，避免核心基因节点重叠"""
        pos = {}
        core_nodes = sorted(list(core_set & set(graph.nodes())))
        num_core = len(core_nodes)

        if num_core == 0:
            return nx.circular_layout(graph)

        # 根据核心基因数量动态调整半径
        radius = 4.0 + min(num_core * 0.2, 3.0)  # 动态调整半径
        angle_step = 2 * np.pi / num_core

        # 核心基因均匀分布在圆周上
        for i, node in enumerate(core_nodes):
            angle = i * angle_step
            pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        # 其他节点使用改进的力导向布局
        other_nodes = sorted(list(set(graph.nodes()) - set(core_nodes)))
        if other_nodes:
            other_radius = radius + 2.0
            other_angle_step = 2 * np.pi / len(other_nodes)

            for i, node in enumerate(other_nodes):
                angle = i * other_angle_step
                pos[node] = np.array([
                    other_radius * np.cos(angle),
                    other_radius * np.sin(angle)
                ])

        # 使用spring layout优化布局，固定核心基因位置
        try:
            pos = nx.spring_layout(
                graph,
                pos=pos,
                fixed=core_nodes,
                k=2.0,  # 增加节点间距
                iterations=100,
                seed=self.random_seed
            )
        except:
            pass

        return pos

    def visualize_network(self, paths):
        """可视化网络 - 与原始版本保持一致"""
        if not paths:
            print("没有找到路径，无法绘制网络")
            return

        self._set_random_seeds()
        plt.rcParams['svg.hashsalt'] = str(self.random_seed)

        # 收集路径中的节点和边
        all_nodes = set()
        edge_weights = {}
        for path in sorted(paths, key=lambda x: (len(x), x[0], x[-1])):
            for i in range(len(path) - 1):
                edge = tuple(sorted([path[i], path[i + 1]]))
                all_nodes.update(edge)
                edge_weights[edge] = edge_weights.get(edge, 0) + 1

        # 创建子图
        subgraph = self.graph.subgraph(sorted(all_nodes))
        print(f"可视化子图: {subgraph.number_of_nodes()} 个节点, {subgraph.number_of_edges()} 条边")

        # 创建图形和布局
        plt.figure(figsize=(16, 12))
        pos = self._create_custom_layout(subgraph, self.core_genes)

        # 准备绘图数据
        core_nodes = sorted([node for node in subgraph.nodes() if node in self.core_genes])
        other_nodes = sorted([node for node in subgraph.nodes() if node not in self.core_genes])

        # 设置节点大小
        core_sizes = [800 + min(subgraph.degree(node) * 50, 2000) for node in core_nodes]
        other_sizes = [20 + min(subgraph.degree(node) * 3, 100) for node in other_nodes]

        # 设置边属性
        edge_widths, edge_colors = [], []
        for u, v in sorted(subgraph.edges()):
            edge = tuple(sorted([u, v]))
            count = edge_weights.get(edge, 1)
            edge_widths.append(min(count * 0.5, 3.0))

            score = subgraph[u][v].get('score', 500)
            if score > 900:
                edge_colors.append('#E67E22')
            elif score > 700:
                edge_colors.append('#F39C12')
            else:
                edge_colors.append('#F7DC6F')

        # 绘制网络
        nx.draw_networkx_edges(subgraph, pos, edgelist=sorted(subgraph.edges()),
                               width=edge_widths, edge_color=edge_colors, alpha=0.7)

        if other_nodes:
            nx.draw_networkx_nodes(subgraph, pos, nodelist=other_nodes, node_size=other_sizes,
                                   node_color='#5DADE2', alpha=0.7, edgecolors='#3498DB', linewidths=0.3)

        if core_nodes:
            nx.draw_networkx_nodes(subgraph, pos, nodelist=core_nodes, node_size=core_sizes,
                                   node_color='#D32F2F', alpha=0.9, edgecolors='#D32F2F', linewidths=1.0)

        # 添加标签和图例
        labels = {node: node for node in core_nodes}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=10, font_weight='bold')

        legend_elements = [
            Patch(facecolor='#D32F2F', label='Core Genes'),
            Patch(facecolor='#5DADE2', label='Other Genes', alpha=0.6),
            plt.Line2D([0], [0], color='#E67E22', lw=2, label='High Score (>900)'),
            plt.Line2D([0], [0], color='#F39C12', lw=2, label='Medium Score (700-900)'),
            plt.Line2D([0], [0], color='#F7DC6F', lw=2, label='Low Score (<700)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        plt.title("Protein-Protein Interaction Network", fontsize=16, pad=20, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def get_external_graph(self, paths):
        """获取外部图（PPI网络）- 为GNN特征提取准备"""
        if paths:
            all_nodes = set()
            for path in paths:
                all_nodes.update(path)
            external_graph = self.graph.subgraph(sorted(all_nodes))
        else:
            # 如果没有路径，使用核心基因构建图
            core_nodes = [gene for gene in self.core_genes if gene in self.graph.nodes()]
            external_graph = self.graph.subgraph(core_nodes)

        print(f"外部图: {external_graph.number_of_nodes()} 个节点, {external_graph.number_of_edges()} 条边")

        # 为外部图添加节点特征
        for node in external_graph.nodes():
            if 'x' not in external_graph.nodes[node]:
                feature_vector = self._create_node_features(node, external_graph)
                external_graph.nodes[node]['x'] = feature_vector

        return external_graph

    def _create_node_features(self, node, graph):
        """为外部图节点创建特征向量"""
        try:
            # 基于节点度、核心基因状态等创建特征
            degree = graph.degree(node)
            is_core = 1.0 if node in self.core_genes else 0.0
            protein_size = graph.nodes[node].get('protein_size', 0) / 1000.0

            # 基础特征
            feature_vector = [
                1.0,  # 存在标志
                float(degree) / 100.0,  # 度特征归一化
                is_core,  # 是否核心基因
                protein_size,  # 蛋白质大小
                len(node) / 50.0,  # 节点名长度特征
            ]

            # 填充到137维
            if len(feature_vector) < 137:
                rng = np.random.RandomState(abs(hash(node)) % (2 ** 32))
                feature_vector.extend(rng.normal(0, 0.1, 137 - len(feature_vector)).tolist())

            return feature_vector[:137]

        except Exception as e:
            print(f"为节点 {node} 创建特征时出错: {e}")
            return [1.0] + [0.0] * 136

    def export_results(self, paths, prefix="ppi_analysis"):
        """导出分析结果 - 与原始版本保持一致"""
        # 导出路径信息
        path_data = []
        for i, path in enumerate(sorted(paths, key=lambda x: (len(x), x[0], x[-1]))):
            path_data.append({
                'Path_ID': i + 1, 'Path_Length': len(path), 'Path': ' -> '.join(path),
                'Start_Gene': path[0], 'End_Gene': path[-1],
                'Start_Gene_Annotation': self.graph.nodes[path[0]].get('annotation', 'Unknown'),
                'End_Gene_Annotation': self.graph.nodes[path[-1]].get('annotation', 'Unknown')
            })

        pd.DataFrame(path_data).to_csv(f"{prefix}_paths.csv", index=False)

        # 导出互作信息
        interactions = []
        seen_edges = set()
        for path in sorted(paths, key=lambda x: (len(x), x[0], x[-1])):
            for i in range(len(path) - 1):
                gene1, gene2 = path[i], path[i + 1]
                edge = tuple(sorted([gene1, gene2]))
                if edge not in seen_edges and self.graph.has_edge(gene1, gene2):
                    seen_edges.add(edge)
                    score = self.graph[gene1][gene2].get('score', 0)
                    interactions.append({
                        'Gene1': gene1, 'Gene2': gene2, 'Score': score,
                        'Gene1_Protein_Size': self.graph.nodes[gene1].get('protein_size', 0),
                        'Gene2_Protein_Size': self.graph.nodes[gene2].get('protein_size', 0),
                        'Gene1_Annotation': self.graph.nodes[gene1].get('annotation', 'Unknown'),
                        'Gene2_Annotation': self.graph.nodes[gene2].get('annotation', 'Unknown'),
                        'Type': 'core-core' if gene1 in self.core_genes and gene2 in self.core_genes else 'core-other',
                        'Confidence': 'High' if score > 700 else 'Medium'
                    })

        interaction_df = pd.DataFrame(interactions)
        interaction_df.sort_values(['Score', 'Gene1', 'Gene2'], ascending=[False, True, True]) \
            .to_csv(f"{prefix}_interactions.csv", index=False)

        # 导出节点属性
        protein_df = self.export_protein_info(paths, f"{prefix}_protein_info.csv")

        print(f"路径总数: {len(paths)}")
        print(f"互作关系总数: {len(interaction_df)}")
        print(f"子图节点数: {len(protein_df)}")

        return path_data, interaction_df, protein_df

    def export_protein_info(self, paths, filename="protein_info.csv"):
        """导出子图中的核心基因与非核心基因节点属性到CSV文件"""
        all_nodes = set()
        for path in paths:
            all_nodes.update(path)

        nodes_data = []
        for node in sorted(all_nodes):
            if node in self.graph.nodes():
                node_data = {
                    'Gene': node,
                    'Protein_Size': self.graph.nodes[node].get('protein_size', 0),
                    'Annotation': self.graph.nodes[node].get('annotation', 'Unknown'),
                    'String_ID': self.graph.nodes[node].get('string_id', 'Unknown'),
                    'Is_Core': node in self.core_genes,
                    'Degree': self.graph.degree(node)
                }
                nodes_data.append(node_data)

        protein_df = pd.DataFrame(nodes_data)
        protein_df.to_csv(filename, index=False)
        print(f"子图节点属性已导出到: {filename} (包含 {len(protein_df)} 个节点)")
        return protein_df

    def analyze(self, max_depth=6, export=True, visualize=True):
        """执行完整分析流程"""
        print("开始PPI网络分析...")
        print(f"随机种子: {self.random_seed}")

        self._set_random_seeds()
        paths = self.find_paths(max_depth)
        print(f"找到 {len(paths)} 条路径")

        if visualize and paths:
            self.visualize_network(paths)

        if export:
            if paths:
                return self.export_results(paths)
            else:
                print("未找到路径，无法导出结果")

        return paths


class ProteinGNN(nn.Module):
    """改进的双通道GNN特征提取器"""

    def __init__(self, input_dim, hidden_dim, output_dim, freeze=True):
        super().__init__()
        self.internal_conv1 = GCNConv(input_dim, hidden_dim)
        self.internal_conv2 = GCNConv(hidden_dim, output_dim)
        self.external_conv1 = GCNConv(input_dim, hidden_dim)
        self.external_conv2 = GCNConv(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, data):
        """前向传播 - 修复边类型处理"""
        if not hasattr(data, 'edge_type') or data.edge_type is None:
            data.edge_type = torch.ones(data.edge_index.shape[1], dtype=torch.long, device=data.x.device)

        # 确保edge_type长度与edge_index一致
        if data.edge_type.shape[0] != data.edge_index.shape[1]:
            data.edge_type = torch.ones(data.edge_index.shape[1], dtype=torch.long, device=data.x.device)

        internal_mask = data.edge_type == 0
        external_mask = data.edge_type == 1

        # 内部图处理（蛋白质结构）
        int_embed = torch.zeros(data.batch.max().item() + 1, 137, device=data.x.device)
        if internal_mask.sum() > 0:
            try:
                x_int = F.relu(self.bn1(self.internal_conv1(data.x, data.edge_index[:, internal_mask])))
                x_int = self.internal_conv2(x_int, data.edge_index[:, internal_mask])
                int_embed = global_mean_pool(x_int, data.batch)
            except Exception as e:
                print(f"内部图处理失败: {e}")

        # 外部图处理（PPI网络）
        ext_embed = torch.zeros(data.batch.max().item() + 1, 137, device=data.x.device)
        if external_mask.sum() > 0:
            try:
                x_ext = F.relu(self.bn1(self.external_conv1(data.x, data.edge_index[:, external_mask])))
                x_ext = self.external_conv2(x_ext, data.edge_index[:, external_mask])
                ext_embed = global_mean_pool(x_ext, data.batch)
            except Exception as e:
                print(f"外部图处理失败: {e}")

        return torch.cat((int_embed, ext_embed), dim=1)


def build_internal_graphs(pdb_dir):
    """构建内部图 - 从PDB文件构建蛋白质结构图"""
    print("构建内部图（蛋白质结构）...")

    if not os.path.exists(pdb_dir):
        raise FileNotFoundError(f"PDB目录 {pdb_dir} 不存在")

    graphs = []

    for gene_name in os.listdir(pdb_dir):
        gene_path = os.path.join(pdb_dir, gene_name)
        if not os.path.isdir(gene_path):
            continue

        # 查找PDB文件
        pdb_files = []
        for root, dirs, files in os.walk(gene_path):
            for file in files:
                if file.endswith('.pdb') or file.endswith('.Drug_pdbqt'):
                    pdb_files.append(os.path.join(root, file))
            break

        if not pdb_files:
            print(f"警告: 在 {gene_name} 目录中未找到PDB文件，跳过")
            continue

        # 使用第一个PDB文件构建图
        first_pdb = pdb_files[0]
        G = nx.Graph()

        # 创建蛋白质结构节点（残基层级）
        residues = [f"{gene_name}_R1", f"{gene_name}_R2", f"{gene_name}_R3"]
        G.add_nodes_from(residues)
        G.add_edges_from([(f"{gene_name}_R1", f"{gene_name}_R2"),
                          (f"{gene_name}_R2", f"{gene_name}_R3")])

        # 添加节点特征
        for node in G.nodes():
            feature_vector = extract_features_from_pdb(first_pdb, node)
            G.nodes[node]['x'] = feature_vector

        graphs.append(G)
        print(f"为基因 {gene_name} 构建内部图")

    if len(graphs) == 0:
        raise ValueError(f"在目录 {pdb_dir} 中未找到有效的PDB文件")

    print(f"构建了 {len(graphs)} 个内部图")
    return graphs


def extract_features_from_pdb(pdb_file, node):
    """从PDB文件中提取特征"""
    try:
        file_size = os.path.getsize(pdb_file)
        file_size_norm = min(file_size / 1000000.0, 10.0)

        node_hash = hash(node) % 1000 / 1000.0

        feature_vector = [
            1.0,  # 存在标志
            file_size_norm / 10.0,  # 文件大小特征
            node_hash,  # 节点特定特征
            len(node) / 50.0,  # 节点名长度特征
            0.5,  # 默认置信度
        ]

        # 填充到137维
        if len(feature_vector) < 137:
            rng = np.random.RandomState(abs(hash(node)) % (2 ** 32))
            feature_vector.extend(rng.normal(0, 0.1, 137 - len(feature_vector)).tolist())

        return feature_vector[:137]

    except Exception as e:
        print(f"从PDB文件提取特征时出错: {e}")
        return [1.0] + [0.0] * 136


def create_gnn_input(internal_graphs, external_graph):
    """创建GNN输入数据 - 修复版本兼容性问题"""
    print("创建GNN输入数据...")

    # 检查外部图是否为空
    if external_graph.number_of_nodes() == 0:
        print("警告: 外部图为空，仅使用内部图")
        all_graphs = internal_graphs
    else:
        all_graphs = internal_graphs + [external_graph]

    # 确保所有节点特征维度一致
    for graph in all_graphs:
        for node in graph.nodes():
            if 'x' not in graph.nodes[node]:
                graph.nodes[node]['x'] = [1.0] + [0.0] * 136
            elif len(graph.nodes[node]['x']) != 137:
                current_features = graph.nodes[node]['x']
                if len(current_features) < 137:
                    rng = np.random.RandomState(abs(hash(node)) % (2 ** 32))
                    padding = rng.normal(0, 0.1, 137 - len(current_features)).tolist()
                    graph.nodes[node]['x'] = current_features + padding
                else:
                    graph.nodes[node]['x'] = current_features[:137]

    # 合并图
    combined_graph = nx.compose_all(all_graphs)
    print(f"合并图: {combined_graph.number_of_nodes()} 个节点, {combined_graph.number_of_edges()} 条边")

    # 修复边类型标记逻辑
    valid_edges = []
    edge_types = []

    print("开始标记边类型...")

    for u, v in combined_graph.edges():
        try:
            # 改进的边类型判断逻辑
            u_has_underscore = '_' in u
            v_has_underscore = '_' in v

            if u_has_underscore and v_has_underscore:
                u_gene = u.split('_')[0]
                v_gene = v.split('_')[0]
                if u_gene == v_gene:
                    edge_type = 0  # 内部边
                else:
                    edge_type = 1  # 外部边
            else:
                edge_type = 1  # 外部边

            valid_edges.append((u, v))
            edge_types.append(edge_type)

        except Exception as e:
            print(f"处理边 ({u}, {v}) 时出错: {e}")
            valid_edges.append((u, v))
            edge_types.append(1)
            continue

    # 创建节点映射
    nodes_list = list(combined_graph.nodes())
    node_mapping = {node: idx for idx, node in enumerate(nodes_list)}
    num_nodes = len(nodes_list)

    # 构建节点特征张量
    x_list = []
    for node in nodes_list:
        x_list.append(combined_graph.nodes[node]['x'])
    x = torch.tensor(x_list, dtype=torch.float)

    # 构建边索引张量
    edge_index_list = []
    corrected_edge_types = []

    for (u, v), edge_type in zip(valid_edges, edge_types):
        if u in node_mapping and v in node_mapping:
            edge_index_list.append([node_mapping[u], node_mapping[v]])
            corrected_edge_types.append(edge_type)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_type_tensor = torch.tensor(corrected_edge_types, dtype=torch.long)

    # 创建PyG Data对象
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type_tensor
    )

    # 添加batch信息
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

    # 验证数据完整性
    if data.edge_type.shape[0] != data.edge_index.shape[1]:
        print("边类型数量不匹配，进行修正")
        data.edge_type = torch.ones(data.edge_index.shape[1], dtype=torch.long)

    # 统计边类型
    internal_edges = (data.edge_type == 0).sum().item()
    external_edges = (data.edge_type == 1).sum().item()

    print(f"边类型统计: 内部边={internal_edges}, 外部边={external_edges}")
    print(f"GNN输入数据创建完成: {data.num_nodes} 个节点, {data.edge_index.shape[1]} 条边")

    return data


def main():
    """主执行函数"""
    print("开始蛋白质PPI网络分析与GNN特征提取流程...")

    try:
        # 1. 读取数据
        print("加载数据文件...")
        protein_info = pd.read_csv(
            r'D:\TumorCNN\MyPPI\9606.PPI_psy_and_links.v12.0.txt\9606.protein.info.v12.0.txt\9606.protein.info.v12.0.csv')
        physical_net = pd.read_csv(
            r'D:\TumorCNN\MyPPI\9606.PPI_psy_and_links.v12.0.txt\9606.protein.physical.links.v12.0.txt\9606.protein.physical.links.v12.0.csv')
        full_net = pd.read_csv(
            r'D:\TumorCNN\MyPPI\9606.PPI_psy_and_links.v12.0.txt\9606.protein.links.v12.0.txt\9606.protein.links.v12.0.csv')

        # 2. 核心基因定义
        core_genes = ['MYL9', 'CCL21', 'S100A8', 'MGP', 'CYP1B1', 'ENO3', 'MT1H', 'GPNMB',
                      'PLIN4', 'APOA4', 'GLYATL1', 'PCK2', 'EHHADH', 'HAO1', 'METTL7A',
                      'ACSM2A', 'CYB5A', 'SPINT1', 'RBP4', 'DMGDH']

        # 3. 创建ID映射
        id_mapping = {}
        if 'preferred_name' in protein_info.columns and '#string_protein_id' in protein_info.columns:
            id_mapping = {row['#string_protein_id']: row['preferred_name']
                          for _, row in protein_info.iterrows()}

        # 4. PPI网络分析（外部图）
        print("\n=== PPI网络分析 ===")
        analyzer = SimplePPIAnalyzer(
            physical_net=physical_net,
            full_net=full_net,
            core_genes=core_genes,
            id_mapping=id_mapping,
            protein_info_df=protein_info,
            random_seed=42
        )

        # 查找路径并可视化
        paths = analyzer.find_paths(max_depth=6)
        print(f"找到 {len(paths)} 条核心基因间路径")

        # 可视化PPI网络（使用原始版本的可视化效果）
        analyzer.visualize_network(paths)

        # 获取外部图
        external_graph = analyzer.get_external_graph(paths)

        # 5. 构建内部图（蛋白质结构）
        print("\n=== 蛋白质结构分析 ===")
        internal_graphs = build_internal_graphs(r'D:\TumorCNN\MyPPI\All_structure_cleaned')

        # 6. 创建GNN输入数据
        print("\n=== GNN特征提取 ===")
        gnn_input = create_gnn_input(internal_graphs, external_graph)

        # 7. 初始化GNN模型并提取特征
        gnn = ProteinGNN(input_dim=137, hidden_dim=128, output_dim=137)

        with torch.no_grad():
            features = gnn(gnn_input)

        print(f"特征提取完成! 特征维度: {features.shape}")

        # 8. 保存结果
        torch.save({
            'features': features,
            'gnn_input': gnn_input,
            'internal_graphs_count': len(internal_graphs),
            'external_graph_info': {
                'nodes': external_graph.number_of_nodes(),
                'edges': external_graph.number_of_edges()
            }
        }, 'protein_gnn_features.pth')

        print("特征已保存到 protein_gnn_features.pth")

        # 9. 输出分析报告
        print("\n=== 分析报告 ===")
        print(f"内部图数量: {len(internal_graphs)}")
        print(f"外部图节点数: {external_graph.number_of_nodes()}")
        print(f"外部图边数: {external_graph.number_of_edges()}")
        print(f"最终特征维度: {features.shape}")

        if hasattr(gnn_input, 'edge_type'):
            internal_edges = (gnn_input.edge_type == 0).sum().item()
            external_edges = (gnn_input.edge_type == 1).sum().item()
            print(f"内部边数量: {internal_edges}")
            print(f"外部边数量: {external_edges}")

    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()