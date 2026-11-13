import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# 设置全局字体为Times New Roman，字号12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 读取表达数据文件
df = pd.read_csv('liver_tumor.csv', sep=",", header=0, index_col="Case")

# 提取最后一行作为标准差数据
last_row_std = df.iloc[-1]  # 获取最后一行数据
gene_names = df.columns  # 获取基因名称

# 创建标准差DataFrame
df_biaozhun = pd.DataFrame({
    'Std_Deviation': last_row_std.values
}, index=gene_names)

# 从表达数据中移除最后一行（标准差行）
df = df.iloc[:-1]  # 移除最后一行

print("=== 数据检查 ===")
print(f"表达数据形状: {df.shape}")  # (样本数, 基因数)
print(f"标准差数据形状: {df_biaozhun.shape}")  # (基因数, 1)
print(f"基因数量: {len(gene_names)}")


def analyze_top_genes(df, df_biaozhun, n_genes=10):
    """综合分析函数"""
    # 计算变异系数(CV)
    cv_values = df.std() / df.abs().mean()

    # 创建合并数据框
    combined = pd.DataFrame({
        'Gene': df.columns,
        'Std_Deviation': df_biaozhun['Std_Deviation'].values,
        'CV': cv_values.values
    })

    print("=== 数据统计 ===")
    print(f"标准差范围: {combined['Std_Deviation'].min():.4f} - {combined['Std_Deviation'].max():.4f}")
    print(f"CV范围: {combined['CV'].min():.4f} - {combined['CV'].max():.4f}")

    # 标准化并计算综合评分
    scaler = StandardScaler()
    std_cv = combined[['Std_Deviation', 'CV']].values
    combined[['Std_scaled', 'CV_scaled']] = scaler.fit_transform(std_cv)

    # 关键修改：移除除以常数的操作，直接使用等权重加权
    combined['Composite_Score'] = 0.5 * combined['Std_scaled'] + 0.5 * combined['CV_scaled']

    return combined, combined.nlargest(n_genes, 'Composite_Score')


def create_visualizations(combined, top_genes):
    """创建可视化图表"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 散点图
    axes[0].scatter(combined['Std_Deviation'], combined['CV'], alpha=0.5, s=10, label='All genes')
    axes[0].scatter(top_genes['Std_Deviation'], top_genes['CV'], color='red', s=50, label='Top genes')
    axes[0].set_xlabel('Standard Deviation')
    axes[0].set_ylabel('Coefficient of Variation')
    axes[0].set_title('Std vs CV (Top Genes Highlighted)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 条形图
    axes[1].barh(range(len(top_genes)), top_genes['Composite_Score'], color='lightcoral')
    axes[1].set_yticks(range(len(top_genes)))
    axes[1].set_yticklabels(top_genes['Gene'])
    axes[1].set_xlabel('Composite Score')
    axes[1].set_title('Top Variable Genes')

    plt.tight_layout()
    plt.savefig('simple_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# 执行分析
try:
    combined, top_genes = analyze_top_genes(df, df_biaozhun, 10)

    # 输出结果
    print("\n=== 前10个差异表达基因 ===")
    print(top_genes[['Gene', 'Composite_Score', 'Std_Deviation', 'CV']].round(4))

    # 可视化
    create_visualizations(combined, top_genes)

    # 保存结果
    top_genes[['Gene', 'Composite_Score', 'Std_Deviation', 'CV']].to_csv('top_genes_simplified.csv', index=False)
    combined.sort_values('Composite_Score', ascending=False).to_csv('all_genes_ranking.csv', index=False)

    print("分析完成！结果已保存。")

except Exception as e:
    print(f"执行过程中出现错误: {e}")
    import traceback

    traceback.print_exc()