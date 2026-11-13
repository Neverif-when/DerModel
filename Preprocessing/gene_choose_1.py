import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def main():
    """Main function - load data, analyze hub genes and create visualizations"""
    # Set Times New Roman font with size 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False  # Proper minus sign display

    # Set font properties for different elements
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 12

    # Set clean color scheme
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # 1. Load data
    df = pd.read_csv('HierCluster.2025-09-22.tsv', sep='\t', index_col='Case')
    print(f"Data loaded successfully: {df.shape[0]} samples, {df.shape[1]} genes")

    # 2. Analyze hub genes
    print("\nAnalyzing hub genes...")
    corr_matrix = df.corr().abs()  # 计算所有基因间的相关系数矩阵，并取绝对值

    # Calculate connectivity for each gene
    connectivity = {gene: (corr_matrix[gene] > 0.6).sum() - 1 for gene in df.columns}  # 找出与该基因相关系数绝对值大于0.6的所有基因
    hub_genes = sorted(connectivity, key=connectivity.get, reverse=True)[:10]

    # 创建只包含相关系数大于0.6的矩阵
    # 复制原始相关系数矩阵
    filtered_corr = corr_matrix.copy()

    # 将小于等于0.6的值设为NaN（这样在CSV中会显示为空）
    filtered_corr[filtered_corr <= 0.6] = np.nan

    # 保存为CSV文件
    filtered_corr.to_csv('filtered_correlation_matrix.csv', float_format='%.4f')
    print("Saved filtered correlation matrix to 'filtered_correlation_matrix.csv'")

    print("Top hub genes (highest connectivity):")
    for gene in hub_genes:
        print(f"  {gene}: Connectivity = {connectivity[gene]}")

    # 3. Create clean visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hub Genes Analysis', fontsize=12, fontweight='bold', fontfamily='Times New Roman')

    # Plot 1: Connectivity bar chart
    hub_conn = {gene: connectivity[gene] for gene in hub_genes}
    genes, conn_values = zip(*sorted(hub_conn.items(), key=lambda x: x[1], reverse=True))

    axes[0, 0].bar(genes, conn_values, color=palette[0])
    axes[0, 0].set_title('Top 10 Hub Genes Connectivity', fontsize=12, fontfamily='Times New Roman')
    axes[0, 0].set_ylabel('Number of Connections (r > 0.6)', fontsize=12, fontfamily='Times New Roman')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Set tick labels font
    for label in axes[0, 0].get_xticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(12)
    for label in axes[0, 0].get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(12)

    # Add value labels
    for i, v in enumerate(conn_values):
        axes[0, 0].text(i, v + 0.5, str(v), ha='center', fontsize=12, fontfamily='Times New Roman')

    # Plot 2: Correlation coefficient distribution
    corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]

    axes[0, 1].hist(corr_values, bins=50, color=palette[1], alpha=0.6)
    axes[0, 1].set_title('Correlation Coefficient Distribution', fontsize=12, fontfamily='Times New Roman')
    axes[0, 1].set_xlabel('Absolute Correlation Coefficient', fontsize=12, fontfamily='Times New Roman')
    axes[0, 1].set_ylabel('Frequency', fontsize=12, fontfamily='Times New Roman')

    # Set tick labels font
    for label in axes[0, 1].get_xticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(12)
    for label in axes[0, 1].get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(12)

    # Add threshold line
    axes[0, 1].axvline(x=0.6, color='red', linestyle='--', alpha=0.7, label='Threshold (r=0.6)')
    legend = axes[0, 1].legend()
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman')
        text.set_fontsize(12)

    # Add statistics
    stats_text = f'Median: {np.median(corr_values):.2f}\nMean: {np.mean(corr_values):.2f}\nGene pairs: {len(corr_values):,}'
    axes[0, 1].text(0.05, 0.95, stats_text, transform=axes[0, 1].transAxes,
                    verticalalignment='top', fontsize=12, fontfamily='Times New Roman',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    # Plot 3: Hub genes correlation network (heatmap) - CHANGED TO RED COLOR SCHEME
    hub_corr = corr_matrix.loc[hub_genes, hub_genes]

    hub_corr = corr_matrix.loc[hub_genes, hub_genes]
    hub_corr.to_csv('hub_genes_correlation.csv', float_format='%.4f')
    print("\nHub genes correlation matrix saved as 'hub_genes_correlation.csv'")

    # Create heatmap with red color scheme
    sns.heatmap(
        hub_corr,
        cmap='Reds',  # Changed from 'Blues' to 'Reds'
        annot=False,
        center=0,
        square=True,
        cbar_kws={"shrink": 0.7},
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Hub Genes Correlation Heatmap', fontsize=12, fontfamily='Times New Roman')

    # Set heatmap tick labels font
    axes[1, 0].set_xticklabels(hub_genes, rotation=45, ha='right', fontfamily='Times New Roman', fontsize=12)
    axes[1, 0].set_yticklabels(hub_genes, fontfamily='Times New Roman', fontsize=12)

    # Plot 4: Connectivity vs average correlation scatter plot
    avg_corr = [corr_matrix[gene].mean() for gene in df.columns]
    axes[1, 1].scatter(avg_corr, list(connectivity.values()), alpha=0.6, color=palette[0])

    # Highlight hub genes
    hub_avg_corr = [corr_matrix[gene].mean() for gene in hub_genes]
    axes[1, 1].scatter(hub_avg_corr, [connectivity[gene] for gene in hub_genes],
                       color=palette[3], s=50, label='Hub Genes')

    axes[1, 1].set_title('Gene Connectivity vs Average Correlation', fontsize=12, fontfamily='Times New Roman')
    axes[1, 1].set_xlabel('Average Correlation Coefficient', fontsize=12, fontfamily='Times New Roman')
    axes[1, 1].set_ylabel('Connectivity', fontsize=12, fontfamily='Times New Roman')

    # Set tick labels font
    for label in axes[1, 1].get_xticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(12)
    for label in axes[1, 1].get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(12)

    legend = axes[1, 1].legend()
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman')
        text.set_fontsize(12)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save visualization
    plt.savefig('hub_genes_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("\nHub genes analysis plot saved as 'hub_genes_analysis.png'")
    plt.show()


if __name__ == "__main__":
    main()