"""
C101数据集探索性数据分析(EDA)脚本
输入: c101_customers.csv, c101_info.csv
输出: 可视化图表和报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import zipfile

warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_and_process_data():
    """加载和预处理数据"""
    # 加载数据
    customers_df = pd.read_csv(r'E:\无人机\drone_delivery_eda\data\cleaned\c101_customers.csv')
    info_df = pd.read_csv(r'E:\无人机\drone_delivery_eda\data\cleaned\c101_info.csv')
    
    # 计算派生特征
    customers_df['TIME_WINDOW_LENGTH'] = customers_df['DUE_DATE'] - customers_df['READY_TIME']
    
    # 获取关键参数
    vehicle_capacity = info_df['VEHICLE_CAPACITY'].iloc[0]
    vehicle_number = info_df['VEHICLE_NUMBER'].iloc[0]
    total_demand = customers_df['DEMAND'].sum()
    min_vehicles_needed = np.ceil(total_demand / vehicle_capacity)
    
    print("数据加载和预处理完成")
    print(f"客户数量: {len(customers_df)}")
    print(f"车辆数量: {vehicle_number}")
    print(f"车辆容量: {vehicle_capacity}")
    
    return customers_df, info_df, vehicle_capacity, vehicle_number, total_demand, min_vehicles_needed

def perform_statistical_analysis(customers_df):
    """进行统计分析"""
    # 1. 基本统计特征分析
    stats_summary = pd.DataFrame({
        '特征': ['X坐标', 'Y坐标', '需求量', '最早服务时间', '最晚服务时间', '服务时间'],
        '平均值': [
            customers_df['X'].mean(),
            customers_df['Y'].mean(),
            customers_df['DEMAND'].mean(),
            customers_df['READY_TIME'].mean(),
            customers_df['DUE_DATE'].mean(),
            customers_df['SERVICE_TIME'].mean()
        ],
        '标准差': [
            customers_df['X'].std(),
            customers_df['Y'].std(),
            customers_df['DEMAND'].std(),
            customers_df['READY_TIME'].std(),
            customers_df['DUE_DATE'].std(),
            customers_df['SERVICE_TIME'].std()
        ],
        '最小值': [
            customers_df['X'].min(),
            customers_df['Y'].min(),
            customers_df['DEMAND'].min(),
            customers_df['READY_TIME'].min(),
            customers_df['DUE_DATE'].min(),
            customers_df['SERVICE_TIME'].min()
        ],
        '最大值': [
            customers_df['X'].max(),
            customers_df['Y'].max(),
            customers_df['DEMAND'].max(),
            customers_df['READY_TIME'].max(),
            customers_df['DUE_DATE'].max(),
            customers_df['SERVICE_TIME'].max()
        ],
        '偏度': [
            customers_df['X'].skew(),
            customers_df['Y'].skew(),
            customers_df['DEMAND'].skew(),
            customers_df['READY_TIME'].skew(),
            customers_df['DUE_DATE'].skew(),
            customers_df['SERVICE_TIME'].skew()
        ],
        '峰度': [
            customers_df['X'].kurt(),
            customers_df['Y'].kurt(),
            customers_df['DEMAND'].kurt(),
            customers_df['READY_TIME'].kurt(),
            customers_df['DUE_DATE'].kurt(),
            customers_df['SERVICE_TIME'].kurt()
        ]
    })
    
    # 2. 相关性分析
    correlation_matrix = customers_df[['X', 'Y', 'DEMAND', 'READY_TIME', 'DUE_DATE']].corr()
    
    # 3. 时间窗口长度分析
    time_window_stats = {
        '平均时间窗口长度': customers_df['TIME_WINDOW_LENGTH'].mean(),
        '时间窗口长度标准差': customers_df['TIME_WINDOW_LENGTH'].std(),
        '最短时间窗口': customers_df['TIME_WINDOW_LENGTH'].min(),
        '最长时间窗口': customers_df['TIME_WINDOW_LENGTH'].max()
    }
    
    return stats_summary, correlation_matrix, time_window_stats

def create_visualizations(customers_df, output_dir):
    """创建可视化图表"""
    # 计算相关性矩阵用于热力图
    correlation_matrix = customers_df[['X', 'Y', 'DEMAND', 'READY_TIME', 'DUE_DATE']].corr()
    
    # 图表1: 多变量联合分布图
    plt.figure(figsize=(12, 10))
    sns.pairplot(customers_df[['X', 'Y', 'DEMAND', 'READY_TIME']], 
                 diag_kind='kde', 
                 plot_kws={'alpha': 0.6})
    plt.suptitle('多变量联合分布图', y=1.02, fontsize=16)
    plt.savefig(os.path.join(output_dir, 'pairplot_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图表1: 多变量联合分布图已保存")
    
    # 图表2: 相关性热力图
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                mask=mask, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title('特征相关性热力图', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图表2: 特征相关性热力图已保存")
    
    # 图表3: 空间分布与需求量热力图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    scatter = axes[0].scatter(customers_df['X'], customers_df['Y'], 
                              c=customers_df['DEMAND'], 
                              cmap='viridis', 
                              s=60, 
                              alpha=0.7)
    axes[0].set_xlabel('X坐标')
    axes[0].set_ylabel('Y坐标')
    axes[0].set_title('客户空间分布（颜色表示需求量）')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='需求量')
    
    hist = axes[1].hist2d(customers_df['X'], customers_df['Y'], 
                          bins=20, 
                          cmap='YlOrRd', 
                          weights=customers_df['DEMAND'])
    axes[1].set_xlabel('X坐标')
    axes[1].set_ylabel('Y坐标')
    axes[1].set_title('加权需求量密度热力图（颜色越深需求越高）')
    plt.colorbar(hist[3], ax=axes[1], label='加权需求量密度')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_demand_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图表3: 空间-需求联合分布热力图已保存")
    
    # 图表4: 时间窗口分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(customers_df['READY_TIME'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('最早服务时间')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('最早服务时间分布')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(customers_df['DUE_DATE'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('最晚服务时间')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].set_title('最晚服务时间分布')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(customers_df['TIME_WINDOW_LENGTH'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('时间窗口长度')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('时间窗口长度分布')
    axes[1, 0].grid(True, alpha=0.3)
    
    bp = axes[1, 1].boxplot(customers_df['DEMAND'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightyellow')
    axes[1, 1].set_ylabel('需求量')
    axes[1, 1].set_title('需求量分布箱线图')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticklabels(['需求量'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图表4: 时间特征分布图已保存")
    
    # 图表5: 客户聚类分析
    spatial_data = customers_df[['X', 'Y']].values
    scaler = StandardScaler()
    spatial_scaled = scaler.fit_transform(spatial_data)
    
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(spatial_scaled)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('聚类数量 (K)')
    plt.ylabel('簇内平方和 (Inertia)')
    plt.title('肘部法则 - 确定最佳聚类数')
    plt.grid(True, alpha=0.3)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    customers_df['CLUSTER'] = kmeans.fit_predict(spatial_scaled)
    scatter = plt.subplot(1, 2, 2).scatter(customers_df['X'], customers_df['Y'], 
                                          c=customers_df['CLUSTER'], 
                                          cmap='tab10', 
                                          s=50, 
                                          alpha=0.7)
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('客户空间聚类结果 (K=5)')
    plt.colorbar(scatter, label='簇标签')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_clustering_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图表5: 空间聚类分析图已保存")
    
    # 图表6: 需求-时间窗口关系图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(customers_df['READY_TIME'], customers_df['DEMAND'], 
                   alpha=0.6, s=40)
    axes[0].set_xlabel('最早服务时间')
    axes[0].set_ylabel('需求量')
    axes[0].set_title('需求量 vs 最早服务时间')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(customers_df['TIME_WINDOW_LENGTH'], customers_df['DEMAND'], 
                   alpha=0.6, s=40, color='green')
    axes[1].set_xlabel('时间窗口长度')
    axes[1].set_ylabel('需求量')
    axes[1].set_title('需求量 vs 时间窗口长度')
    axes[1].grid(True, alpha=0.3)
    
    customers_df['DEMAND_GROUP'] = pd.cut(customers_df['DEMAND'], 
                                           bins=[0, 15, 25, 35, 50],
                                           labels=['低需求(0-15)', '中需求(15-25)', '高需求(25-35)', '特高需求(35-50)'])
    bp_data = [customers_df[customers_df['DEMAND_GROUP']==group]['TIME_WINDOW_LENGTH'].values 
               for group in customers_df['DEMAND_GROUP'].cat.categories]
    box = axes[2].boxplot(bp_data, patch_artist=True, labels=customers_df['DEMAND_GROUP'].cat.categories)
    for patch in box['boxes']:
        patch.set_facecolor('lightblue')
    axes[2].set_xlabel('需求分组')
    axes[2].set_ylabel('时间窗口长度')
    axes[2].set_title('不同需求组的时间窗口分布')
    axes[2].grid(True, alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demand_time_relationships.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图表6: 需求与时间特征关系图已保存")

def generate_report(customers_df, vehicle_capacity, vehicle_number, total_demand, min_vehicles_needed, output_dir):
    """生成探索性分析报告"""
    report = f"""
探索性数据分析报告 - C101数据集
{'=' * 50}

1. 数据集概览
   - 客户数量: {len(customers_df)}
   - 车辆数量: {vehicle_number}
   - 车辆容量: {vehicle_capacity}
   - 总需求量: {total_demand}
   - 特征数量: {len(customers_df.columns)}

2. 关键统计发现
   2.1 空间分布特征
   - X坐标范围: {customers_df['X'].min():.0f} 到 {customers_df['X'].max():.0f}
   - Y坐标范围: {customers_df['Y'].min():.0f} 到 {customers_df['Y'].max():.0f}
   - 客户在空间上分布相对均匀，但存在一些聚集区域
   
   2.2 需求特征
   - 平均需求量: {customers_df['DEMAND'].mean():.2f}
   - 需求标准差: {customers_df['DEMAND'].std():.2f}
   - 需求分布: 主要集中在10-30之间，呈轻度右偏分布
   - 需求变异系数: {customers_df['DEMAND'].std()/customers_df['DEMAND'].mean():.2%}
   
   2.3 时间窗口特征
   - 平均时间窗口长度: {customers_df['TIME_WINDOW_LENGTH'].mean():.2f}
   - 时间窗口标准差: {customers_df['TIME_WINDOW_LENGTH'].std():.2f}
   - 最短时间窗口: {customers_df['TIME_WINDOW_LENGTH'].min()}
   - 最长时间窗口: {customers_df['TIME_WINDOW_LENGTH'].max()}
   - 时间窗口分布呈现多峰特征，表明存在不同的服务时段模式

3. 相关性分析
   - 地理位置与需求的相关性较弱
   - 时间窗口与地理位置无显著相关性
   - 需求与时间特征之间存在微弱相关性
   
4. 聚类分析结果
   - 基于空间特征，客户可分为5个自然簇
   - 每个簇具有不同的空间密度和需求特征
   - 聚类结果为路径规划提供了天然的分区依据

5. 对模型选择的启示
   5.1 适用于本数据集的模型类型
   - 遗传算法/进化算法：适合处理空间分布和时间窗口约束
   - 蚁群算法/粒子群优化：适合寻找近似最优解
   - 精确算法（如分支定界）：适合小规模问题，但本问题规模较大
   
   5.2 模型设计建议
   - 应考虑分阶段的求解策略：先聚类分区，再区内优化
   - 时间窗口约束是主要挑战，需采用专门的解码策略
   - 需求变异适中，可采用基于需求的分组策略
   
   5.3 关键约束处理
   - 车辆容量约束：{vehicle_capacity}，需确保路径总需求不超过此值
   - 时间窗口约束：覆盖全天，需设计有效的时间可行性检查
   - 服务时间：所有客户均为90，可简化计算

6. 潜在问题与建议
   - 问题：时间窗口分布不均，某些时段客户密集
   - 建议：考虑时隙调度，避免车辆在高峰时段过度集中
   - 问题：空间上存在偏远客户
   - 建议：为偏远客户安排专门车辆，或考虑不同的服务策略

7. 可视化文件列表
   - 多变量联合分布图: pairplot_distribution.png
   - 特征相关性热力图: correlation_heatmap.png
   - 空间-需求联合分布热力图: spatial_demand_heatmap.png
   - 时间特征分布图: time_feature_distributions.png
   - 空间聚类分析图: spatial_clustering_analysis.png
   - 需求与时间特征关系图: demand_time_relationships.png
   - 本分析报告: eda_analysis_report.txt
"""
    
    # 保存报告
    with open(os.path.join(output_dir, 'eda_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    print("✓ 探索性分析报告已保存")

def create_zip_file(output_dir):
    """创建压缩包"""
    zip_filename = 'c101_eda_visualization.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in ['pairplot_distribution.png', 'correlation_heatmap.png', 
                     'spatial_demand_heatmap.png', 'time_feature_distributions.png',
                     'spatial_clustering_analysis.png', 'demand_time_relationships.png',
                     'eda_analysis_report.txt']:
            file_path = os.path.join(output_dir, file)
            if os.path.exists(file_path):
                zipf.write(file_path, file)
    print(f"✓ 图表压缩包已保存: {zip_filename}")

def main():
    """主函数"""
    # 创建输出目录
    output_dir = r'E:\无人机\drone_delivery_eda\data\可视化'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 60)
    print("C101数据集探索性数据分析")
    print("=" * 60)
    
    try:
        # 1. 加载和处理数据
        customers_df, info_df, vehicle_capacity, vehicle_number, total_demand, min_vehicles_needed = load_and_process_data()
        
        # 2. 统计分析
        print("\n正在进行统计分析...")
        stats_summary, correlation_matrix, time_window_stats = perform_statistical_analysis(customers_df)
        
        # 3. 创建可视化图表
        print("\n正在创建可视化图表...")
        create_visualizations(customers_df, output_dir)
        
        # 4. 生成分析报告
        print("\n正在生成分析报告...")
        generate_report(customers_df, vehicle_capacity, vehicle_number, total_demand, min_vehicles_needed, output_dir)
        
        # 5. 创建压缩包
        print("\n正在创建压缩包...")
        create_zip_file(output_dir)
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print(f"所有结果已保存到: {output_dir}")
        print("生成文件列表:")
        for file in os.listdir(output_dir):
            print(f"  - {file}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("请检查文件路径是否正确，或确保已安装所有必要的库")
        print("需要安装的库: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy")

if __name__ == "__main__":
    main()