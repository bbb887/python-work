import numpy as np
import pandas as pd
import os

def generate_distance_matrix(txt_file):
    coords = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 提取坐标（Solomon标准：第10行开始为节点数据）
    for line in lines[9:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            x = float(parts[1])
            y = float(parts[2])
            coords.append([x, y])

    coords = np.array(coords)
    n = coords.shape[0]

    # 计算欧式距离矩阵（无人机直线飞行）
    dist_matrix = np.sqrt((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    return dist_matrix, n
    dist_matrix = np.floor(dist_matrix).astype(int)
def batch_process_solomon(folder_path):

    # 获取文件夹内所有txt文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not txt_files:
        print(" 文件夹中未找到任何 .txt 文件！")
        return

    print(f" 处理...\n")

    # 遍历处理每个文件
    for idx, filename in enumerate(txt_files, 1):
        file_path = os.path.join(folder_path, filename)
        
        # 生成矩阵
        dist_matrix, node_num = generate_distance_matrix(file_path)
        
        # 保存CSV
        save_name= os.path.splitext(filename)[0] + "_距离矩阵.csv"
        save_path = os.path.join(folder_path, save_name)
        pd.DataFrame(dist_matrix).to_csv(save_path, index=False, header=False)
        
        print(f"【{idx}/{len(txt_files)}】{filename}")
        print(f"   节点总数：{node_num} | 矩阵尺寸：{dist_matrix.shape}")
        print(f"   已保存：{save_name}\n")

    print("处理完成")


if __name__ == "__main__":
    # 填写数据集文件夹路径（放置txt的文件夹）

    SOLOMON_FOLDER = "./raw/Solomon(1)"#绝对路径
    
    # 批量执行
    batch_process_solomon(SOLOMON_FOLDER)