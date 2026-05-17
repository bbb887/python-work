import os
import pandas as pd
from pathlib import Path

def process_solomon_file(input_file: str, output_folder: str):
    """
    读取 Solomon 格式 TXT 文件，提取车辆信息和客户数据，保存为 CSV
    """
    # 读取文件所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 初始化数据容器
    vehicle_num = 0
    vehicle_cap = 0
    depot_data = {}
    customer_data = []

    # 状态标记，用于追踪当前读到哪一部分
    current_section = None
    
    for line in lines:
        stripped_line = line.strip()
        
        # 1. 识别问题名称和车辆信息块
        if "VEHICLE" in stripped_line:
            current_section = "VEHICLE"
            continue
            
        # 2. 识别客户信息块
        if "CUSTOMER" in stripped_line:
            current_section = "CUSTOMER"
            continue
            
        # 3. 如果是空行，跳过
        if not stripped_line:
            continue

        # 4. 处理车辆信息 (NUMBER, CAPACITY)
        if current_section == "VEHICLE":
            parts = stripped_line.split()
            # 确保分割后至少有两个元素，且不是标题行
            if len(parts) >= 2 and parts[0].isdigit():
                vehicle_num = int(parts[0])
                vehicle_cap = int(parts[1])
                current_section = None # 读取完车辆信息，重置状态

        # 5. 处理客户数据
        elif current_section == "CUSTOMER":
            # 检查是否是表头行 (包含 CUST NO.)
            if "CUST NO." in stripped_line or "XCOORD" in stripped_line:
                continue # 跳过表头
                
            parts = stripped_line.split()
            
            # 确保数据行有效（至少有7个数据列）
            if len(parts) >= 7:
                try:
                    cust_no = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    demand = int(parts[3])
                    ready_time = int(parts[4])
                    due_date = int(parts[5])
                    service_time = int(parts[6])
                    
                    # 0号是仓库(Depot)，1-100是客户
                    if cust_no == 0:
                        depot_data = {
                            'CUST_NO': cust_no, 'X': x, 'Y': y, 'DEMAND': demand,
                            'READY_TIME': ready_time, 'DUE_DATE': due_date, 'SERVICE_TIME': service_time
                        }
                    else:
                        customer_data.append({
                            'CUST_NO': cust_no, 'X': x, 'Y': y, 'DEMAND': demand,
                            'READY_TIME': ready_time, 'DUE_DATE': due_date, 'SERVICE_TIME': service_time
                        })
                except ValueError:
                    # 如果某一行转换失败，跳过它
                    continue

    # --- 生成 CSV ---
    
    # 1. 生成客户数据 CSV
    if customer_data:
        df_customers = pd.DataFrame(customer_data)
        # 定义列顺序，使其更规范
        cols = ['CUST_NO', 'X', 'Y', 'DEMAND', 'READY_TIME', 'DUE_DATE', 'SERVICE_TIME']
        df_customers = df_customers[cols]
        
        # 获取文件名作为输出文件名
        base_name = Path(input_file).stem
        output_csv_path = os.path.join(output_folder, f"{base_name}_customers.csv")
        df_customers.to_csv(output_csv_path, index=False)
        
        print(f"✅ 成功处理: {base_name} -> 客户数据已保存至 {output_csv_path}")
        print(f"   车辆数: {vehicle_num}, 容量: {vehicle_cap}, 客户数: {len(df_customers)}")
        
    # 2. 生成汇总信息 CSV (可选，方便查看整体参数)
    info_data = {
        'PROBLEM_NAME': Path(input_file).stem,
        'VEHICLE_NUMBER': vehicle_num,
        'VEHICLE_CAPACITY': vehicle_cap,
        'DEPOT_X': depot_data.get('X', 0),
        'DEPOT_Y': depot_data.get('Y', 0),
        'DEPOT_READY_TIME': depot_data.get('READY_TIME', 0),
        'DEPOT_DUE_DATE': depot_data.get('DUE_DATE', 0)
    }
    df_info = pd.DataFrame([info_data])
    info_csv_path = os.path.join(output_folder, f"{base_name}_info.csv")
    df_info.to_csv(info_csv_path, index=False)


def main():
    # 源数据文件夹路径
    source_folder = r"E:\无人机\drone_delivery_eda\data\raw\Solomon(1)"
    
    # 目标输出文件夹路径
    target_folder = r"E:\无人机\drone_delivery_eda\data\cleaned"
    
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"📂 已创建目标文件夹: {target_folder}")
    
    print(f"🔍 扫描源文件夹: {source_folder}")
    files = os.listdir(source_folder)
    txt_files = [f for f in files if f.endswith('.txt')]
    
    print(f"📝 找到 {len(txt_files)} 个txt文件需要处理")
    print("==========================================")
    
    if not txt_files:
        print("❌ 未找到任何 .txt 文件，请检查源文件夹路径。")
        return

    for txt_file in txt_files:
        full_path = os.path.join(source_folder, txt_file)
        try:
            process_solomon_file(full_path, target_folder)
        except Exception as e:
            print(f"❌ 处理文件 {txt_file} 时发生严重错误: {e}")
            
    print("==========================================")
    print("🎉 所有文件处理完毕！")

if __name__ == "__main__":
    main()