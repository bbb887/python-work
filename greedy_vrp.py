import pandas as pd
import numpy as np
import math
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt

# ================== 路径配置 ==================
BASE_DIR = r"E:\无人机\drone_delivery_eda"
DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned")
RESULT_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "log")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CUSTOMER_FILE = os.path.join(DATA_DIR, "c101_customers.csv")
INFO_FILE = os.path.join(DATA_DIR, "c101_info.csv")
RESULT_FILE = os.path.join(RESULT_DIR, "greedy_metrics.csv")
ROUTE_FIG = os.path.join(RESULT_DIR, "greedy_routes.png")
LOG_FILE = os.path.join(LOG_DIR, "greedy_run.log")

# ================== 参数 ==================
DISTANCE_WEIGHT = 1
WT = 20          # 超时惩罚权重
WP = 1000        # 硬约束惩罚权重

# ================== 日志 ==================
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ================== 距离 ==================
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ================== 路径可视化 ==================
def plot_routes(customers, routes, depot, save_path):
    plt.figure(figsize=(12, 10))

    # 绘制客户点
    plt.scatter(customers["x"], customers["y"], c="black", s=30, label="Customers")
    plt.scatter(depot[0], depot[1], c="red", s=200, marker="s", label="Depot", zorder=5)

    # 使用更多颜色
    colors = plt.cm.tab20.colors + plt.cm.Set3.colors

    for i, route in enumerate(routes):
        if not route:
            continue
            
        color = colors[i % len(colors)]
        
        # 获取路线上的客户点
        route_customers = customers[customers["customer_id"].isin(route)]
        
        if len(route_customers) == 0:
            continue
            
        # 按路线顺序排列
        route_customers = route_customers.set_index("customer_id").loc[route].reset_index()
        
        # 从仓库到第一个客户
        plt.plot([depot[0], route_customers.iloc[0]["x"]],
                 [depot[1], route_customers.iloc[0]["y"]], 
                 color=color, linewidth=1.5, alpha=0.7)
        
        # 客户之间的路线
        for j in range(len(route_customers) - 1):
            plt.plot([route_customers.iloc[j]["x"], route_customers.iloc[j + 1]["x"]],
                     [route_customers.iloc[j]["y"], route_customers.iloc[j + 1]["y"]],
                     color=color, linewidth=1.5, alpha=0.7)
        
        # 最后一个客户返回仓库
        plt.plot([route_customers.iloc[-1]["x"], depot[0]],
                 [route_customers.iloc[-1]["y"], depot[1]],
                 color=color, linewidth=1.5, alpha=0.7)

    plt.title("Multi-UAV Greedy VRP Routes", fontsize=14)
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    log(f"路线图已保存至: {save_path}")

# ================== 多无人机贪心算法 ==================
def greedy_vrp_multi():
    start_time = time.time()
    log("=" * 60)
    log("开始运行多无人机贪心算法")
    log("=" * 60)
    
    # 读取数据
    log("正在读取数据...")
    customers = pd.read_csv(CUSTOMER_FILE)
    info = pd.read_csv(INFO_FILE).iloc[0]

    # 重命名列
    customers.rename(columns={
        "CUST_NO": "customer_id",
        "X": "x",
        "Y": "y",
        "DEMAND": "demand",
        "READY_TIME": "ready_time",
        "DUE_DATE": "due_date",
        "SERVICE_TIME": "service_time"
    }, inplace=True)

    depot = (info["DEPOT_X"], info["DEPOT_Y"])
    capacity = info["VEHICLE_CAPACITY"]
    num_vehicles = info["VEHICLE_NUMBER"]

    log(f"客户数量: {len(customers)}")
    log(f"无人机数量: {num_vehicles}")
    log(f"无人机容量: {capacity}")
    log(f"仓库位置: {depot}")
    
    # 初始化未服务客户
    unserved = customers.copy()
    
    # 初始化所有无人机的状态
    routes = [[] for _ in range(num_vehicles)]
    positions = [depot for _ in range(num_vehicles)]
    loads = [0 for _ in range(num_vehicles)]
    times = [0 for _ in range(num_vehicles)]
    
    total_distance = 0
    total_time_penalty = 0
    total_distance_no_penalty = 0  # 记录不含惩罚的原始距离
    overdue_count = 0
    constraint_violations = 0
    
    iteration = 0
    last_unserved_count = len(unserved)
    
    log("开始分配客户...")
    
    # 主循环：分配所有客户
    while not unserved.empty:
        iteration += 1
        
        # 存储所有可能的候选 (预估成本, 无人机索引, 客户, 到达时间)
        all_candidates = []
        
        # 为每架有容量的无人机评估所有未服务客户
        for v_idx in range(num_vehicles):
            # 如果无人机已满（容量使用率>95%），跳过
            if loads[v_idx] >= capacity * 0.95:
                continue
            
            # 评估所有未服务客户
            for _, row in unserved.iterrows():
                # 计算距离
                d = dist(positions[v_idx], (row["x"], row["y"]))
                arrival = times[v_idx] + d
                
                # 容量检查（硬约束）
                if loads[v_idx] + row["demand"] > capacity:
                    continue
                
                # 预估时间惩罚（软约束）
                est_time_penalty = 0
                if arrival > row["due_date"]:
                    est_time_penalty = WT * (arrival - row["due_date"])
                
                # 总预估成本（距离 + 时间惩罚）
                estimated_cost = d + est_time_penalty
                
                all_candidates.append((estimated_cost, v_idx, row, arrival))
        
        # 如果没有候选（所有无人机都满了），强制所有无人机返回仓库
        if not all_candidates:
            log(f"  迭代 {iteration}: 所有无人机已满，强制返回仓库")
            for v_idx in range(num_vehicles):
                if loads[v_idx] > 0:
                    # 返回仓库
                    d_back = dist(positions[v_idx], depot)
                    total_distance += d_back
                    total_distance_no_penalty += d_back
                    times[v_idx] += d_back
                    positions[v_idx] = depot
                    
                    log(f"    无人机 {v_idx+1}: 返回仓库, 已服务 {len(routes[v_idx])} 个客户, 总载重 {loads[v_idx]}")
                    
                    # 重置状态，准备下一轮
                    loads[v_idx] = 0
                    # 注意：不清空路线，保持历史记录
            continue
        
        # 选择全局最优的分配（成本最低）
        all_candidates.sort(key=lambda x: x[0])
        est_cost, v_idx, cust, arrival = all_candidates[0]
        
        # 记录原始距离（不含惩罚）
        d_to_cust = dist(positions[v_idx], (cust["x"], cust["y"]))
        total_distance_no_penalty += d_to_cust
        
        # 检查并记录超时惩罚
        if arrival > cust["due_date"]:
            overdue_count += 1
            penalty = WT * (arrival - cust["due_date"])
            total_time_penalty += penalty
        
        # 等待到最早服务时间
        if arrival < cust["ready_time"]:
            arrival = cust["ready_time"]
        
        # 更新无人机状态
        times[v_idx] = arrival + cust["service_time"]
        loads[v_idx] += cust["demand"]
        positions[v_idx] = (cust["x"], cust["y"])
        
        # 记录路线
        routes[v_idx].append(cust["customer_id"])
        
        # 移除已服务的客户
        unserved = unserved[unserved["customer_id"] != cust["customer_id"]]
        
        # 记录总距离（仅当实际飞行时）
        total_distance += d_to_cust
        
        # 每100次迭代输出进度
        if iteration % 100 == 0:
            served_count = len(customers) - len(unserved)
            log(f"  迭代 {iteration}: 已服务 {served_count}/{len(customers)} 客户, "
                f"当前无人机 {v_idx+1} 服务客户 {cust['customer_id']}, "
                f"当前距离: {total_distance:.2f}")
    
    # 所有客户分配完毕，所有无人机返回仓库
    log("所有客户已分配，无人机返回仓库...")
    for v_idx in range(num_vehicles):
        if routes[v_idx]:  # 只处理有任务的无人机
            d_back = dist(positions[v_idx], depot)
            total_distance += d_back
            total_distance_no_penalty += d_back
            times[v_idx] += d_back
            log(f"  无人机 {v_idx+1}: 返回仓库, 已服务 {len(routes[v_idx])} 个客户, "
                f"总载重 {loads[v_idx]}, 返回距离: {d_back:.2f}")
    
    # 计算 makespan（最长任务完成时间）
    makespan = max([times[v_idx] for v_idx in range(num_vehicles) if routes[v_idx]], default=0)
    
    # 过滤空路线
    non_empty_routes = [r for r in routes if r]
    
    # 约束惩罚检查
    for route in non_empty_routes:
        route_demand = customers[customers["customer_id"].isin(route)]["demand"].sum()
        if route_demand > capacity:
            constraint_violations += 1
            log(f"  警告: 路线 {non_empty_routes.index(route)+1} 超载, "
                f"需求={route_demand}, 容量={capacity}")
    
    total_constraint_penalty = constraint_violations * WP
    
    # 计算总成本
    total_cost = (
        total_distance * DISTANCE_WEIGHT
        + total_time_penalty
        + total_constraint_penalty
    )
    
    runtime = round(time.time() - start_time, 3)
    
    # 统计信息
    route_lengths = [len(r) for r in non_empty_routes]
    
    # 保存结果
    metrics = {
        "指标": [
            "总成本",
            "总飞行距离",
            "原始距离(无惩罚)",
            "时间惩罚",
            "约束惩罚",
            "Makespan(最长任务时间)",
            "超时订单数",
            "约束违规数",
            "使用的无人机数",
            "总服务客户数",
            "平均每架服务客户数",
            "最多服务客户数",
            "最少服务客户数",
            "算法运行时间(秒)"
        ],
        "数值": [
            f"{total_cost:.2f}",
            f"{total_distance:.2f}",
            f"{total_distance_no_penalty:.2f}",
            f"{total_time_penalty:.2f}",
            f"{total_constraint_penalty:.2f}",
            f"{makespan:.2f}",
            f"{overdue_count}",
            f"{constraint_violations}",
            f"{len(non_empty_routes)}",
            f"{len(customers)}",
            f"{np.mean(route_lengths) if route_lengths else 0:.1f}",
            f"{max(route_lengths) if route_lengths else 0}",
            f"{min(route_lengths) if route_lengths else 0}",
            f"{runtime}"
        ]
    }
    
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    
    # 详细输出
    log("=" * 60)
    log("算法运行完成！")
    log("=" * 60)
    log(f"总成本: {total_cost:.2f}")
    log(f"总飞行距离: {total_distance:.2f}")
    log(f"原始距离(无惩罚): {total_distance_no_penalty:.2f}")
    log(f"时间惩罚: {total_time_penalty:.2f}")
    log(f"约束惩罚: {total_constraint_penalty:.2f}")
    log(f"Makespan: {makespan:.2f}")
    log(f"超时订单数: {overdue_count}/{len(customers)}")
    log(f"约束违规数: {constraint_violations}")
    log(f"使用的无人机数: {len(non_empty_routes)}/{num_vehicles}")
    log(f"算法运行时间: {runtime} 秒")
    
    # 输出每架无人机的详细信息
    log("\n无人机任务分配详情:")
    for i, route in enumerate(non_empty_routes):
        route_demand = customers[customers["customer_id"].isin(route)]["demand"].sum()
        log(f"  无人机 {i+1}: {len(route)} 个客户, 总需求 {route_demand}, "
            f"客户ID: {route[:5]}{'...' if len(route) > 5 else ''}")
    
    # 保存结果到文本文件
    result_summary = os.path.join(RESULT_DIR, "greedy_summary.txt")
    with open(result_summary, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("多无人机贪心算法结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"数据文件: {CUSTOMER_FILE}\n")
        f.write(f"客户数量: {len(customers)}\n")
        f.write(f"无人机数量: {num_vehicles}\n")
        f.write(f"无人机容量: {capacity}\n\n")
        f.write(f"总成本: {total_cost:.2f}\n")
        f.write(f"总飞行距离: {total_distance:.2f}\n")
        f.write(f"原始距离(无惩罚): {total_distance_no_penalty:.2f}\n")
        f.write(f"时间惩罚: {total_time_penalty:.2f}\n")
        f.write(f"约束惩罚: {total_constraint_penalty:.2f}\n")
        f.write(f"Makespan: {makespan:.2f}\n\n")
        f.write(f"超时订单数: {overdue_count}\n")
        f.write(f"约束违规数: {constraint_violations}\n")
        f.write(f"使用的无人机数: {len(non_empty_routes)}\n\n")
        f.write("=" * 60 + "\n")
        f.write("各无人机任务详情\n")
        f.write("=" * 60 + "\n")
        for i, route in enumerate(non_empty_routes):
            route_demand = customers[customers["customer_id"].isin(route)]["demand"].sum()
            f.write(f"\n无人机 {i+1}:\n")
            f.write(f"  客户数: {len(route)}\n")
            f.write(f"  总需求: {route_demand}\n")
            f.write(f"  客户ID: {route}\n")
    
    # 绘制路线图
    plot_routes(customers, non_empty_routes, depot, ROUTE_FIG)
    
    # 显示图表（可选）
    # plt.show()
    
    return total_cost, total_distance, non_empty_routes

# ================== 主程序入口 ==================
if __name__ == "__main__":
    try:
        total_cost, total_distance, routes = greedy_vrp_multi()
        print("\n" + "=" * 60)
        print("程序执行成功！")
        print(f"最终结果 - 总成本: {total_cost:.2f}, 总距离: {total_distance:.2f}")
        print(f"结果已保存至: {RESULT_DIR}")
        print("=" * 60)
    except Exception as e:
        log(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()