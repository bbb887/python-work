"""
无人机配送路径优化 - 遗传算法
读取同目录下的 c101_info.csv 和 c101_customers.csv
输出格式仿照贪心算法：log、metrics CSV、summary 文本、路线图
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime

# ==================== 路径与目录配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "log")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CUSTOMER_FILE = os.path.join(BASE_DIR, "c101_customers.csv")
INFO_FILE = os.path.join(BASE_DIR, "c101_info.csv")
RESULT_FILE = os.path.join(RESULT_DIR, "ga_metrics.csv")
ROUTE_FIG = os.path.join(RESULT_DIR, "ga_routes.png")
LOG_FILE = os.path.join(LOG_DIR, "ga_run.log")
SUMMARY_FILE = os.path.join(RESULT_DIR, "ga_summary.txt")

# ==================== 日志函数 ====================
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ==================== 从 c101_info.csv 读取配置 ====================
def read_info():
    with open(INFO_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    raise Exception("c101_info.csv 为空或格式错误")

info = read_info()
depot = (float(info['DEPOT_X']), float(info['DEPOT_Y']))
depot_ready = float(info['DEPOT_READY_TIME'])
depot_due = float(info['DEPOT_DUE_DATE'])

# ==================== C101 数据集（从文件读取） ====================
customer_data = []
with open(CUSTOMER_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        customer_data.append([
            int(row['CUST_NO']),
            float(row['X']),
            float(row['Y']),
            int(row['DEMAND']),
            float(row['READY_TIME']),
            float(row['DUE_DATE']),
            float(row['SERVICE_TIME'])
        ])

# 无人机参数
W_max = int(info.get('VEHICLE_CAPACITY', 200))
D_max = 300                     # 最大航程（info 中无此字段，保留默认）
K_max = int(info.get('VEHICLE_NUMBER', 25))
w_t = 20                        # 超时惩罚权重
w_p = 1000                      # 硬约束惩罚权重
w_k = 1000                      # 超出数量惩罚权重

# 距离矩阵
n_customers = len(customer_data)
coords = [depot] + [(c[1], c[2]) for c in customer_data]

def euclidean(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

dist_matrix = [[0.0]*(n_customers+1) for _ in range(n_customers+1)]
for i in range(n_customers+1):
    for j in range(n_customers+1):
        if i != j:
            dist_matrix[i][j] = euclidean(coords[i], coords[j])

# 客户信息数组
demand = [0] + [c[3] for c in customer_data]
ready = [0] + [c[4] for c in customer_data]
due = [0] + [c[5] for c in customer_data]
service = [0] + [c[6] for c in customer_data]

# ================== 成本计算 ==================
def compute_cost(permutation):
    routes = []
    current_route = []
    current_load = 0
    current_dist = 0.0
    current_time = 0.0
    last_node = 0
    total_distance = 0.0
    total_time_penalty = 0.0
    total_hard_penalty = 0.0

    def finish_route():
        nonlocal total_distance, total_hard_penalty
        if not current_route:
            return
        return_dist = dist_matrix[last_node][0]
        return_time = current_time + return_dist
        depot_overtime = max(0, return_time - depot_due)
        total_time_penalty_local = w_t * depot_overtime
        total_route_dist = current_dist + return_dist
        load_penalty = max(0, current_load - W_max) * w_p
        range_penalty = max(0, total_route_dist - D_max) * w_p
        total_hard_penalty += load_penalty + range_penalty
        total_distance += total_route_dist
        routes.append((current_route.copy(), total_route_dist, current_load, return_time))

    for cust in permutation:
        travel_dist = dist_matrix[last_node][cust]
        arrival_time = current_time + travel_dist
        start_service = max(arrival_time, ready[cust])
        overtime = max(0, start_service - due[cust])
        finish_service = start_service + service[cust]
        temp_load = current_load + demand[cust]
        if temp_load > W_max or overtime > 100:
            finish_route()
            current_route = []
            current_load = 0
            current_dist = 0.0
            current_time = 0.0
            last_node = 0
            travel_dist = dist_matrix[0][cust]
            arrival_time = travel_dist
            start_service = max(arrival_time, ready[cust])
            overtime = max(0, start_service - due[cust])
            finish_service = start_service + service[cust]
            temp_load = demand[cust]
        current_route.append(cust)
        current_load = temp_load
        current_dist += travel_dist
        current_time = finish_service
        last_node = cust
        total_time_penalty += w_t * overtime

    finish_route()
    num_penalty = w_k * max(0, len(routes) - K_max)
    total_cost = total_distance + total_time_penalty + total_hard_penalty + num_penalty
    return total_cost, routes

# ================== 遗传算法 ==================
def create_individual():
    ind = list(range(1, n_customers+1))
    random.shuffle(ind)
    return ind

def crossover_ox(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b] = parent1[a:b]
    p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
            p2_idx += 1
    return child

def mutate_swap(individual, prob=0.1):
    if random.random() < prob:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def tournament_selection(population, fitnesses, k=3):
    best_idx = random.randint(0, len(population)-1)
    for _ in range(k-1):
        idx = random.randint(0, len(population)-1)
        if fitnesses[idx] < fitnesses[best_idx]:
            best_idx = idx
    return population[best_idx]

def genetic_algorithm(pop_size=150, generations=300, cx_prob=0.8, mut_prob=0.2, elite_size=2):
    population = [create_individual() for _ in range(pop_size)]
    best_individual = None
    best_cost = float('inf')
    best_routes = None

    for gen in range(generations):
        fitnesses = []
        for ind in population:
            cost, routes = compute_cost(ind)
            fitnesses.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_individual = ind.copy()
                best_routes = routes

        elite_indices = np.argsort(fitnesses)[:elite_size]
        elite = [population[i] for i in elite_indices]
        new_population = []

        while len(new_population) < pop_size - elite_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            if random.random() < cx_prob:
                child = crossover_ox(p1, p2)
            else:
                child = p1.copy()
            child = mutate_swap(child, mut_prob)
            new_population.append(child)

        new_population.extend(elite)
        population = new_population

        if (gen+1) % 50 == 0:
            log(f"Generation {gen+1:4d} | Best Cost = {best_cost:.2f} | Drones = {len(best_routes)}")

    return best_individual, best_cost, best_routes

# ================== 约束违反详细分析 ==================
def analyze_solution(best_routes):
    lines = []
    lines.append("\n" + "=" * 85)
    lines.append(f"{'路线':<4} {'客户数':<6} {'总载重':<8} {'载重违规':<10} {'航程':<10} {'航程违规':<10} {'客户超时总长':<12} {'仓库超时':<10} {'可行':<6}")
    lines.append("-" * 85)

    feasible_total = True
    for idx, (route, dist, load, ret_time) in enumerate(best_routes, start=1):
        total_cust_overtime = 0.0
        cur_time = 0.0
        last = 0
        for cust in route:
            travel = dist_matrix[last][cust]
            arr = cur_time + travel
            start = max(arr, ready[cust])
            over = max(0, start - due[cust])
            total_cust_overtime += over
            cur_time = start + service[cust]
            last = cust
        return_dist = dist_matrix[last][0]
        return_time = cur_time + return_dist
        depot_overtime = max(0, return_time - depot_due)
        load_viol = max(0, load - W_max)
        range_viol = max(0, dist - D_max)
        feasible = (load_viol == 0 and range_viol == 0 and total_cust_overtime == 0 and depot_overtime == 0)
        if not feasible:
            feasible_total = False
        lines.append(f"{idx:<4} {len(route):<6} {load:<8.1f} {load_viol:<10.1f} {dist:<10.2f} {range_viol:<10.2f} {total_cust_overtime:<12.2f} {depot_overtime:<10.2f} {'✓' if feasible else '✗':<6}")

    lines.append("-" * 85)
    if feasible_total:
        lines.append("总体结论：所有路线均满足载重、航程、客户时间窗、仓库时间窗约束。")
    else:
        lines.append("总体结论：存在违反约束的路线（惩罚项已计入总成本）。")

    result = "\n".join(lines)
    log(result)
    return feasible_total, result

# ================== 绘制路线图 ==================
def plot_routes(best_routes, save_path):
    plt.figure(figsize=(12, 8))
    for cust in range(1, n_customers+1):
        x, y = coords[cust]
        plt.scatter(x, y, c='skyblue', edgecolors='black', s=60, zorder=2)
        plt.annotate(str(cust), (x, y), fontsize=7, ha='center', va='center')
    plt.scatter(depot[0], depot[1], c='red', marker='s', s=200, edgecolors='black', label='Depot', zorder=3)
    plt.annotate('Depot', depot, fontsize=10, ha='center', va='bottom')
    colors = plt.cm.tab20(np.linspace(0, 1, len(best_routes)))
    for idx, (route, _, _, _) in enumerate(best_routes):
        path = [depot] + [coords[c] for c in route] + [depot]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        plt.plot(xs, ys, color=colors[idx], linewidth=1.5, alpha=0.7, label=f'Drone {idx+1}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title(f'GA UAV Delivery Routes (Total {len(best_routes)} drones)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"路线图已保存至: {save_path}")

# ================== 保存指标 CSV ==================
def save_metrics(best_routes, best_cost, runtime):
    total_distance = 0.0
    total_cust_overtime = 0.0
    total_overload = 0.0
    total_over_dist = 0.0
    makespan = 0.0

    for route, dist, load, ret_time in best_routes:
        total_distance += dist
        total_overload += max(0, load - W_max)
        total_over_dist += max(0, dist - D_max)
        if ret_time > makespan:
            makespan = ret_time
        # 计算客户超时
        cur_time = 0.0
        last = 0
        for cust in route:
            travel = dist_matrix[last][cust]
            arr = cur_time + travel
            start = max(arr, ready[cust])
            over = max(0, start - due[cust])
            total_cust_overtime += over
            cur_time = start + service[cust]
            last = cust

    Cdist = total_distance
    Ctime = w_t * total_cust_overtime
    Cpunish = w_p * (total_overload + total_over_dist)
    total_cost = Cdist + Ctime + Cpunish

    metrics = {
        "指标": [
            "总成本", "飞行距离成本", "原始距离(无惩罚)", "时间窗超时惩罚",
            "硬约束违规惩罚", "Makespan(最长任务时间)", "超时总时长",
            "超载量", "超里程量", "使用的无人机数", "总服务客户数",
            "算法运行时间(秒)"
        ],
        "数值": [
            f"{total_cost:.2f}", f"{Cdist:.2f}", f"{total_distance:.2f}",
            f"{Ctime:.2f}", f"{Cpunish:.2f}", f"{makespan:.2f}",
            f"{total_cust_overtime:.2f}", f"{total_overload:.1f}", f"{total_over_dist:.2f}",
            f"{len(best_routes)}", f"{n_customers}", f"{runtime:.2f}"
        ]
    }
    with open(RESULT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(metrics["指标"])
        writer.writerow(metrics["数值"])
    log(f"指标已保存至: {RESULT_FILE}")

# ================== 保存文本摘要 ==================
def save_summary(best_routes, best_cost, runtime, analysis_text):
    total_distance = 0.0
    total_cust_overtime = 0.0
    for route, dist, load, ret_time in best_routes:
        total_distance += dist
        cur_time = 0.0
        last = 0
        for cust in route:
            travel = dist_matrix[last][cust]
            arr = cur_time + travel
            start = max(arr, ready[cust])
            over = max(0, start - due[cust])
            total_cust_overtime += over
            cur_time = start + service[cust]
            last = cust

    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("无人机配送路径优化 - 遗传算法结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"数据文件: {CUSTOMER_FILE}\n")
        f.write(f"配置文件: {INFO_FILE}\n")
        f.write(f"参数: 容量={W_max}, 航程上限={D_max}, 最大无人机数={K_max}\n")
        f.write(f"权重: 时间惩罚={w_t}, 硬约束惩罚={w_p}, 数量惩罚={w_k}\n")
        f.write(f"仓库: {depot}, 时间窗=[{depot_ready}, {depot_due}]\n\n")
        f.write(f"总成本: {best_cost:.2f}\n")
        f.write(f"飞行距离: {total_distance:.2f}\n")
        f.write(f"客户超时总时长: {total_cust_overtime:.2f}\n")
        f.write(f"使用的无人机数: {len(best_routes)} / {K_max}\n")
        f.write(f"算法运行时间: {runtime:.2f} 秒\n\n")
        f.write("=" * 60 + "\n")
        f.write("各无人机任务详情\n")
        f.write("=" * 60 + "\n")
        for i, (route, dist, load, ret_time) in enumerate(best_routes):
            f.write(f"\n无人机 {i+1}: Load={load}/{W_max}  Dist={dist:.2f}/{D_max}  Return={ret_time:.2f}\n")
            f.write(f"  客户ID: {route}\n")
        f.write("\n" + analysis_text + "\n")
    log(f"摘要已保存至: {SUMMARY_FILE}")

# ================== 主程序 ==================
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    log("=" * 60)
    log("无人机配送路径优化 - 遗传算法")
    log("=" * 60)
    log(f"仓库: {depot}")
    log(f"仓库时间窗: [{depot_ready}, {depot_due}]")
    log(f"容量: {W_max}, 航程上限: {D_max}, 最大无人机数: {K_max}")
    log(f"权重: 时间惩罚={w_t}, 硬约束惩罚={w_p}, 数量惩罚={w_k}")
    log(f"客户数量: {n_customers}")

    import time
    start = time.time()

    best_solution, best_cost, best_routes = genetic_algorithm(
        pop_size=150, generations=4500, cx_prob=0.8, mut_prob=0.2, elite_size=2
    )

    elapsed = time.time() - start

    log(f"\n========== 最优解概览 ==========")
    log(f"总成本: {best_cost:.2f}")
    log(f"使用无人机数量: {len(best_routes)} / {K_max}")
    log(f"计算耗时: {elapsed:.2f} 秒")

    for i, (route, dist, load, ret_time) in enumerate(best_routes):
        route_str = str(route[:5]) + ('...' if len(route)>5 else '')
        log(f"无人机 {i+1:2d}: 客户 {route_str:20s} | 载重 {load:3d}/{W_max} | 航程 {dist:6.2f}/{D_max} | 返回时间 {ret_time:6.2f}")

    # 详细约束表
    feasible, analysis_text = analyze_solution(best_routes)

    # 保存指标和摘要
    save_metrics(best_routes, best_cost, elapsed)
    save_summary(best_routes, best_cost, elapsed, analysis_text)

    # 路线图
    plot_routes(best_routes, ROUTE_FIG)

    print("\n" + "=" * 60)
    print("程序执行成功！")
    print(f"最终结果 - 总成本: {best_cost:.2f}, 使用无人机: {len(best_routes)}")
    print(f"结果已保存至: {RESULT_DIR}")
    print(f"日志已保存至: {LOG_FILE}")
    print("=" * 60)