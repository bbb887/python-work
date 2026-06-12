import csv
import math
import random
import copy
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Tuple

# ==================== 路径与全局配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "log")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CUSTOMER_FILE = os.path.join(BASE_DIR, "c101_customers.csv")
INFO_FILE = os.path.join(BASE_DIR, "c101_info.csv")
RESULT_FILE = os.path.join(RESULT_DIR, "mixed_ga_metrics.csv")
ROUTE_FIG = os.path.join(RESULT_DIR, "mixed_ga_routes.png")
LOG_FILE = os.path.join(LOG_DIR, "mixed_ga_run.log")
SUMMARY_FILE = os.path.join(RESULT_DIR, "mixed_ga_summary.txt")

# ==================== 参数 (默认，实际从 info.csv 读取) ====================
DMAX = 300.0                # 无人机最大航程 (info.csv 中无此字段，暂保留默认值)
WT = 20                     # 时间窗超时惩罚权重
WP = 1000                   # 硬约束违规惩罚权重 (超载/超里程)

# 从 info 文件读取的变量（将在 main 中赋值）
DEPOT = (0.0, 0.0)
CAPACITY = 200
DEPOT_READY_TIME = 0.0
DEPOT_DUE_DATE = 1236.0

# ==================== 日志函数 ====================
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ==================== 数据读取 ====================
def read_info():
    """从 c101_info.csv 读取配置参数"""
    with open(INFO_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row  # 仅第一行
    raise Exception("c101_info.csv 为空或格式错误")

def read_customers(filename):
    """读取客户数据 (不再传入 depot, capacity)"""
    customers = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            customers.append((
                int(row['CUST_NO']),
                float(row['X']),
                float(row['Y']),
                int(row['DEMAND']),
                float(row['READY_TIME']),
                float(row['DUE_DATE']),
                float(row['SERVICE_TIME'])
            ))
    return customers

def build_distance_matrix(customers, depot):
    points = [depot] + [(c[1], c[2]) for c in customers]
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dist[i][j] = math.hypot(dx, dy)
    return dist

# ==================== 核心评价函数 ====================
def evaluate_route(route, dist_matrix, customers, depot_idx=0):
    """返回: (路线距离, 载重, 超时总时长, 是否超里程)"""
    if not route:
        return 0.0, 0, 0.0, False

    prev = depot_idx
    current_time = 0.0
    total_dist = 0.0
    load = 0
    time_excess = 0.0

    for cust_idx in route:
        cust = customers[cust_idx]
        demand = cust[3]
        ready = cust[4]
        due = cust[5]
        service = cust[6]

        dist = dist_matrix[prev][cust_idx + 1]
        current_time += dist
        total_dist += dist

        if current_time > due:
            time_excess += (current_time - due)
        elif current_time < ready:
            current_time = ready

        current_time += service
        load += demand
        prev = cust_idx + 1

    dist_back = dist_matrix[prev][depot_idx]
    total_dist += dist_back
    current_time += dist_back

    # 仓库关闭时间使用从 info 读取的值
    if current_time > DEPOT_DUE_DATE:
        time_excess += (current_time - DEPOT_DUE_DATE)

    return total_dist, load, time_excess, total_dist > DMAX

def evaluate_solution(solution, dist_matrix, customers, capacity):
    total_dist = 0.0
    total_overload = 0
    total_time_excess = 0.0
    total_over_dist = 0.0

    for route in solution:
        d, load, tex, _ = evaluate_route(route, dist_matrix, customers)
        total_dist += d
        total_time_excess += tex
        overload = max(0, load - capacity)
        total_overload += overload
        if d > DMAX:
            total_over_dist += (d - DMAX)

    return total_dist, total_overload, total_time_excess, total_over_dist

def fitness(solution, dist_matrix, customers, capacity):
    dist, overload, time_ex, over_dist = evaluate_solution(solution, dist_matrix, customers, capacity)
    return dist + WT * time_ex + WP * (overload + over_dist)

# ==================== 时间窗感知的初始解构造 (Solomon I1 思想) ====================
def build_time_aware_solution(customers, capacity, dist_matrix):
    n = len(customers)
    unvisited = set(range(n))
    routes = []

    while unvisited:
        route = []
        load = 0.0
        current_time = 0.0
        prev = 0  # depot index

        while True:
            best_idx = None
            best_cost = float('inf')
            for c in unvisited:
                cust = customers[c]
                d = dist_matrix[prev][c + 1]
                arrival = current_time + d

                if load + cust[3] > capacity:
                    continue
                if arrival > cust[5]:   # 无法在 due_date 前到达
                    continue

                wait = max(0.0, cust[4] - arrival)
                cost = d + 0.2 * wait
                if cost < best_cost:
                    best_cost = cost
                    best_idx = c

            if best_idx is None:
                break

            cust = customers[best_idx]
            d = dist_matrix[prev][best_idx + 1]
            current_time += d
            if current_time < cust[4]:
                current_time = cust[4]
            current_time += cust[6]
            load += cust[3]
            route.append(best_idx)
            unvisited.remove(best_idx)
            prev = best_idx + 1

        if route:
            routes.append(route)

    return routes

def generate_initial_population(pop_size, customers, capacity, dist_matrix):
    population = []
    pop_t = build_time_aware_solution(customers, capacity, dist_matrix)
    population.append(pop_t)

    for _ in range(pop_size - 1):
        perm = list(range(len(customers)))
        random.shuffle(perm)
        sol = []
        current_route = []
        current_load = 0
        for c in perm:
            demand = customers[c][3]
            if current_load + demand > capacity:
                if current_route:
                    sol.append(current_route)
                current_route = [c]
                current_load = demand
            else:
                current_route.append(c)
                current_load += demand
        if current_route:
            sol.append(current_route)
        population.append(sol)
    return population

# ==================== 遗传算法操作 ====================
def route_to_giant(solution):
    tour = []
    for r in solution:
        tour.extend(r)
    return tour

def giant_to_routes(tour, customers, capacity):
    routes = []
    cur_route = []
    cur_load = 0
    for c in tour:
        demand = customers[c][3]
        if cur_load + demand > capacity:
            if cur_route:
                routes.append(cur_route)
            cur_route = [c]
            cur_load = demand
        else:
            cur_route.append(c)
            cur_load += demand
    if cur_route:
        routes.append(cur_route)
    return routes

def cross_order(p1, p2):
    size = len(p1)
    if size < 2:
        return copy.deepcopy(p1), copy.deepcopy(p2)
    c1 = [None] * size
    c2 = [None] * size
    i, j = sorted(random.sample(range(size), 2))
    c1[i:j] = p1[i:j]
    c2[i:j] = p2[i:j]

    def fill(child, parent_other):
        pos = j % size
        for gene in parent_other:
            if gene not in child:
                child[pos] = gene
                pos = (pos + 1) % size
        return child

    c1 = fill(c1, p2[j:] + p2[:j])
    c2 = fill(c2, p1[j:] + p1[:j])
    return c1, c2

def mutate(tour, mutation_rate=0.3):
    if random.random() < mutation_rate:
        if random.random() < 0.5:
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        else:
            i, j = sorted(random.sample(range(len(tour)), 2))
            tour[i:j+1] = reversed(tour[i:j+1])

def two_opt(route, dist_matrix, customers, max_iter=100):
    if len(route) < 2:
        return route

    def route_cost(r):
        d, _, tex, _ = evaluate_route(r, dist_matrix, customers)
        over = max(0, d - DMAX)
        return d + WT * tex + WP * over

    best_route = route[:]
    best_cost = route_cost(best_route)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        for i in range(1, len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_cost = route_cost(new_route)
                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    improved = True
        it += 1
    return best_route

def local_search(solution, dist_matrix, customers):
    return [two_opt(route, dist_matrix, customers) for route in solution]

# ==================== 遗传算法主流程 ====================
def genetic_algorithm(customers, capacity, dist_matrix,
                      pop_size=50, generations=500,
                      elite_rate=0.1, crossover_rate=0.9,
                      mutation_rate=0.3, local_search_freq=10):
    elite_count = max(1, int(pop_size * elite_rate))
    population = generate_initial_population(pop_size, customers, capacity, dist_matrix)

    best_solution = None
    best_cost = float('inf')

    for gen in range(generations):
        fitness_vals = [fitness(sol, dist_matrix, customers, capacity) for sol in population]

        for i, fv in enumerate(fitness_vals):
            if fv < best_cost:
                best_cost = fv
                best_solution = copy.deepcopy(population[i])

        sorted_pop = [sol for _, sol in sorted(zip(fitness_vals, population), key=lambda x: x[0])]
        new_pop = sorted_pop[:elite_count]

        while len(new_pop) < pop_size:
            def tournament_select(fitness_vals, k=3):
                selected = random.sample(range(len(population)), k)
                best = min(selected, key=lambda i: fitness_vals[i])
                return population[best]

            p1 = tournament_select(fitness_vals)
            p2 = tournament_select(fitness_vals)

            if random.random() < crossover_rate:
                tour1, tour2 = cross_order(route_to_giant(p1), route_to_giant(p2))
                mutate(tour1, mutation_rate)
                mutate(tour2, mutation_rate)
                child1 = giant_to_routes(tour1, customers, capacity)
                child2 = giant_to_routes(tour2, customers, capacity)
                new_pop.append(child1)
                if len(new_pop) < pop_size:
                    new_pop.append(child2)
            else:
                new_pop.append(copy.deepcopy(p1))
                if len(new_pop) < pop_size:
                    new_pop.append(copy.deepcopy(p2))

        population = new_pop

        if gen > 0 and gen % local_search_freq == 0:
            for i in range(elite_count):
                population[i] = local_search(population[i], dist_matrix, customers)

        if gen % 50 == 0:
            d, ov, tex, od = evaluate_solution(best_solution, dist_matrix, customers, capacity)
            log(f"Gen {gen:3d} | Fit={d + WT*tex + WP*(ov+od):.2f}  Dist={d:.2f}  TimeEx={tex:.2f}  Overload={ov}  OverDist={od:.2f}")

    return best_solution, best_cost

# ==================== 结果可视化与输出 ====================
def plot_solution(solution, customers, depot, save_path):
    colors = ['b','g','r','c','m','y','k','orange','purple']
    plt.figure(figsize=(10,8))
    xs = [c[1] for c in customers]
    ys = [c[2] for c in customers]
    plt.scatter(xs, ys, c='gray', s=30, label='Customers')
    plt.scatter(depot[0], depot[1], c='red', s=200, marker='s', label='Depot')
    for i, route in enumerate(solution):
        if not route: continue
        color = colors[i % len(colors)]
        pts = [depot] + [(customers[c][1], customers[c][2]) for c in route] + [depot]
        plt.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', color=color, linewidth=1, markersize=5)
    plt.title("VRPTW Solution - Repaired GA")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.grid(True); plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"路线图已保存至: {save_path}")

def save_metrics(best_solution, dist_matrix, customers, capacity,
                 dist, overload, time_ex, over_dist, makespan,
                 num_routes, total_cust, runtime, best_fit):
    Cdist = dist
    Ctime = WT * time_ex
    Cpunish = WP * (overload + over_dist)
    total_cost = Cdist + Ctime + Cpunish
    metrics = {
        "指标": [...],
        "数值": [...]
    }
    with open(RESULT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(metrics["指标"])
        writer.writerow(metrics["数值"])
    log(f"指标已保存至: {RESULT_FILE}")
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        # ... 前面的内容保持不变 ...
        f.write("各无人机任务详情\n")
        f.write("=" * 60 + "\n")
        for i, route in enumerate(best_solution):
            d, load, tex, over = evaluate_route(route, dist_matrix, customers)
            ids = [customers[c][0] for c in route]
            f.write(f"\n无人机 {i+1}: Load={load}/{capacity}  Dist={d:.2f}  TimeEx={tex:.2f}  OverDist={d > DMAX}\n")
            f.write(f"  客户ID: {ids}\n")
    log(f"摘要已保存至: {SUMMARY_FILE}")

    # 文本摘要
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VRPTW 混合遗传算法结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"数据文件: {CUSTOMER_FILE}\n")
        f.write(f"参数: 容量={CAPACITY}, 航程上限={DMAX}, WT={WT}, WP={WP}\n")
        f.write(f"仓库: {DEPOT}, 关闭时间={DEPOT_DUE_DATE}\n\n")
        f.write(f"总成本: {total_cost:.2f}\n")
        f.write(f"飞行距离: {dist:.2f}\n")
        f.write(f"时间窗超时惩罚: {Ctime:.2f} (超时总时长 {time_ex:.2f})\n")
        f.write(f"硬约束违规惩罚: {Cpunish:.2f}\n")
        f.write(f"  超载量: {overload} (惩罚 {WP*overload:.2f})\n")
        f.write(f"  超里程量: {over_dist:.2f} (惩罚 {WP*over_dist:.2f})\n")
        f.write(f"使用的无人机数: {num_routes}\n")
        f.write(f"总服务客户数: {total_cust}\n")
        f.write(f"Makespan: {makespan:.2f}\n")
        f.write(f"算法运行时间: {runtime:.2f} 秒\n\n")
        f.write("=" * 60 + "\n")
        f.write("各无人机任务详情\n")
        f.write("=" * 60 + "\n")
        for i, route in enumerate(best_solution):
            d, load, tex, over = evaluate_route(route, DIST, CUSTOMERS)
            ids = [CUSTOMERS[c][0] for c in route]
            f.write(f"\n无人机 {i+1}: Load={load}/{CAPACITY}  Dist={d:.2f}  TimeEx={tex:.2f}  OverDist={d > DMAX}\n")
            f.write(f"  客户ID: {ids}\n")
    log(f"摘要已保存至: {SUMMARY_FILE}")

def get_makespan(solution, dist_matrix, customers):
    max_time = 0.0
    for route in solution:
        _, _, _, _ = evaluate_route(route, dist_matrix, customers)
        # evaluate_route 没有直接返回完成时间，需要重新计算
        # 快速计算路径的总完成时间
        current_time = 0.0
        prev = 0
        for c in route:
            cust = customers[c]
            d = dist_matrix[prev][c + 1]
            current_time += d
            if current_time < cust[4]:
                current_time = cust[4]
            current_time += cust[6]
            prev = c + 1
        current_time += dist_matrix[prev][0]
        if current_time > max_time:
            max_time = current_time
    return max_time

# ==================== 主程序 ====================
if __name__ == "__main__":
    log("=" * 60)
    log("开始运行混合遗传算法")
    log("=" * 60)

    # 读取配置
    info = read_info()
    DEPOT = (float(info['DEPOT_X']), float(info['DEPOT_Y']))
    CAPACITY = int(info['VEHICLE_CAPACITY'])
    DEPOT_READY_TIME = float(info['DEPOT_READY_TIME'])
    DEPOT_DUE_DATE = float(info['DEPOT_DUE_DATE'])
    # VEHICLE_NUMBER 暂未使用，仅记录
    VEHICLE_NUMBER = int(info['VEHICLE_NUMBER'])

    log(f"仓库: {DEPOT}")
    log(f"容量: {CAPACITY}")
    log(f"车辆数上限: {VEHICLE_NUMBER}")
    log(f"仓库时间窗: [{DEPOT_READY_TIME}, {DEPOT_DUE_DATE}]")

    # 读取客户
    CUSTOMERS = read_customers(CUSTOMER_FILE)
    log(f"客户数量: {len(CUSTOMERS)}")

    DIST = build_distance_matrix(CUSTOMERS, DEPOT)

    random.seed(42)

    start = time.time()
    best_sol, best_fit = genetic_algorithm(
        customers=CUSTOMERS,
        capacity=CAPACITY,
        dist_matrix=DIST,
        pop_size=50,
        generations=500,
        local_search_freq=10
    )
    elapsed = time.time() - start

    # 计算详细指标
    dist, overload, time_ex, over_dist = evaluate_solution(best_sol, DIST, CUSTOMERS, CAPACITY)
    makespan = get_makespan(best_sol, DIST, CUSTOMERS)
    num_routes = len(best_sol)
    total_cust = sum(len(r) for r in best_sol)

    log(f"计算耗时: {elapsed:.2f} 秒")
    log(f"总成本: {best_fit:.2f}  (距离={dist:.2f}, 超时={time_ex:.2f}, 违规={overload+over_dist:.2f})")

    # 保存指标和摘要
    save_metrics(best_sol, DIST, CUSTOMERS, CAPACITY,
             dist, overload, time_ex, over_dist, makespan,
             num_routes, total_cust, elapsed, best_fit)

    # 保存路线图
    plot_solution(best_sol, CUSTOMERS, DEPOT, ROUTE_FIG)

    # 输出路线详情到日志
    log("\n无人机任务分配详情:")
    for i, route in enumerate(best_sol):
        d, load, tex, over = evaluate_route(route, DIST, CUSTOMERS)
        ids = [CUSTOMERS[c][0] for c in route]
        log(f"  无人机 {i+1}: Load={load}/{CAPACITY}  Dist={d:.2f}  TimeEx={tex:.2f}  OverDist={d > DMAX}")
        log(f"    客户ID: {ids}")

    print("\n" + "=" * 60)
    print("程序执行成功！")
    print(f"最终结果 - 总成本: {best_fit:.2f}, 总距离: {dist:.2f}")
    print(f"结果已保存至: {RESULT_DIR}")
    print("=" * 60)