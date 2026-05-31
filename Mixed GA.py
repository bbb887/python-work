"""
VRPTW 混合遗传算法 (修复版)
目标函数：总成本 = 飞行距离 + 时间窗超时惩罚 + 载重/航程硬约束惩罚
适配 C101 数据集：车辆容量 200，航程上限 300，时间窗权重 20，硬约束权重 1000
修复重点：用 Solomon 插入启发式 I1 构造时间窗可行的初始种群，避免巨量超时惩罚
"""

import csv
import math
import random
import copy
import time
import matplotlib.pyplot as plt
from typing import List, Tuple

# ==================== 全局参数 ====================
DMAX = 300.0          # 无人机最大航程
WT = 20               # 时间窗超时惩罚权重（每超时 1 单位）
WP = 1000             # 硬约束违规惩罚权重（超载 / 超里程）
DEPOT_DUE = 1236.0    # 仓库关闭时间（车辆必须在此之前返回）

# ==================== 数据读取 ====================
def read_customers_csv(filename: str, depot=(40.0, 50.0), capacity=200):
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
    return depot, customers, capacity

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

    if current_time > DEPOT_DUE:
        time_excess += (current_time - DEPOT_DUE)

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
    """
    基于插入启发式构造一个初始解，优先考虑时间窗可行性和最小等待/距离增加。
    """
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

                # 容量约束
                if load + cust[3] > capacity:
                    continue

                # 必须能在 due_date 前到达（允许到达后超时？不允许超时，但这里只允许等待）
                if arrival > cust[5]:
                    continue

                # 计算插入代价：距离 + 等待时间惩罚（权重 0.2，使得等待比超时代价小）
                wait = max(0.0, cust[4] - arrival)
                cost = d + 0.2 * wait

                if cost < best_cost:
                    best_cost = cost
                    best_idx = c

            if best_idx is None:
                break  # 当前路线无法再插入任何客户，开新路线

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

    # 如果还有未访问客户（理论上不应该，但万一出现），随机分配，但这种情况很少
    return routes

def generate_initial_population(pop_size, customers, capacity, dist_matrix):
    """生成多样性种群：一部分用时间感知构造，一部分随机打乱并修复"""
    population = []
    # 至少有一个时间感知个体
    pop_t = build_time_aware_solution(customers, capacity, dist_matrix)
    population.append(pop_t)

    # 剩余个体：随机打乱后用容量分割
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
    """仅根据容量切分路线，用于交叉后重建"""
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
    """顺序交叉 (OX)"""
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

# ==================== 2-opt 局部搜索（使用新目标） ====================
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

    # 使用改进的初始种群
    population = generate_initial_population(pop_size, customers, capacity, dist_matrix)

    best_solution = None
    best_cost = float('inf')

    for gen in range(generations):
        fitness_vals = [fitness(sol, dist_matrix, customers, capacity) for sol in population]

        for i, fv in enumerate(fitness_vals):
            if fv < best_cost:
                best_cost = fv
                best_solution = copy.deepcopy(population[i])

        # 精英保留
        sorted_pop = [sol for _, sol in sorted(zip(fitness_vals, population), key=lambda x: x[0])]
        new_pop = sorted_pop[:elite_count]

        # 繁殖
        while len(new_pop) < pop_size:
            # 锦标赛选择
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

        # 定期局部搜索
        if gen > 0 and gen % local_search_freq == 0:
            for i in range(elite_count):
                population[i] = local_search(population[i], dist_matrix, customers)

        if gen % 50 == 0:
            d, ov, tex, od = evaluate_solution(best_solution, dist_matrix, customers, capacity)
            print(f"Gen {gen:3d} | Cost={d + WT*tex + WP*(ov+od):.2f}  Dist={d:.2f}  TimeExc={tex:.2f}  Overload={ov}  OverDist={od:.2f}")

    return best_solution, best_cost

# ==================== 结果输出 ====================
def print_routes(solution, customers, dist_matrix, capacity):
    for i, route in enumerate(solution):
        d, load, tex, over = evaluate_route(route, dist_matrix, customers)
        ids = [customers[c][0] for c in route]
        print(f"Route {i+1}: Load={load}/{capacity}  Dist={d:.2f}  TimeEx={tex:.2f}  OverDist={d > DMAX}")
        print(f"  Customers: {ids}")

def plot_solution(solution, customers, depot):
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
    plt.grid(True); plt.legend(); plt.show()

# ==================== 主程序 ====================
if __name__ == "__main__":
    random.seed(42)
    DEPOT, CUSTOMERS, CAPACITY = read_customers_csv("c101_customers.csv", capacity=200)
    DIST = build_distance_matrix(CUSTOMERS, DEPOT)

    print(f"仓库: {DEPOT}  客户数: {len(CUSTOMERS)}  容量: {CAPACITY}  航程上限: {DMAX}")
    print(f"参数: WT={WT}, WP={WP}, 仓库关闭={DEPOT_DUE}")

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
    print(f"\n计算耗时: {elapsed:.2f} 秒")

    # 详细结果
    dist, overload, time_ex, over_dist = evaluate_solution(best_sol, DIST, CUSTOMERS, CAPACITY)
    Cdist = dist
    Ctime = WT * time_ex
    Cpunish = WP * (overload + over_dist)
    total_cost = Cdist + Ctime + Cpunish

    print("\n" + "="*50)
    print("最优解 总成本分析")
    print("-"*50)
    print(f"飞行距离成本 Cdist      : {Cdist:.2f}")
    print(f"时间窗超时惩罚 Ctime    : {Ctime:.2f}  (超时总时长 {time_ex:.2f})")
    print(f"硬约束违规惩罚 Cpunish  : {Cpunish:.2f}")
    print(f"  其中超载量            : {overload}  (惩罚 {WP*overload:.2f})")
    print(f"  其中超里程量          : {over_dist:.2f}  (惩罚 {WP*over_dist:.2f})")
    print(f"总成本 = {total_cost:.2f}")
    print("="*50)

    print_routes(best_sol, CUSTOMERS, DIST, CAPACITY)
    plot_solution(best_sol, CUSTOMERS, DEPOT)