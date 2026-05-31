import math
import random
import numpy as np
import matplotlib.pyplot as plt

# ==================== C102 数据集 ====================
# 客户数据：[CUST_NO, X, Y, DEMAND, READY_TIME, DUE_DATE, SERVICE_TIME]
customer_data = [
    [1,45.0,68.0,10,0,1000,90],
    [2,45.0,70.0,30,0,1000,90],
    [3,42.0,66.0,10,0,1000,90],
    [4,42.0,68.0,10,0,1000,90],
    [5,42.0,65.0,10,0,1000,90],
    [6,40.0,69.0,20,0,1000,90],
    [7,40.0,66.0,20,0,1000,90],
    [8,38.0,68.0,20,0,1000,90],
    [9,38.0,70.0,10,0,1000,90],
    [10,35.0,66.0,10,0,1000,90],
    [11,35.0,69.0,10,0,1000,90],
    [12,25.0,85.0,20,0,1000,90],
    [13,22.0,75.0,30,0,1000,90],
    [14,22.0,85.0,10,0,1000,90],
    [15,20.0,80.0,40,0,1000,90],
    [16,20.0,85.0,40,0,1000,90],
    [17,18.0,75.0,20,0,1000,90],
    [18,15.0,75.0,20,0,1000,90],
    [19,15.0,80.0,10,0,1000,90],
    [20,30.0,50.0,10,0,1000,90],
    [21,30.0,52.0,20,0,1000,90],
    [22,28.0,52.0,20,0,1000,90],
    [23,28.0,55.0,10,0,1000,90],
    [24,25.0,50.0,10,0,1000,90],
    [25,25.0,52.0,40,0,1000,90],
    [26,25.0,55.0,10,0,1000,90],
    [27,23.0,52.0,10,0,1000,90],
    [28,23.0,55.0,20,0,1000,90],
    [29,20.0,50.0,10,0,1000,90],
    [30,20.0,55.0,10,0,1000,90],
    [31,10.0,35.0,20,0,1000,90],
    [32,10.0,40.0,30,0,1000,90],
    [33,8.0,40.0,40,0,1000,90],
    [34,8.0,45.0,20,0,1000,90],
    [35,5.0,35.0,10,0,1000,90],
    [36,5.0,45.0,10,0,1000,90],
    [37,2.0,40.0,20,0,1000,90],
    [38,0.0,40.0,30,0,1000,90],
    [39,0.0,45.0,20,0,1000,90],
    [40,35.0,30.0,10,0,1000,90],
    [41,35.0,32.0,10,0,1000,90],
    [42,33.0,32.0,20,0,1000,90],
    [43,33.0,35.0,10,0,1000,90],
    [44,32.0,30.0,10,0,1000,90],
    [45,30.0,30.0,10,0,1000,90],
    [46,30.0,32.0,30,0,1000,90],
    [47,30.0,35.0,10,0,1000,90],
    [48,28.0,30.0,10,0,1000,90],
    [49,28.0,35.0,10,0,1000,90],
    [50,26.0,32.0,10,0,1000,90],
    [51,25.0,30.0,10,0,1000,90],
    [52,25.0,35.0,10,0,1000,90],
    [53,44.0,5.0,20,0,1000,90],
    [54,42.0,10.0,40,0,1000,90],
    [55,42.0,15.0,10,0,1000,90],
    [56,40.0,5.0,30,0,1000,90],
    [57,40.0,15.0,40,0,1000,90],
    [58,38.0,5.0,30,0,1000,90],
    [59,38.0,15.0,10,0,1000,90],
    [60,35.0,5.0,20,0,1000,90],
    [61,50.0,30.0,10,0,1000,90],
    [62,50.0,35.0,20,0,1000,90],
    [63,50.0,40.0,50,0,1000,90],
    [64,48.0,30.0,10,0,1000,90],
    [65,48.0,40.0,10,0,1000,90],
    [66,47.0,35.0,10,0,1000,90],
    [67,47.0,40.0,10,0,1000,90],
    [68,45.0,30.0,10,0,1000,90],
    [69,45.0,35.0,10,0,1000,90],
    [70,95.0,30.0,30,0,1000,90],
    [71,95.0,35.0,20,0,1000,90],
    [72,53.0,30.0,10,0,1000,90],
    [73,92.0,30.0,10,0,1000,90],
    [74,53.0,35.0,50,0,1000,90],
    [75,45.0,65.0,20,0,1000,90],
    [76,90.0,35.0,10,0,1000,90],
    [77,88.0,30.0,10,0,1000,90],
    [78,88.0,35.0,20,0,1000,90],
    [79,87.0,30.0,10,0,1000,90],
    [80,85.0,25.0,10,0,1000,90],
    [81,85.0,35.0,30,0,1000,90],
    [82,75.0,55.0,20,0,1000,90],
    [83,72.0,55.0,10,0,1000,90],
    [84,70.0,58.0,20,0,1000,90],
    [85,68.0,60.0,30,0,1000,90],
    [86,66.0,55.0,10,0,1000,90],
    [87,65.0,55.0,20,0,1000,90],
    [88,65.0,60.0,30,0,1000,90],
    [89,63.0,58.0,10,0,1000,90],
    [90,60.0,55.0,10,0,1000,90],
    [91,60.0,60.0,10,0,1000,90],
    [92,67.0,85.0,20,0,1000,90],
    [93,65.0,85.0,40,0,1000,90],
    [94,65.0,82.0,10,0,1000,90],
    [95,62.0,80.0,30,0,1000,90],
    [96,60.0,80.0,10,0,1000,90],
    [97,60.0,85.0,30,0,1000,90],
    [98,58.0,75.0,20,0,1000,90],
    [99,55.0,80.0,10,0,1000,90],
    [100,55.0,85.0,20,0,1000,90],
]


depot = (40.0, 50.0)
depot_ready = 0
depot_due = 1236
# 无人机参数
W_max = 200      # 最大载重
D_max = 300      # 最大航程
K_max = 25       # 最大无人机数量
w_t = 20         # 超时惩罚权重
w_p = 1000       # 硬约束惩罚权重
w_k = 1000       # 超出数量惩罚权重

# 距离矩阵
n_customers = len(customer_data)
coords = [depot] + [(c[1], c[2]) for c in customer_data]  # 索引0为仓库

def euclidean(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

dist_matrix = [[0.0]*(n_customers+1) for _ in range(n_customers+1)]
for i in range(n_customers+1):
    for j in range(n_customers+1):
        if i != j:
            dist_matrix[i][j] = euclidean(coords[i], coords[j])

# 客户信息数组（索引1..100）
demand = [0] + [c[3] for c in customer_data]
ready = [0] + [c[4] for c in customer_data]
due = [0] + [c[5] for c in customer_data]
service = [0] + [c[6] for c in customer_data]

# ================== 2. 成本计算（解码 + 综合成本） ==================
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

        # 新开无人机条件：容量超限 或 超时过大
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

# ================== 3. 遗传算法 ==================
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

        if (gen+1) % 30 == 0:
            print(f"Generation {gen+1:3d} | Best Cost = {best_cost:.2f} | Drones = {len(best_routes)}")

    return best_individual, best_cost, best_routes

# ================== 4. 约束违反详细分析 ==================
def analyze_solution(best_routes):
    print("\n========== 约束违反情况详细表 ==========")
    print(f"{'路线':<4} {'客户数':<6} {'总载重':<8} {'载重违规':<10} {'航程':<10} {'航程违规':<10} {'客户超时总长':<12} {'仓库超时':<10} {'可行':<6}")
    print("-" * 85)

    feasible_total = True
    for idx, (route, dist, load, ret_time) in enumerate(best_routes, start=1):
        # 计算客户超时总和
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
        # 仓库超时
        return_dist = dist_matrix[last][0]
        return_time = cur_time + return_dist
        depot_overtime = max(0, return_time - depot_due)

        load_viol = max(0, load - W_max)
        range_viol = max(0, dist - D_max)
        feasible = (load_viol == 0 and range_viol == 0 and total_cust_overtime == 0 and depot_overtime == 0)
        if not feasible:
            feasible_total = False

        print(f"{idx:<4} {len(route):<6} {load:<8.1f} {load_viol:<10.1f} {dist:<10.2f} {range_viol:<10.2f} {total_cust_overtime:<12.2f} {depot_overtime:<10.2f} {'✓' if feasible else '✗':<6}")

    print("-" * 85)
    if feasible_total:
        print("总体结论：所有路线均满足载重、航程、客户时间窗、仓库时间窗约束。")
    else:
        print("总体结论：存在违反约束的路线（惩罚项已计入总成本，解不可直接用于实际，需继续优化）。")
    return feasible_total

# ================== 5. 绘制路线图 ==================
def plot_routes(best_routes):
    plt.figure(figsize=(12, 8))

    # 绘制所有客户点
    for cust in range(1, n_customers+1):
        x, y = coords[cust]
        plt.scatter(x, y, c='skyblue', edgecolors='black', s=60, zorder=2)
        plt.annotate(str(cust), (x, y), fontsize=7, ha='center', va='center')

    # 仓库
    plt.scatter(depot[0], depot[1], c='red', marker='s', s=200, edgecolors='black', label='Depot', zorder=3)
    plt.annotate('Depot', depot, fontsize=10, ha='center', va='bottom')

    # 各路线不同颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(best_routes)))
    for idx, (route, _, _, _) in enumerate(best_routes):
        path = [depot] + [coords[c] for c in route] + [depot]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        plt.plot(xs, ys, color=colors[idx], linewidth=1.5, alpha=0.7, label=f'Drone {idx+1}')

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title(f'UAV Delivery Routes (Total {len(best_routes)} drones)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('uav_routes.png', dpi=150)
    plt.show()
    print("\n路线图已保存为 uav_routes.png")

# ================== 6. 主程序 ==================
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    print("=" * 60)
    print("无人机配送路径优化 - 遗传算法")
    print(f"参数：容量={W_max}, 航程上限={D_max}, 最大无人机数={K_max}")
    print(f"权重：时间惩罚={w_t}, 硬约束惩罚={w_p}, 数量惩罚={w_k}")
    print("=" * 60)

    best_solution, best_cost, best_routes = genetic_algorithm(
        pop_size=150, generations=4500, cx_prob=0.8, mut_prob=0.2, elite_size=2
    )

    print("\n========== 最优解概览 ==========")
    print(f"总成本: {best_cost:.2f}")
    print(f"使用无人机数量: {len(best_routes)} / {K_max}")

    for i, (route, dist, load, ret_time) in enumerate(best_routes):
        route_str = str(route[:5]) + ('...' if len(route)>5 else '')
        print(f"无人机 {i+1:2d}: 客户 {route_str:20s} | 载重 {load:3d}/{W_max} | 航程 {dist:6.2f}/{D_max} | 返回时间 {ret_time:6.2f}")

    # 详细约束表
    analyze_solution(best_routes)

    # 路线图
    plot_routes(best_routes)