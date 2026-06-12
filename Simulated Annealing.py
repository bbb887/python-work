import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
import random
import math
import csv
from typing import List, Tuple, Dict, Any, Optional
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 路径配置（同目录读取） ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "log")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CUSTOMER_FILE = os.path.join(BASE_DIR, "c101_customers.csv")
INFO_FILE = os.path.join(BASE_DIR, "c101_info.csv")
DISTANCE_FILE = os.path.join(BASE_DIR, "c101_距离矩阵.csv")

RESULT_METRICS = os.path.join(RESULT_DIR, "sa_metrics.csv")
RESULT_SUMMARY = os.path.join(RESULT_DIR, "sa_summary.txt")
RESULT_ROUTES_CSV = os.path.join(RESULT_DIR, "sa_routes.csv")
ROUTE_FIG = os.path.join(RESULT_DIR, "sa_routes.png")
LOG_FILE = os.path.join(LOG_DIR, "sa_run.log")

# ==================== 日志函数（仿贪心算法） ====================
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ==================== 硬约束参数 ====================
DRONE_MAX_LOAD = 200.0
DRONE_MAX_RANGE = 200.0
MAX_DRONES = 25
TIME_PENALTY_WEIGHT = 20.0
HARD_PENALTY_WEIGHT = 1000.0

class DroneSimulatedAnnealing:
    def __init__(self):
        self.load_data()
        self.init_parameters()
        self.nn_model = None
        self.scaler = None

    def load_data(self):
        log("加载数据...")

        self.df_customers = pd.read_csv(CUSTOMER_FILE)
        self.df_info = pd.read_csv(INFO_FILE)

        # 仓库信息
        self.depot_x = float(self.df_info['DEPOT_X'].iloc[0])
        self.depot_y = float(self.df_info['DEPOT_Y'].iloc[0])
        self.depot_ready_time = int(self.df_info['DEPOT_READY_TIME'].iloc[0])
        self.depot_due_date = int(self.df_info['DEPOT_DUE_DATE'].iloc[0])

        # 从 info 中读取车辆参数（如果有的话）
        if 'VEHICLE_CAPACITY' in self.df_info.columns:
            global DRONE_MAX_LOAD
            DRONE_MAX_LOAD = float(self.df_info['VEHICLE_CAPACITY'].iloc[0])
        if 'VEHICLE_NUMBER' in self.df_info.columns:
            global MAX_DRONES
            MAX_DRONES = int(self.df_info['VEHICLE_NUMBER'].iloc[0])

        self.n_customers = len(self.df_customers)

        # 坐标和属性数组
        self.coords = np.vstack([[[self.depot_x, self.depot_y]],
                                 self.df_customers[['X', 'Y']].values])
        self.demands = np.concatenate([[0], self.df_customers['DEMAND'].values])
        self.ready_times = np.concatenate([[self.depot_ready_time],
                                           self.df_customers['READY_TIME'].values])
        self.due_times = np.concatenate([[self.depot_due_date],
                                         self.df_customers['DUE_DATE'].values])
        self.service_times = np.concatenate([[0],
                                             self.df_customers['SERVICE_TIME'].values])

        # 距离矩阵：优先读取同目录文件，否则计算欧氏距离
        if os.path.exists(DISTANCE_FILE):
            self.distance_matrix = pd.read_csv(DISTANCE_FILE, header=None).values
            log(f"从文件加载距离矩阵: {DISTANCE_FILE}")

        log(f"仓库: ({self.depot_x}, {self.depot_y}), 时间窗=[{self.depot_ready_time}, {self.depot_due_date}]")
        log(f"客户数: {self.n_customers}, 最大载重: {DRONE_MAX_LOAD}, 最大航程: {DRONE_MAX_RANGE}, 最多无人机: {MAX_DRONES}")

    def _compute_distance_matrix(self):
        n = len(self.coords)
        self.distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.coords[i, 0] - self.coords[j, 0]
                    dy = self.coords[i, 1] - self.coords[j, 1]
                    self.distance_matrix[i, j] = np.sqrt(dx * dx + dy * dy)
        log(f"距离矩阵计算完成: {n}x{n}")

    def init_parameters(self):
        self.speed = 1.0
        self.initial_temperature = 1000.0
        self.final_temperature = 1e-6
        self.cooling_rate = 0.97
        self.iterations_per_temp = 300
        self.max_iterations = 120000
        log(f"SA参数: T0={self.initial_temperature}, cooling={self.cooling_rate}, "
            f"iter_per_temp={self.iterations_per_temp}, max_iter={self.max_iterations}")

    # ==================== 神经网络辅助 ====================
    def build_nn_model(self):
        log("训练神经网络（插入成本预测）...")
        features, targets = [], []
        for _ in range(1000):
            route = random.sample(range(1, self.n_customers + 1),
                                  k=random.randint(1, min(10, self.n_customers)))
            cust = random.choice([c for c in range(1, self.n_customers + 1) if c not in route])
            pos = random.randint(0, len(route))
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0
            cost_delta = (self.distance_matrix[prev, cust] +
                          self.distance_matrix[cust, nxt] -
                          self.distance_matrix[prev, nxt])
            feat = [self.distance_matrix[prev, cust], self.distance_matrix[cust, nxt],
                    self.demands[cust], self.ready_times[cust], self.due_times[cust],
                    len(route)]
            features.append(feat)
            targets.append(cost_delta)

        features = np.array(features)
        targets = np.array(targets)
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, targets, test_size=0.2, random_state=42
        )
        self.nn_model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu',
                                     max_iter=500, random_state=42)
        self.nn_model.fit(X_train, y_train)
        r2 = self.nn_model.score(X_test, y_test)
        log(f"神经网络 R² = {r2:.4f}")

    def predict_insertion_cost(self, route, cust, pos):
        if self.nn_model is None:
            return self._calc_insertion_cost(route, cust, pos)
        prev = route[pos - 1] if pos > 0 else 0
        nxt = route[pos] if pos < len(route) else 0
        feat = np.array([[self.distance_matrix[prev, cust],
                          self.distance_matrix[cust, nxt],
                          self.demands[cust], self.ready_times[cust],
                          self.due_times[cust], len(route)]])
        return max(0, self.nn_model.predict(self.scaler.transform(feat))[0])

    def _calc_insertion_cost(self, route, cust, pos):
        if not route:
            return 2 * self.distance_matrix[0, cust]
        prev = route[pos - 1] if pos > 0 else 0
        nxt = route[pos] if pos < len(route) else 0
        return (self.distance_matrix[prev, cust] +
                self.distance_matrix[cust, nxt] -
                self.distance_matrix[prev, nxt])

    # ==================== 评估函数 ====================
    def evaluate_solution(self, routes):
        total_dist = 0.0
        time_penalty = 0.0
        hard_penalty = 0.0
        violations = {'time': 0, 'capacity': 0, 'range': 0, 'drone_count': 0}

        if len(routes) > MAX_DRONES:
            violations['drone_count'] = len(routes) - MAX_DRONES
            hard_penalty += violations['drone_count'] * HARD_PENALTY_WEIGHT

        for route in routes:
            if not route:
                continue
            route_dist = 0.0
            curr_node = 0
            curr_time = self.depot_ready_time
            curr_load = 0.0
            for cust in route:
                d = self.distance_matrix[curr_node, cust]
                route_dist += d
                arrival = curr_time + d / self.speed
                if arrival < self.ready_times[cust]:
                    arrival = self.ready_times[cust]
                elif arrival > self.due_times[cust]:
                    overtime = arrival - self.due_times[cust]
                    time_penalty += overtime * TIME_PENALTY_WEIGHT
                    violations['time'] += 1
                curr_time = arrival + self.service_times[cust]
                curr_load += self.demands[cust]
                curr_node = cust
            route_dist += self.distance_matrix[curr_node, 0]
            total_dist += route_dist
            if curr_load > DRONE_MAX_LOAD:
                hard_penalty += (curr_load - DRONE_MAX_LOAD) * HARD_PENALTY_WEIGHT
                violations['capacity'] += 1
            if route_dist > DRONE_MAX_RANGE:
                hard_penalty += (route_dist - DRONE_MAX_RANGE) * HARD_PENALTY_WEIGHT
                violations['range'] += 1

        total_cost = total_dist + time_penalty + hard_penalty
        return total_cost, total_dist, time_penalty, hard_penalty, violations

    # ==================== 初始解生成 ====================
    def generate_initial_solution(self):
        log("生成初始解（严格限制无人机数 ≤ {}）...".format(MAX_DRONES))
        unvisited = set(range(1, self.n_customers + 1))
        routes = []
        cust_order = sorted(unvisited, key=lambda c: self.ready_times[c])

        while unvisited and len(routes) < MAX_DRONES:
            route = []
            curr_load = 0.0
            curr_time = self.depot_ready_time
            curr_node = 0
            for cust in cust_order:
                if cust not in unvisited:
                    continue
                if curr_load + self.demands[cust] > DRONE_MAX_LOAD:
                    continue
                travel = self.distance_matrix[curr_node, cust]
                arrival = curr_time + travel / self.speed
                if arrival > self.due_times[cust] + 10:
                    continue
                route.append(cust)
                curr_load += self.demands[cust]
                if arrival < self.ready_times[cust]:
                    arrival = self.ready_times[cust]
                curr_time = arrival + self.service_times[cust]
                curr_node = cust
                unvisited.remove(cust)
                if len(route) >= 15:
                    break
            if route:
                routes.append(route)
            else:
                if unvisited and len(routes) < MAX_DRONES:
                    cust = unvisited.pop()
                    routes.append([cust])

        if unvisited:
            log("无人机已达上限，剩余 {} 个客户强行插入现有路径".format(len(unvisited)))
            for cust in list(unvisited):
                best_route = -1
                best_pos = -1
                best_delta = float('inf')
                for i, route in enumerate(routes):
                    for pos in range(len(route) + 1):
                        delta = self._calc_insertion_cost(route, cust, pos)
                        if delta < best_delta:
                            best_delta = delta
                            best_route = i
                            best_pos = pos
                if best_route != -1:
                    routes[best_route].insert(best_pos, cust)
                    unvisited.remove(cust)

        log(f"初始解使用 {len(routes)} 架无人机")
        return routes

    # ==================== 邻域操作 ====================
    def neighbor_operation(self, routes):
        if not routes:
            return routes
        new_routes = [r.copy() for r in routes]
        op = random.choice(['2-opt', 'relocate', 'exchange', 'split', 'merge'])
        try:
            if op == '2-opt' and len(new_routes) > 0:
                idx = random.randint(0, len(new_routes) - 1)
                if len(new_routes[idx]) >= 2:
                    i = random.randint(0, len(new_routes[idx]) - 2)
                    j = random.randint(i + 1, len(new_routes[idx]) - 1)
                    new_routes[idx][i:j + 1] = reversed(new_routes[idx][i:j + 1])
            elif op == 'relocate':
                from_idx = random.randint(0, len(new_routes) - 1)
                if not new_routes[from_idx]:
                    return new_routes
                cust_pos = random.randint(0, len(new_routes[from_idx]) - 1)
                cust = new_routes[from_idx].pop(cust_pos)
                if not new_routes[from_idx]:
                    new_routes.pop(from_idx)
                    if not new_routes:
                        new_routes = [[cust]]
                        return new_routes
                if len(new_routes) < MAX_DRONES:
                    to_idx = random.randint(0, len(new_routes))
                    if to_idx == len(new_routes):
                        new_routes.append([cust])
                    else:
                        pos = random.randint(0, len(new_routes[to_idx]))
                        new_routes[to_idx].insert(pos, cust)
                else:
                    to_idx = random.randint(0, len(new_routes) - 1)
                    pos = random.randint(0, len(new_routes[to_idx]))
                    new_routes[to_idx].insert(pos, cust)
            elif op == 'exchange' and len(new_routes) >= 2:
                i1, i2 = random.sample(range(len(new_routes)), 2)
                if new_routes[i1] and new_routes[i2]:
                    p1 = random.randint(0, len(new_routes[i1]) - 1)
                    p2 = random.randint(0, len(new_routes[i2]) - 1)
                    new_routes[i1][p1], new_routes[i2][p2] = \
                        new_routes[i2][p2], new_routes[i1][p1]
            elif op == 'split' and len(new_routes) < MAX_DRONES:
                candidates = [i for i, r in enumerate(new_routes) if len(r) >= 2]
                if candidates:
                    idx = random.choice(candidates)
                    split_pos = random.randint(1, len(new_routes[idx]) - 1)
                    new_route = new_routes[idx][split_pos:]
                    new_routes[idx] = new_routes[idx][:split_pos]
                    new_routes.append(new_route)
            elif op == 'merge' and len(new_routes) >= 2:
                i1, i2 = random.sample(range(len(new_routes)), 2)
                new_routes[i1].extend(new_routes[i2])
                new_routes.pop(i2)
        except Exception:
            pass
        new_routes = [r for r in new_routes if r]
        return new_routes

    # ==================== 模拟退火主循环 ====================
    def run(self):
        start = time.time()
        self.build_nn_model()

        current_routes = self.generate_initial_solution()
        current_cost, current_dist, current_tp, current_hp, _ = self.evaluate_solution(current_routes)
        best_routes = [r.copy() for r in current_routes]
        best_cost = current_cost

        log(f"初始解成本: {current_cost:.2f} (距离={current_dist:.2f}, "
            f"时间罚={current_tp:.2f}, 硬罚={current_hp:.2f})")

        T = self.initial_temperature
        iteration = 0
        log_interval = max(5000, self.max_iterations // 10)

        while T > self.final_temperature and iteration < self.max_iterations:
            for _ in range(self.iterations_per_temp):
                new_routes = self.neighbor_operation(current_routes)
                new_cost, new_dist, new_tp, new_hp, _ = self.evaluate_solution(new_routes)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_routes = new_routes
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_routes = [r.copy() for r in current_routes]
                        best_cost = current_cost
                iteration += 1
            T *= self.cooling_rate
            if iteration % log_interval == 0 or iteration == 1:
                log(f"Iter {iteration:6d}, T={T:.4f}, best_cost={best_cost:.2f}, "
                    f"drones={len(best_routes)}")

        runtime = time.time() - start
        final_cost, final_dist, final_tp, final_hp, violations = self.evaluate_solution(best_routes)
        evaluation = {
            'total_cost': final_cost,
            'total_distance': final_dist,
            'time_penalty': final_tp,
            'hard_penalty': final_hp,
            'runtime': runtime,
            'violations': violations,
            'num_drones': len(best_routes)
        }
        log(f"优化完成: 成本={final_cost:.2f}, 距离={final_dist:.2f}, "
            f"无人机数={len(best_routes)}/{MAX_DRONES}, 耗时={runtime:.2f}s")

        return best_routes, evaluation

    # ==================== 获取每架无人机的详细指标 ====================
    def get_route_details(self, route):
        """返回单条路线的载重、航程、返回时间"""
        curr_load = sum(self.demands[c] for c in route)
        route_dist = 0.0
        curr_node = 0
        for cust in route:
            route_dist += self.distance_matrix[curr_node, cust]
            curr_node = cust
        route_dist += self.distance_matrix[curr_node, 0]

        curr_time = self.depot_ready_time
        curr_node = 0
        for cust in route:
            travel = self.distance_matrix[curr_node, cust]
            arrival = curr_time + travel / self.speed
            if arrival < self.ready_times[cust]:
                arrival = self.ready_times[cust]
            curr_time = arrival + self.service_times[cust]
            curr_node = cust
        return_time = curr_time + self.distance_matrix[curr_node, 0] / self.speed

        return curr_load, route_dist, return_time

    # ==================== 约束违反详细分析（仿贪心算法） ====================
    def analyze_solution(self, routes):
        lines = []
        lines.append("\n" + "=" * 85)
        header = (f"{'路线':<4} {'客户数':<6} {'总载重':<8} {'载重违规':<10} "
                  f"{'航程':<10} {'航程违规':<10} {'客户超时总长':<12} {'仓库超时':<10} {'可行':<6}")
        lines.append(header)
        lines.append("-" * 85)

        feasible_total = True
        for idx, route in enumerate(routes, start=1):
            if not route:
                continue
            curr_load, route_dist, return_time = self.get_route_details(route)

            # 客户超时总计
            total_cust_overtime = 0.0
            curr_time = self.depot_ready_time
            curr_node = 0
            for cust in route:
                travel = self.distance_matrix[curr_node, cust]
                arr = curr_time + travel / self.speed
                start = max(arr, self.ready_times[cust])
                over = max(0, start - self.due_times[cust])
                total_cust_overtime += over
                curr_time = start + self.service_times[cust]
                curr_node = cust

            depot_overtime = max(0, return_time - self.depot_due_date)
            load_viol = max(0, curr_load - DRONE_MAX_LOAD)
            range_viol = max(0, route_dist - DRONE_MAX_RANGE)

            feasible = (load_viol == 0 and range_viol == 0 and
                        total_cust_overtime == 0 and depot_overtime == 0)
            if not feasible:
                feasible_total = False

            lines.append(f"{idx:<4} {len(route):<6} {curr_load:<8.1f} {load_viol:<10.1f} "
                         f"{route_dist:<10.2f} {range_viol:<10.2f} {total_cust_overtime:<12.2f} "
                         f"{depot_overtime:<10.2f} {'✓' if feasible else '✗':<6}")

        lines.append("-" * 85)
        if feasible_total:
            lines.append("总体结论：所有路线均满足载重、航程、客户时间窗、仓库时间窗约束。")
        else:
            lines.append("总体结论：存在违反约束的路线（惩罚项已计入总成本，解不可直接用于实际，需继续优化）。")
        return "\n".join(lines), feasible_total

    # ==================== 打印最优解概览（仿贪心算法格式） ====================
    def print_solution_summary(self, routes, evaluation):
        log("\n======== 最优解概览 =========")
        log(f"总成本: {evaluation['total_cost']:.2f}")
        log(f"使用无人机数量: {len(routes)} / {MAX_DRONES}")

        for i, route in enumerate(routes, 1):
            curr_load, route_dist, return_time = self.get_route_details(route)
            route_str = str(route[:5]) + ('...' if len(route) > 5 else '')
            log(f"无人机 {i:2d}: 客户 {route_str:20s} | "
                f"载重 {curr_load:3.0f}/{DRONE_MAX_LOAD} | "
                f"航程 {route_dist:6.2f}/{DRONE_MAX_RANGE} | "
                f"返回时间 {return_time:6.2f}")

    # ==================== 保存指标 CSV（仿贪心算法） ====================
    def save_metrics(self, routes, evaluation):
        total_distance = evaluation['total_distance']
        time_penalty = evaluation['time_penalty']
        hard_penalty = evaluation['hard_penalty']
        total_cost = evaluation['total_cost']

        # 计算 makespan 和超载/超里程总计
        makespan = 0.0
        total_overload = 0.0
        total_over_dist = 0.0
        total_cust_overtime = 0.0
        for route in routes:
            curr_load, route_dist, return_time = self.get_route_details(route)
            total_overload += max(0, curr_load - DRONE_MAX_LOAD)
            total_over_dist += max(0, route_dist - DRONE_MAX_RANGE)
            if return_time > makespan:
                makespan = return_time
            # 客户超时
            curr_time = self.depot_ready_time
            curr_node = 0
            for cust in route:
                travel = self.distance_matrix[curr_node, cust]
                arr = curr_time + travel / self.speed
                start = max(arr, self.ready_times[cust])
                over = max(0, start - self.due_times[cust])
                total_cust_overtime += over
                curr_time = start + self.service_times[cust]
                curr_node = cust

        metrics = {
            "指标": [
                "总成本", "飞行距离成本", "原始距离(无惩罚)", "时间窗超时惩罚",
                "硬约束违规惩罚", "Makespan(最长任务时间)", "超时总时长",
                "超载量", "超里程量", "使用的无人机数", "总服务客户数",
                "算法运行时间(秒)"
            ],
            "数值": [
                f"{total_cost:.2f}", f"{total_distance:.2f}", f"{total_distance:.2f}",
                f"{time_penalty:.2f}", f"{hard_penalty:.2f}", f"{makespan:.2f}",
                f"{total_cust_overtime:.2f}", f"{total_overload:.1f}",
                f"{total_over_dist:.2f}", f"{len(routes)}", f"{self.n_customers}",
                f"{evaluation['runtime']:.2f}"
            ]
        }
        with open(RESULT_METRICS, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(metrics["指标"])
            writer.writerow(metrics["数值"])
        log(f"指标已保存至: {RESULT_METRICS}")

    # ==================== 保存文本摘要（仿贪心算法） ====================
    def save_summary(self, routes, evaluation, analysis_text):
        total_distance = evaluation['total_distance']
        time_penalty = evaluation['time_penalty']
        hard_penalty = evaluation['hard_penalty']

        with open(RESULT_SUMMARY, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("无人机配送路径规划 - 模拟退火算法结果\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"数据文件: {CUSTOMER_FILE}\n")
            f.write(f"配置文件: {INFO_FILE}\n")
            f.write(f"参数: 容量={DRONE_MAX_LOAD}, 航程上限={DRONE_MAX_RANGE}, "
                    f"最大无人机数={MAX_DRONES}\n")
            f.write(f"权重: 时间惩罚={TIME_PENALTY_WEIGHT}, "
                    f"硬约束惩罚={HARD_PENALTY_WEIGHT}\n")
            f.write(f"仓库: ({self.depot_x}, {self.depot_y}), "
                    f"时间窗=[{self.depot_ready_time}, {self.depot_due_date}]\n\n")
            f.write(f"总成本: {evaluation['total_cost']:.2f}\n")
            f.write(f"飞行距离: {total_distance:.2f}\n")
            f.write(f"时间窗超时惩罚: {time_penalty:.2f}\n")
            f.write(f"硬约束违规惩罚: {hard_penalty:.2f}\n")
            f.write(f"使用的无人机数: {len(routes)} / {MAX_DRONES}\n")
            f.write(f"算法运行时间: {evaluation['runtime']:.2f} 秒\n\n")
            f.write("=" * 60 + "\n")
            f.write("各无人机任务详情\n")
            f.write("=" * 60 + "\n")
            for i, route in enumerate(routes, 1):
                curr_load, route_dist, return_time = self.get_route_details(route)
                f.write(f"\n无人机 {i}: Load={curr_load:.0f}/{DRONE_MAX_LOAD}  "
                        f"Dist={route_dist:.2f}/{DRONE_MAX_RANGE}  "
                        f"Return={return_time:.2f}\n")
                f.write(f"  客户ID: {route}\n")
            f.write("\n" + analysis_text + "\n")
        log(f"摘要已保存至: {RESULT_SUMMARY}")

    # ==================== 保存路线明细 CSV ====================
    def save_routes_csv(self, routes):
        rows = []
        for i, route in enumerate(routes, 1):
            curr_load, route_dist, return_time = self.get_route_details(route)
            rows.append({
                '无人机ID': i,
                '客户序列': ' -> '.join(map(str, route)),
                '客户数': len(route),
                '总载重': f"{curr_load:.1f}",
                '总航程': f"{route_dist:.2f}",
                '返回时间': f"{return_time:.2f}",
                '载重违规': f"{max(0, curr_load - DRONE_MAX_LOAD):.1f}",
                '航程违规': f"{max(0, route_dist - DRONE_MAX_RANGE):.2f}"
            })
        pd.DataFrame(rows).to_csv(RESULT_ROUTES_CSV, index=False, encoding='utf-8-sig')
        log(f"路线明细已保存至: {RESULT_ROUTES_CSV}")

    # ==================== 绘制路线图 ====================
    def plot_routes(self, routes, evaluation):
        plt.figure(figsize=(12, 10))
        # 客户点
        cust_x = self.coords[1:, 0]
        cust_y = self.coords[1:, 1]
        plt.scatter(cust_x, cust_y, c='skyblue', edgecolors='black', s=60, zorder=2,
                    label='客户点')
        for i in range(1, self.n_customers + 1):
            plt.annotate(str(i), (self.coords[i, 0], self.coords[i, 1]),
                         fontsize=7, ha='center', va='center')
        # 仓库
        plt.scatter(self.depot_x, self.depot_y, c='red', marker='s', s=200,
                    edgecolors='black', label='仓库', zorder=3)
        plt.annotate('Depot', (self.depot_x, self.depot_y), fontsize=10,
                     ha='center', va='bottom')

        colors = plt.cm.tab20(np.linspace(0, 1, len(routes)))
        for i, route in enumerate(routes):
            if not route:
                continue
            path = [0] + route + [0]
            xs = [self.coords[idx, 0] for idx in path]
            ys = [self.coords[idx, 1] for idx in path]
            plt.plot(xs, ys, marker='o', linewidth=1.5, color=colors[i],
                     alpha=0.7, label=f'无人机 {i + 1}')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'SA 无人机配送路径 (总成本={evaluation["total_cost"]:.2f}, '
                  f'无人机数={len(routes)})')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(ROUTE_FIG, dpi=150, bbox_inches='tight')
        plt.close()
        log(f"路线图已保存至: {ROUTE_FIG}")

    # ==================== 保存所有结果 ====================
    def save_results(self, routes, evaluation):
        # 约束分析
        analysis_text, feasible = self.analyze_solution(routes)
        log(analysis_text)

        # 保存指标 CSV
        self.save_metrics(routes, evaluation)

        # 保存路线明细 CSV
        self.save_routes_csv(routes)

        # 保存文本摘要
        self.save_summary(routes, evaluation, analysis_text)

        # 绘制路线图
        self.plot_routes(routes, evaluation)


def main():
    log("=" * 60)
    log("无人机配送路径规划 - 模拟退火算法（严格限制无人机 ≤ 25）")
    log(f"数据目录: {BASE_DIR}")
    log("优化目标: 距离 + 20×超时时间 + 1000×(载重超限+航程超限+无人机超限)")
    log("=" * 60)

    random.seed(42)
    np.random.seed(42)

    solver = DroneSimulatedAnnealing()
    best_routes, evaluation = solver.run()

    # 打印最优解概览（仿贪心算法）
    solver.print_solution_summary(best_routes, evaluation)

    # 保存全部结果
    solver.save_results(best_routes, evaluation)

    print("\n" + "=" * 60)
    print("程序执行成功！")
    print(f"最终结果 - 总成本: {evaluation['total_cost']:.2f}, "
          f"使用无人机: {evaluation['num_drones']}/{MAX_DRONES}")
    print(f"结果已保存至: {RESULT_DIR}")
    print(f"日志已保存至: {LOG_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()