# drone_sa_final.py
"""
无人机配送路径规划 - 模拟退火算法（严格限制无人机 ≤ 25）
数据路径: D:/python/大作业/data/cleaned 和 D:/python/大作业/data/距离矩阵
优化目标: min 总距离 + 20*超时时间 + 1000*(载重超限+航程超限+无人机超限)
输出: 控制台打印最优解概览，并在 results 目录生成详细报告
"""

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
from typing import List, Tuple, Dict, Any, Optional
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 路径配置 ====================
DATA_ROOT = Path("D:/python/大作业")
CLEANED_DIR = DATA_ROOT / "data" / "cleaned"
DISTANCE_DIR = DATA_ROOT / "data" / "距离矩阵"
LOG_DIR = DATA_ROOT / "log"
RESULTS_DIR = DATA_ROOT / "results"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 硬约束参数 ====================
DRONE_MAX_LOAD = 200.0          # 最大载重
DRONE_MAX_RANGE = 200.0         # 最大航程
MAX_DRONES = 25                 # 最多可使用无人机数量（严格限制）
TIME_PENALTY_WEIGHT = 20.0
HARD_PENALTY_WEIGHT = 1000.0

class DroneSimulatedAnnealing:
    def __init__(self):
        self.setup_logging()
        self.load_data()
        self.init_parameters()
        self.nn_model = None
        self.scaler = None

    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"drone_sa_{timestamp}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file, encoding='utf-8'),
                      logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        self.logger.info("加载数据...")
        customers_path = CLEANED_DIR / "c101_customers.csv"
        info_path = CLEANED_DIR / "c101_info.csv"
        distance_path = DISTANCE_DIR / "c101_距离矩阵.csv"
        
        self.df_customers = pd.read_csv(customers_path)
        self.df_info = pd.read_csv(info_path)
        self.distance_matrix = pd.read_csv(distance_path, header=None).values
        
        self.depot_x = float(self.df_info['DEPOT_X'].iloc[0])
        self.depot_y = float(self.df_info['DEPOT_Y'].iloc[0])
        self.depot_ready_time = int(self.df_info['DEPOT_READY_TIME'].iloc[0])
        self.depot_due_date = int(self.df_info['DEPOT_DUE_DATE'].iloc[0])
        self.n_customers = len(self.df_customers)
        
        self.coords = np.vstack([[[self.depot_x, self.depot_y]],
                                 self.df_customers[['X', 'Y']].values])
        self.demands = np.concatenate([[0], self.df_customers['DEMAND'].values])
        self.ready_times = np.concatenate([[self.depot_ready_time],
                                           self.df_customers['READY_TIME'].values])
        self.due_times = np.concatenate([[self.depot_due_date],
                                         self.df_customers['DUE_DATE'].values])
        self.service_times = np.concatenate([[0],
                                             self.df_customers['SERVICE_TIME'].values])
        
        if self.distance_matrix.shape[0] != self.n_customers + 1:
            self.logger.warning("距离矩阵尺寸不匹配，重新计算欧氏距离")
            self._compute_distance_matrix()
        
        self.logger.info(f"客户数: {self.n_customers}, 最大载重: {DRONE_MAX_LOAD}, 最大航程: {DRONE_MAX_RANGE}, 最多无人机: {MAX_DRONES}")

    def _compute_distance_matrix(self):
        n = len(self.coords)
        self.distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.coords[i,0] - self.coords[j,0]
                    dy = self.coords[i,1] - self.coords[j,1]
                    self.distance_matrix[i,j] = np.sqrt(dx*dx + dy*dy)

    def init_parameters(self):
        self.speed = 1.0
        self.initial_temperature = 1000.0
        self.final_temperature = 1e-6
        self.cooling_rate = 0.97
        self.iterations_per_temp = 300
        self.max_iterations = 120000
        self.logger.info(f"SA参数: T0={self.initial_temperature}, cooling={self.cooling_rate}")

    # ==================== 神经网络辅助 ====================
    def build_nn_model(self):
        self.logger.info("训练神经网络（插入成本预测）...")
        features, targets = [], []
        for _ in range(1000):
            route = random.sample(range(1, self.n_customers+1), k=random.randint(1, min(10, self.n_customers)))
            cust = random.choice([c for c in range(1, self.n_customers+1) if c not in route])
            pos = random.randint(0, len(route))
            prev = route[pos-1] if pos>0 else 0
            nxt = route[pos] if pos<len(route) else 0
            cost_delta = self.distance_matrix[prev, cust] + self.distance_matrix[cust, nxt] - self.distance_matrix[prev, nxt]
            feat = [self.distance_matrix[prev, cust], self.distance_matrix[cust, nxt],
                    self.demands[cust], self.ready_times[cust], self.due_times[cust], len(route)]
            features.append(feat)
            targets.append(cost_delta)
        features = np.array(features)
        targets = np.array(targets)
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, targets, test_size=0.2, random_state=42)
        self.nn_model = MLPRegressor(hidden_layer_sizes=(32,16), activation='relu', max_iter=500, random_state=42)
        self.nn_model.fit(X_train, y_train)
        self.logger.info(f"神经网络R²: {self.nn_model.score(X_test, y_test):.4f}")

    def predict_insertion_cost(self, route, cust, pos):
        if self.nn_model is None:
            return self._calc_insertion_cost(route, cust, pos)
        prev = route[pos-1] if pos>0 else 0
        nxt = route[pos] if pos<len(route) else 0
        feat = np.array([[self.distance_matrix[prev,cust], self.distance_matrix[cust,nxt],
                          self.demands[cust], self.ready_times[cust], self.due_times[cust], len(route)]])
        return max(0, self.nn_model.predict(self.scaler.transform(feat))[0])

    def _calc_insertion_cost(self, route, cust, pos):
        if not route:
            return 2 * self.distance_matrix[0, cust]
        prev = route[pos-1] if pos>0 else 0
        nxt = route[pos] if pos<len(route) else 0
        return self.distance_matrix[prev, cust] + self.distance_matrix[cust, nxt] - self.distance_matrix[prev, nxt]

    # ==================== 评估函数 ====================
    def evaluate_solution(self, routes):
        total_dist = 0.0
        time_penalty = 0.0
        hard_penalty = 0.0
        violations = {'time':0, 'capacity':0, 'range':0, 'drone_count':0}
        
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
        self.logger.info("生成初始解（严格限制无人机数 ≤ {}）...".format(MAX_DRONES))
        unvisited = set(range(1, self.n_customers+1))
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
            self.logger.warning("无人机已达上限，剩余{}个客户强行插入现有路径".format(len(unvisited)))
            for cust in list(unvisited):
                best_route = -1
                best_pos = -1
                best_delta = float('inf')
                for i, route in enumerate(routes):
                    for pos in range(len(route)+1):
                        delta = self._calc_insertion_cost(route, cust, pos)
                        if delta < best_delta:
                            best_delta = delta
                            best_route = i
                            best_pos = pos
                if best_route != -1:
                    routes[best_route].insert(best_pos, cust)
                    unvisited.remove(cust)
        self.logger.info(f"初始解使用 {len(routes)} 架无人机")
        return routes

    # ==================== 邻域操作 ====================
    def neighbor_operation(self, routes):
        if not routes:
            return routes
        new_routes = [r.copy() for r in routes]
        op = random.choice(['2-opt', 'relocate', 'exchange', 'split', 'merge'])
        try:
            if op == '2-opt' and len(new_routes) > 0:
                idx = random.randint(0, len(new_routes)-1)
                if len(new_routes[idx]) >= 2:
                    i = random.randint(0, len(new_routes[idx])-2)
                    j = random.randint(i+1, len(new_routes[idx])-1)
                    new_routes[idx][i:j+1] = reversed(new_routes[idx][i:j+1])
            elif op == 'relocate':
                from_idx = random.randint(0, len(new_routes)-1)
                if not new_routes[from_idx]:
                    return new_routes
                cust_pos = random.randint(0, len(new_routes[from_idx])-1)
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
                    to_idx = random.randint(0, len(new_routes)-1)
                    pos = random.randint(0, len(new_routes[to_idx]))
                    new_routes[to_idx].insert(pos, cust)
            elif op == 'exchange' and len(new_routes) >= 2:
                i1, i2 = random.sample(range(len(new_routes)), 2)
                if new_routes[i1] and new_routes[i2]:
                    p1 = random.randint(0, len(new_routes[i1])-1)
                    p2 = random.randint(0, len(new_routes[i2])-1)
                    new_routes[i1][p1], new_routes[i2][p2] = new_routes[i2][p2], new_routes[i1][p1]
            elif op == 'split' and len(new_routes) < MAX_DRONES:
                idx = random.choice([i for i, r in enumerate(new_routes) if len(r) >= 2])
                split_pos = random.randint(1, len(new_routes[idx])-1)
                new_route = new_routes[idx][split_pos:]
                new_routes[idx] = new_routes[idx][:split_pos]
                new_routes.append(new_route)
            elif op == 'merge' and len(new_routes) >= 2:
                i1, i2 = random.sample(range(len(new_routes)), 2)
                new_routes[i1].extend(new_routes[i2])
                new_routes.pop(i2)
        except Exception as e:
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
        
        self.logger.info(f"初始解成本: {current_cost:.2f} (距离={current_dist:.2f}, 时间罚={current_tp:.2f}, 硬罚={current_hp:.2f})")
        
        T = self.initial_temperature
        iteration = 0
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
            if iteration % 5000 == 0:
                self.logger.info(f"Iter {iteration}, T={T:.4f}, best_cost={best_cost:.2f}")
        
        runtime_ms = (time.time() - start) * 1000
        final_cost, final_dist, final_tp, final_hp, violations = self.evaluate_solution(best_routes)
        evaluation = {
            'total_cost': final_cost,
            'total_distance': final_dist,
            'time_penalty': final_tp,
            'hard_penalty': final_hp,
            'runtime_ms': runtime_ms,
            'violations': violations,
            'num_drones': len(best_routes)
        }
        self.logger.info(f"优化完成: 成本={final_cost:.2f}, 距离={final_dist:.2f}, 无人机数={len(best_routes)}/{MAX_DRONES}, 耗时={runtime_ms:.2f}ms")
        
        # 控制台打印最优解概览（与示例一致）
        self.print_solution_summary(best_routes, evaluation)
        
        return best_routes, evaluation

    # ==================== 打印最优解概览（控制台输出） ====================
    def print_solution_summary(self, routes, evaluation):
        """按照要求的格式打印最优解概览"""
        print("\n======== 最优解概览 =========")
        print(f"总成本：{evaluation['total_cost']:.2f}")
        print(f"使用无人机数量：{len(routes)} / {MAX_DRONES}")
        
        for i, route in enumerate(routes, 1):
            # 载重
            total_demand = sum(self.demands[c] for c in route)
            # 航程
            route_dist = 0.0
            curr_node = 0
            for cust in route:
                route_dist += self.distance_matrix[curr_node, cust]
                curr_node = cust
            route_dist += self.distance_matrix[curr_node, 0]
            # 返回时间
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
            
            print(f"无人机 {i}: 客户 {route} | 载重 {total_demand}/{DRONE_MAX_LOAD} | "
                  f"航程 {route_dist:.2f}/{DRONE_MAX_RANGE} | 返回时间 {return_time:.2f}")
        print("=" * 40)

    # ==================== 保存结果与绘图 ====================
    def plot_solution(self, routes, evaluation, save_dir):
        plt.figure(figsize=(12,10))
        plt.scatter(self.depot_x, self.depot_y, c='red', s=200, marker='*', label='仓库')
        cust_x = self.coords[1:,0]; cust_y = self.coords[1:,1]
        plt.scatter(cust_x, cust_y, c='gray', s=50, alpha=0.7, label='客户点')
        colors = plt.cm.tab20(np.linspace(0,1,len(routes)))
        for i, route in enumerate(routes):
            if not route: continue
            path = [0] + route + [0]
            xs = [self.coords[idx,0] for idx in path]
            ys = [self.coords[idx,1] for idx in path]
            plt.plot(xs, ys, marker='o', linewidth=2, color=colors[i], label=f'无人机{i+1}({len(route)}客户)')
        plt.title(f'无人机路径规划 (模拟退火)\n总成本={evaluation["total_cost"]:.2f}  总距离={evaluation["total_distance"]:.2f}')
        plt.xlabel('X'); plt.ylabel('Y'); plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_dir/"route_map.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, routes, evaluation):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = RESULTS_DIR / f"drone_sa_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 路线明细
        pd.DataFrame([{'无人机ID':i+1, '客户序列':'->'.join(map(str,r)), '客户数':len(r), 
                       '总需求':sum(self.demands[c] for c in r)} for i,r in enumerate(routes) if r]
                    ).to_csv(out_dir/"routes.csv", index=False, encoding='utf-8-sig')
        # 成本对比
        pd.DataFrame([['总成本', evaluation['total_cost']],
                      ['飞行距离成本', evaluation['total_distance']],
                      ['时间窗超时惩罚', evaluation['time_penalty']],
                      ['硬约束违规惩罚', evaluation['hard_penalty']]],
                     columns=['成本项','数值']).to_csv(out_dir/"cost_breakdown.csv", index=False, encoding='utf-8-sig')
        # 违反分析
        v = evaluation['violations']
        pd.DataFrame([['时间窗违反次数', v['time']],
                      ['载重超限次数', v['capacity']],
                      ['航程超限次数', v['range']],
                      ['无人机数量超限次数', v['drone_count']],
                      ['总违反次数', sum(v.values())]],
                     columns=['违反类型','次数']).to_csv(out_dir/"violation_analysis.csv", index=False, encoding='utf-8-sig')
        # 汇总
        pd.DataFrame([{'算法':'模拟退火+神经网络', '总成本':evaluation['total_cost'],
                       '总距离':evaluation['total_distance'], '运行时间(ms)':evaluation['runtime_ms'],
                       '总违反次数':sum(v.values()), '使用无人机数':evaluation['num_drones'],
                       '允许最大无人机数':MAX_DRONES, '客户覆盖率':f"{len(set().union(*[set(r) for r in routes]))}/{self.n_customers}"}]
                    ).to_csv(out_dir/"summary.csv", index=False, encoding='utf-8-sig')
        self.plot_solution(routes, evaluation, out_dir)
        
        report = f"""# 无人机配送路径规划报告（模拟退火）

## 优化目标
综合成本 = 总飞行距离 + 20×超时时间 + 1000×(载重超限+航程超限+无人机超限)

## 最终结果
- 总成本: {evaluation['total_cost']:.2f}
- 总飞行距离: {evaluation['total_distance']:.2f}
- 时间窗惩罚: {evaluation['time_penalty']:.2f}
- 硬约束惩罚: {evaluation['hard_penalty']:.2f}
- 运行时间: {evaluation['runtime_ms']:.2f} ms
- 使用无人机数: {evaluation['num_drones']} / {MAX_DRONES}

## 约束违反分析
- 时间窗违反: {v['time']}
- 载重超限: {v['capacity']}
- 航程超限: {v['range']}
- 无人机超限: {v['drone_count']}
- 总违反: {sum(v.values())}

## 成本占比
| 成本项 | 数值 | 占比 |
|--------|------|------|
| 飞行距离 | {evaluation['total_distance']:.2f} | {evaluation['total_distance']/evaluation['total_cost']*100:.1f}% |
| 时间惩罚 | {evaluation['time_penalty']:.2f} | {evaluation['time_penalty']/evaluation['total_cost']*100:.1f}% |
| 硬约束惩罚 | {evaluation['hard_penalty']:.2f} | {evaluation['hard_penalty']/evaluation['total_cost']*100:.1f}% |
"""
        with open(out_dir/"report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        self.logger.info(f"结果保存至 {out_dir}")
        return out_dir

def main():
    print("="*60)
    print("无人机配送路径规划 - 模拟退火算法（严格限制无人机 ≤ 25）")
    print(f"数据路径: {DATA_ROOT}/data/cleaned")
    print("优化目标: 距离 + 20*超时时间 + 1000*(载重超限+航程超限+无人机超限)")
    print("="*60)
    solver = DroneSimulatedAnnealing()
    best_routes, evaluation = solver.run()
    out_dir = solver.save_results(best_routes, evaluation)
    print(f"\n✅ 优化完成！详细结果已保存至: {out_dir}")
    print(f"   总成本 = {evaluation['total_cost']:.2f}")
    print(f"   总距离 = {evaluation['total_distance']:.2f}")
    print(f"   运行时间 = {evaluation['runtime_ms']:.2f} ms")
    print(f"   使用无人机 = {evaluation['num_drones']} / {MAX_DRONES}")

if __name__ == "__main__":
    main()