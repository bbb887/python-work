# scripts/5_simulated_annealing_vrp.py
"""
基于C101标准数据集的VRP模拟退火算法
修复版 - 解决车辆数过多问题
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
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 路径配置 ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

CLEANED_DIR = BASE_DIR / "data" / "cleaned"
DISTANCE_DIR = BASE_DIR / "data" / "距离矩阵"

LOG_DIR = Path(r"E:\无人机\drone_delivery_eda\log")
RESULTS_DIR = Path(r"E:\无人机\drone_delivery_eda\results")

LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class Solution:
    """解决方案类，表示一个解"""
    
    def __init__(self, routes: List[List[int]], cost: float = None):
        self.routes = routes  # 存储路径
        self.cost = cost
        self.distance = 0
        self.violation = 0
        self.vehicle_penalty = 0
        self.total_vehicles = len(routes)
        
    def __lt__(self, other):
        return self.cost < other.cost if self.cost is not None and other.cost is not None else False
    
    def __repr__(self):
        return f"Solution(cost={self.cost:.2f}, vehicles={self.total_vehicles}, distance={self.distance:.2f})"

class C101SimulatedAnnealingVRP:
    """基于C101标准数据集的VRP模拟退火算法"""
    
    def __init__(self, data_dir="data"):
        """初始化"""
        self.data_dir = Path(data_dir)
        self.setup_logging()
        
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("C101 数据集VRP求解 - 模拟退火算法")
        logger.info("=" * 60)
        
        self.load_data()
        self.init_parameters()
        
    def setup_logging(self):
        """设置日志系统"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"c101_simulated_annealing_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def load_data(self):
        """加载C101数据集"""
        logger = logging.getLogger(__name__)
        logger.info("正在加载C101数据集...")
        
        try:
            customers_path = CLEANED_DIR / "rc101_customers.csv"
            self.df_customers = pd.read_csv(customers_path)
            
            info_path = CLEANED_DIR / "rc101_info.csv"
            self.df_info = pd.read_csv(info_path)
            
            distance_path = DISTANCE_DIR / "rc101_距离矩阵.csv"
            self.distance_matrix = pd.read_csv(distance_path, header=None).values
            
            self.problem_name = self.df_info['PROBLEM_NAME'].iloc[0]
            self.n_vehicles = int(self.df_info['VEHICLE_NUMBER'].iloc[0])
            self.vehicle_capacity = int(self.df_info['VEHICLE_CAPACITY'].iloc[0])
            self.depot_x = float(self.df_info['DEPOT_X'].iloc[0])
            self.depot_y = float(self.df_info['DEPOT_Y'].iloc[0])
            self.depot_ready_time = int(self.df_info['DEPOT_READY_TIME'].iloc[0])
            self.depot_due_date = int(self.df_info['DEPOT_DUE_DATE'].iloc[0])
            
            self.n_customers = len(self.df_customers)
            
            self.coords = np.vstack([
                [[self.depot_x, self.depot_y]],
                self.df_customers[['X', 'Y']].values
            ])
            
            self.demands = np.concatenate([
                [0],
                self.df_customers['DEMAND'].values
            ])
            
            self.ready_times = np.concatenate([
                [self.depot_ready_time],
                self.df_customers['READY_TIME'].values
            ])
            self.due_times = np.concatenate([
                [self.depot_due_date],
                self.df_customers['DUE_DATE'].values
            ])
            
            self.service_times = np.concatenate([
                [0],
                self.df_customers['SERVICE_TIME'].values
            ])
            
            expected_size = self.n_customers + 1
            actual_size = self.distance_matrix.shape[0]
            
            if expected_size != actual_size:
                logger.warning(f"距离矩阵大小不匹配: 期望 {expected_size}x{expected_size}, 实际 {actual_size}x{self.distance_matrix.shape[1]}")
                if expected_size > actual_size:
                    logger.info("将自动计算缺失部分...")
                    self.compute_distance_matrix()
                else:
                    self.distance_matrix = self.distance_matrix[:expected_size, :expected_size]
            
            logger.info("数据加载完成:")
            logger.info(f"  问题名称: {self.problem_name}")
            logger.info(f"  客户点数量: {self.n_customers}")
            logger.info(f"  车辆数量: {self.n_vehicles}")
            logger.info(f"  车辆容量: {self.vehicle_capacity}")
            logger.info(f"  总需求: {self.demands[1:].sum()}")
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def compute_distance_matrix(self):
        """计算欧氏距离矩阵"""
        logger = logging.getLogger(__name__)
        logger.info("计算欧氏距离矩阵...")
        
        n = len(self.coords)
        self.distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.coords[i, 0] - self.coords[j, 0]
                    dy = self.coords[i, 1] - self.coords[j, 1]
                    self.distance_matrix[i, j] = np.sqrt(dx**2 + dy**2)
        
        logger.info(f"距离矩阵计算完成: {self.distance_matrix.shape}")
    
    def init_parameters(self):
        """初始化算法参数"""
        # 基本参数
        self.speed = 1.0
        self.time_penalty_weight = 100
        self.capacity_penalty_weight = 1000
        
        # 模拟退火参数
        self.initial_temperature = 1000.0
        self.final_temperature = 1e-8
        self.cooling_rate = 0.95
        self.iterations_per_temperature = 100
        self.max_iterations = 10000
        
        # 车辆数惩罚权重
        self.vehicle_penalty_weight = 10000
        
        logger = logging.getLogger(__name__)
        logger.info("\n模拟退火算法参数:")
        logger.info(f"  初始温度: {self.initial_temperature}")
        logger.info(f"  终止温度: {self.final_temperature}")
        logger.info(f"  降温系数: {self.cooling_rate}")
        logger.info(f"  每个温度迭代次数: {self.iterations_per_temperature}")
        logger.info(f"  最大迭代次数: {self.max_iterations}")
        logger.info(f"  车辆数惩罚权重: {self.vehicle_penalty_weight}")
    
    def generate_initial_solution(self) -> List[List[int]]:
        """生成初始解 - 改进的最近邻算法"""
        logger = logging.getLogger(__name__)
        logger.info("生成初始解...")
        
        unvisited = set(range(1, self.n_customers + 1))
        routes = []
        
        # 计算每个客户的时间窗紧迫性
        customer_info = []
        for cust in range(1, self.n_customers + 1):
            time_window_length = self.due_times[cust] - self.ready_times[cust]
            urgency = self.ready_times[cust] + time_window_length * 0.3
            customer_info.append((cust, urgency))
        
        # 按紧迫性排序
        customer_info.sort(key=lambda x: x[1])
        sorted_customers = [c[0] for c in customer_info]
        
        # 贪心分配客户到路径
        while unvisited and len(routes) < self.n_vehicles:
            current_route = []
            current_load = 0
            current_time = self.depot_ready_time
            last_node = 0
            
            for cust in sorted_customers:
                if cust not in unvisited:
                    continue
                    
                demand = self.demands[cust]
                if current_load + demand > self.vehicle_capacity:
                    # 当前路径容量已满
                    break
                
                # 检查时间窗
                travel_time = self.distance_matrix[last_node, cust] / self.speed
                arrival_time = current_time + travel_time
                ready_time = self.ready_times[cust]
                due_time = self.due_times[cust]
                
                if arrival_time < ready_time:
                    arrival_time = ready_time
                elif arrival_time > due_time:
                    continue  # 时间窗违反，跳过
                
                # 添加客户到当前路径
                current_route.append(cust)
                current_load += demand
                current_time = arrival_time + self.service_times[cust]
                last_node = cust
                unvisited.remove(cust)
            
            if current_route:
                routes.append(current_route)
            else:
                break
        
        # 如果还有未分配的客户，尝试分配到现有路径
        if unvisited:
            routes = self.insert_remaining_customers(routes, unvisited)
        
        logger.info(f"初始解生成完成: {len(routes)} 条路径")
        return routes
    
    def insert_remaining_customers(self, routes: List[List[int]], unvisited: set) -> List[List[int]]:
        """将剩余客户插入到现有路径中"""
        for cust in list(unvisited):
            best_route_idx = -1
            best_position = -1
            best_cost = float('inf')
            
            for route_idx, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    # 计算插入后的路径总需求
                    temp_route = route.copy()
                    temp_route.insert(pos, cust)
                    route_demand = sum(self.demands[c] for c in temp_route)
                    
                    if route_demand > self.vehicle_capacity:
                        continue
                    
                    # 计算插入成本
                    insert_cost = self.calculate_insertion_cost(route, cust, pos)
                    if insert_cost < best_cost:
                        best_cost = insert_cost
                        best_route_idx = route_idx
                        best_position = pos
            
            if best_route_idx != -1:
                routes[best_route_idx].insert(best_position, cust)
                unvisited.remove(cust)
            elif len(routes) < self.n_vehicles:
                # 创建新路径
                routes.append([cust])
                unvisited.remove(cust)
        
        return routes
    
    def calculate_insertion_cost(self, route: List[int], customer: int, position: int) -> float:
        """计算插入客户的成本"""
        if not route:
            return 2 * self.distance_matrix[0, customer]
        
        if position == 0:
            prev_node = 0
            next_node = route[0]
        elif position == len(route):
            prev_node = route[-1]
            next_node = 0
        else:
            prev_node = route[position - 1]
            next_node = route[position]
        
        original_distance = self.distance_matrix[prev_node, next_node] if prev_node != 0 or next_node != 0 else 0
        new_distance = self.distance_matrix[prev_node, customer] + self.distance_matrix[customer, next_node]
        
        return new_distance - original_distance
    
    def evaluate_solution(self, routes: List[List[int]]) -> Tuple[float, float, float, float]:
        """评估解决方案，返回总距离、总违反惩罚、车辆数惩罚、总成本"""
        total_distance = 0
        total_violation = 0
        
        for route in routes:
            if not route:
                continue
            
            # 计算路径距离
            current_node = 0
            distance = 0
            
            for cust in route:
                distance += self.distance_matrix[current_node, cust]
                current_node = cust
            
            # 返回仓库
            distance += self.distance_matrix[current_node, 0]
            total_distance += distance
            
            # 计算违反惩罚
            violation = self.calculate_route_violation(route)
            total_violation += violation
        
        # 车辆数惩罚
        vehicle_penalty = max(0, len(routes) - self.n_vehicles) * self.vehicle_penalty_weight
        
        # 总成本
        total_cost = total_distance + total_violation + vehicle_penalty
        
        return total_distance, total_violation, vehicle_penalty, total_cost
    
    def calculate_route_violation(self, route: List[int]) -> float:
        """计算单条路径的违反惩罚"""
        violation = 0
        current_time = self.depot_ready_time
        current_node = 0
        current_load = 0
        
        for cust in route:
            # 距离和时间
            travel_time = self.distance_matrix[current_node, cust] / self.speed
            arrival_time = current_time + travel_time
            
            # 时间窗违反
            ready_time = self.ready_times[cust]
            due_time = self.due_times[cust]
            
            if arrival_time < ready_time:
                arrival_time = ready_time
            elif arrival_time > due_time:
                time_violation = arrival_time - due_time
                violation += time_violation * self.time_penalty_weight
            
            # 服务时间
            departure_time = arrival_time + self.service_times[cust]
            current_time = departure_time
            
            # 容量违反
            current_load += self.demands[cust]
            if current_load > self.vehicle_capacity:
                capacity_violation = current_load - self.vehicle_capacity
                violation += capacity_violation * self.capacity_penalty_weight
            
            current_node = cust
        
        # 返回仓库的时间窗
        return_time = current_time + self.distance_matrix[current_node, 0] / self.speed
        if return_time > self.depot_due_date:
            time_violation = return_time - self.depot_due_date
            violation += time_violation * self.time_penalty_weight
        
        return violation
    
    def generate_neighbor_solution(self, current_solution: List[List[int]]) -> List[List[int]]:
        """生成邻域解 - 通过多种扰动操作"""
        new_routes = [route.copy() for route in current_solution]
        
        # 随机选择一种扰动操作
        operation = random.choice(['2-opt', 'relocate', 'exchange', 'split', 'merge'])
        
        try:
            if operation == '2-opt' and len(new_routes) > 0:
                # 2-opt操作：反转路径中的一段
                route_idx = random.randint(0, len(new_routes) - 1)
                if len(new_routes[route_idx]) >= 2:
                    i = random.randint(0, len(new_routes[route_idx]) - 2)
                    j = random.randint(i + 1, len(new_routes[route_idx]) - 1)
                    new_routes[route_idx][i:j+1] = list(reversed(new_routes[route_idx][i:j+1]))
            
            elif operation == 'relocate':
                # 将一个客户从一个路径移到另一个路径
                from_route_idx = random.choice([i for i, route in enumerate(new_routes) if route])
                if not new_routes[from_route_idx]:
                    return new_routes
                
                cust_idx = random.randint(0, len(new_routes[from_route_idx]) - 1)
                customer = new_routes[from_route_idx].pop(cust_idx)
                
                if not new_routes[from_route_idx]:
                    new_routes.pop(from_route_idx)
                    from_route_idx = from_route_idx if from_route_idx < len(new_routes) else 0
                
                to_route_idx = random.choice([i for i in range(len(new_routes) + 1) if i != from_route_idx or len(new_routes) == 0])
                
                if to_route_idx < len(new_routes):
                    insert_idx = random.randint(0, len(new_routes[to_route_idx]))
                    new_routes[to_route_idx].insert(insert_idx, customer)
                elif len(new_routes) < self.n_vehicles:
                    new_routes.append([customer])
            
            elif operation == 'exchange':
                # 交换两个路径中的客户
                if len(new_routes) >= 2:
                    route1_idx, route2_idx = random.sample(range(len(new_routes)), 2)
                    if new_routes[route1_idx] and new_routes[route2_idx]:
                        cust1_idx = random.randint(0, len(new_routes[route1_idx]) - 1)
                        cust2_idx = random.randint(0, len(new_routes[route2_idx]) - 1)
                        new_routes[route1_idx][cust1_idx], new_routes[route2_idx][cust2_idx] = \
                            new_routes[route2_idx][cust2_idx], new_routes[route1_idx][cust1_idx]
            
            elif operation == 'split':
                # 分割一条路径为两条
                route_idx = random.choice([i for i, route in enumerate(new_routes) if len(route) >= 2])
                if len(new_routes[route_idx]) >= 2 and len(new_routes) < self.n_vehicles:
                    split_point = random.randint(1, len(new_routes[route_idx]) - 1)
                    new_route = new_routes[route_idx][split_point:]
                    new_routes[route_idx] = new_routes[route_idx][:split_point]
                    new_routes.append(new_route)
            
            elif operation == 'merge':
                # 合并两条路径
                if len(new_routes) >= 2:
                    route1_idx, route2_idx = random.sample(range(len(new_routes)), 2)
                    combined_route = new_routes[route1_idx] + new_routes[route2_idx]
                    route_demand = sum(self.demands[c] for c in combined_route)
                    
                    if route_demand <= self.vehicle_capacity:
                        new_routes[route1_idx] = combined_route
                        new_routes.pop(route2_idx)
        
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.debug(f"扰动操作失败: {e}")
            return current_solution
        
        # 清理空路径
        new_routes = [route for route in new_routes if route]
        
        return new_routes
    
    def repair_solution(self, routes: List[List[int]]) -> List[List[int]]:
        """修复解决方案，确保满足约束条件"""
        # 检查所有客户是否都包含在解决方案中
        all_customers = set(range(1, self.n_customers + 1))
        served_customers = set()
        for route in routes:
            served_customers.update(route)
        
        missing_customers = all_customers - served_customers
        
        # 添加遗漏的客户
        for cust in missing_customers:
            best_route_idx = -1
            best_position = -1
            best_cost = float('inf')
            
            for route_idx, route in enumerate(routes):
                route_demand = sum(self.demands[c] for c in route)
                if route_demand + self.demands[cust] <= self.vehicle_capacity:
                    for pos in range(len(route) + 1):
                        cost = self.calculate_insertion_cost(route, cust, pos)
                        if cost < best_cost:
                            best_cost = cost
                            best_route_idx = route_idx
                            best_position = pos
            
            if best_route_idx != -1:
                routes[best_route_idx].insert(best_position, cust)
            elif len(routes) < self.n_vehicles:
                routes.append([cust])
        
        # 修复容量违反
        repaired_routes = []
        for route in routes:
            route_demand = sum(self.demands[c] for c in route)
            
            if route_demand <= self.vehicle_capacity:
                repaired_routes.append(route)
            else:
                # 分割超容量的路径
                sub_route = []
                sub_demand = 0
                
                for cust in route:
                    if sub_demand + self.demands[cust] <= self.vehicle_capacity:
                        sub_route.append(cust)
                        sub_demand += self.demands[cust]
                    else:
                        if sub_route:
                            repaired_routes.append(sub_route)
                        sub_route = [cust]
                        sub_demand = self.demands[cust]
                
                if sub_route:
                    repaired_routes.append(sub_route)
        
        return repaired_routes
    
    def simulated_annealing(self) -> Tuple[List[List[int]], Dict[str, Any]]:
        """模拟退火算法主函数"""
        logger = logging.getLogger(__name__)
        logger.info("\n开始模拟退火优化...")
        start_time = time.time()
        
        # 初始化当前解
        current_routes = self.generate_initial_solution()
        current_distance, current_violation, current_vehicle_penalty, current_cost = self.evaluate_solution(current_routes)
        
        # 初始化最佳解
        best_routes = [route.copy() for route in current_routes]
        best_distance, best_violation, best_vehicle_penalty, best_cost = \
            current_distance, current_violation, current_vehicle_penalty, current_cost
        
        # 初始化温度
        temperature = self.initial_temperature
        
        # 记录优化过程
        temperature_history = []
        best_cost_history = []
        current_cost_history = []
        acceptance_rate_history = []
        vehicle_count_history = []
        
        iteration = 0
        accepted_count = 0
        total_neighbors = 0
        
        logger.info(f"初始解: 成本={current_cost:.2f}, 距离={current_distance:.2f}, 违反={current_violation:.2f}, 车辆数={len(current_routes)}")
        
        while temperature > self.final_temperature and iteration < self.max_iterations:
            accepted_in_cycle = 0
            tested_in_cycle = 0
            
            for _ in range(self.iterations_per_temperature):
                # 生成邻域解
                new_routes = self.generate_neighbor_solution(current_routes)
                new_routes = self.repair_solution(new_routes)
                
                # 评估新解
                new_distance, new_violation, new_vehicle_penalty, new_cost = self.evaluate_solution(new_routes)
                
                # 计算成本差
                delta_cost = new_cost - current_cost
                accepted = False
                
                # Metropolis准则
                if delta_cost < 0:
                    # 接受更好解
                    current_routes = new_routes
                    current_cost = new_cost
                    current_distance = new_distance
                    current_violation = new_violation
                    accepted = True
                    accepted_in_cycle += 1
                else:
                    # 以概率接受劣解
                    acceptance_probability = math.exp(-delta_cost / temperature)
                    if random.random() < acceptance_probability:
                        current_routes = new_routes
                        current_cost = new_cost
                        current_distance = new_distance
                        current_violation = new_violation
                        accepted = True
                        accepted_in_cycle += 1
                
                # 更新最佳解
                if new_cost < best_cost:
                    best_routes = [route.copy() for route in new_routes]
                    best_cost = new_cost
                    best_distance = new_distance
                    best_violation = new_violation
                
                tested_in_cycle += 1
                iteration += 1
            
            # 计算接受率
            acceptance_rate = accepted_in_cycle / tested_in_cycle if tested_in_cycle > 0 else 0
            
            # 记录历史
            temperature_history.append(temperature)
            best_cost_history.append(best_cost)
            current_cost_history.append(current_cost)
            acceptance_rate_history.append(acceptance_rate)
            vehicle_count_history.append(len(current_routes))
            
            # 输出进度
            if iteration % (self.iterations_per_temperature * 10) == 0 or temperature <= self.final_temperature:
                logger.info(f"迭代 {iteration:4d}: 温度={temperature:.6f}, "
                          f"最佳成本={best_cost:.2f}, 当前成本={current_cost:.2f}, "
                          f"接受率={acceptance_rate:.2%}, 车辆数={len(current_routes)}")
            
            # 降温
            temperature *= self.cooling_rate
        
        runtime = time.time() - start_time
        
        # 最终评估
        evaluation = self.final_evaluate_solution(best_routes, "模拟退火算法")
        evaluation.update({
            'runtime': runtime,
            'iterations': iteration,
            'best_cost_history': best_cost_history,
            'current_cost_history': current_cost_history,
            'temperature_history': temperature_history,
            'acceptance_rate_history': acceptance_rate_history,
            'vehicle_count_history': vehicle_count_history
        })
        
        logger.info("\n模拟退火完成:")
        logger.info(f"  总运行时间: {runtime:.3f} 秒")
        logger.info(f"  总迭代次数: {iteration}")
        logger.info(f"  最终成本: {evaluation['total_cost']:.2f}")
        logger.info(f"  总距离: {evaluation['total_distance']:.2f}")
        logger.info(f"  总违反惩罚: {evaluation['total_violation']:.2f}")
        logger.info(f"  使用车辆数: {evaluation['vehicles_used']}/{self.n_vehicles}")
        
        # 绘制收敛曲线
        self.plot_convergence_curve(evaluation)
        
        return best_routes, evaluation
    
    def final_evaluate_solution(self, routes: List[List[int]], algorithm_name: str) -> Dict[str, Any]:
        """最终评估解决方案的质量"""
        total_distance = 0
        total_violation = 0
        details = []
        served_customers = set()
        
        for vehicle_idx, route in enumerate(routes):
            if not route:
                continue
            
            # 计算路径
            current_node = 0
            current_time = self.depot_ready_time
            current_load = 0
            distance = 0
            violation = 0
            time_window_violations = 0
            capacity_violations = 0
            
            for customer_idx in route:
                # 计算距离
                d = self.distance_matrix[current_node, customer_idx]
                distance += d
                travel_time = d / self.speed
                
                # 更新到达时间
                arrival_time = current_time + travel_time
                
                # 检查时间窗
                ready_time = self.ready_times[customer_idx]
                due_time = self.due_times[customer_idx]
                service_time = self.service_times[customer_idx]
                
                if arrival_time < ready_time:
                    arrival_time = ready_time
                elif arrival_time > due_time:
                    time_violation = arrival_time - due_time
                    violation += time_violation * self.time_penalty_weight
                    time_window_violations += 1
                
                # 服务时间
                departure_time = arrival_time + service_time
                current_time = departure_time
                
                # 更新载重
                demand = self.demands[customer_idx]
                current_load += demand
                if current_load > self.vehicle_capacity:
                    capacity_violations += 1
                    violation += (current_load - self.vehicle_capacity) * self.capacity_penalty_weight
                
                served_customers.add(customer_idx)
                current_node = customer_idx
            
            # 返回仓库
            d = self.distance_matrix[current_node, 0]
            distance += d
            
            # 检查返回时间窗
            return_time = current_time + d / self.speed
            if return_time > self.depot_due_date:
                time_violation = return_time - self.depot_due_date
                violation += time_violation * self.time_penalty_weight
                time_window_violations += 1
            
            total_distance += distance
            total_violation += violation
            
            details.append({
                'vehicle': vehicle_idx + 1,
                'distance': distance,
                'load': current_load,
                'time_window_violations': time_window_violations,
                'capacity_violations': capacity_violations,
                'customers_served': len(route)
            })
        
        # 车辆数惩罚
        vehicle_penalty = max(0, len([r for r in routes if r]) - self.n_vehicles) * self.vehicle_penalty_weight
        
        # 总成本
        total_cost = total_distance + total_violation + vehicle_penalty
        
        # 计算覆盖率
        served_count = len(served_customers)
        coverage_rate = served_count / self.n_customers * 100
        
        result = {
            'algorithm': algorithm_name,
            'total_cost': total_cost,
            'total_distance': total_distance,
            'total_violation': total_violation,
            'vehicle_penalty': vehicle_penalty,
            'coverage_rate': coverage_rate,
            'vehicles_used': len([r for r in routes if r]),
            'unserved_customers': self.n_customers - served_count,
            'details': details
        }
        
        return result
    
    def plot_convergence_curve(self, evaluation: Dict[str, Any]):
        """绘制收敛曲线"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            iterations = list(range(len(evaluation['best_cost_history'])))
            
            # 1. 成本收敛曲线
            axes[0, 0].plot(iterations, evaluation['best_cost_history'], 'b-', linewidth=2, label='最佳成本')
            axes[0, 0].plot(iterations, evaluation['current_cost_history'], 'r-', alpha=0.7, label='当前成本')
            axes[0, 0].set_xlabel('温度循环')
            axes[0, 0].set_ylabel('成本')
            axes[0, 0].set_title('成本收敛曲线')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 温度变化曲线
            axes[0, 1].plot(iterations, evaluation['temperature_history'], 'g-')
            axes[0, 1].set_xlabel('温度循环')
            axes[0, 1].set_ylabel('温度')
            axes[0, 1].set_title('温度变化')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 接受率变化
            axes[0, 2].plot(iterations, evaluation['acceptance_rate_history'], 'm-')
            axes[0, 2].set_xlabel('温度循环')
            axes[0, 2].set_ylabel('接受率')
            axes[0, 2].set_title('接受率变化')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. 车辆数变化曲线
            axes[1, 0].plot(iterations, evaluation['vehicle_count_history'], 'orange')
            axes[1, 0].axhline(y=self.n_vehicles, color='r', linestyle='--', label=f'车辆限制 ({self.n_vehicles})')
            axes[1, 0].set_xlabel('温度循环')
            axes[1, 0].set_ylabel('车辆数')
            axes[1, 0].set_title('车辆数变化')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. 成本改进率
            if len(evaluation['best_cost_history']) > 1:
                improvement_rate = []
                for i in range(1, len(evaluation['best_cost_history'])):
                    if evaluation['best_cost_history'][i-1] > 0:
                        rate = (evaluation['best_cost_history'][i-1] - evaluation['best_cost_history'][i]) / evaluation['best_cost_history'][i-1] * 100
                        improvement_rate.append(rate)
                    else:
                        improvement_rate.append(0)
                
                axes[1, 1].plot(iterations[1:], improvement_rate, 'b-')
                axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                axes[1, 1].set_xlabel('温度循环')
                axes[1, 1].set_ylabel('改进率 (%)')
                axes[1, 1].set_title('成本改进率')
                axes[1, 1].grid(True, alpha=0.3)
            
            # 6. 温度与成本关系
            if len(evaluation['temperature_history']) > 0:
                scatter = axes[1, 2].scatter(evaluation['temperature_history'], evaluation['best_cost_history'], 
                                           c=iterations, cmap='viridis', alpha=0.6, s=20)
                axes[1, 2].set_xlabel('温度')
                axes[1, 2].set_ylabel('最佳成本')
                axes[1, 2].set_title('温度-成本关系')
                axes[1, 2].set_xscale('log')
                axes[1, 2].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[1, 2], label='迭代次数')
            
            plt.suptitle(f'模拟退火优化过程 (最终成本: {evaluation["total_cost"]:.2f}, 车辆数: {evaluation["vehicles_used"]}/{self.n_vehicles})', fontsize=12)
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            convergence_plot = RESULTS_DIR / f"sa_convergence_plot_{timestamp}.png"
            plt.savefig(convergence_plot, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger = logging.getLogger(__name__)
            logger.info(f"收敛曲线图已保存: {convergence_plot}")
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"绘制收敛曲线失败: {e}")
    
    def save_results(self, routes: List[List[int]], evaluation: Dict[str, Any], save_path: Path = None) -> Path:
        """保存结果到文件"""
        logger = logging.getLogger(__name__)
        logger.info("\n保存结果到文件...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if save_path is None:
            save_path = RESULTS_DIR / f"c101_simulated_annealing_{timestamp}"
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存解决方案
        routes_data = []
        for vehicle_idx, route in enumerate(routes):
            if route:
                route_str = '0->' + '->'.join(map(str, route)) + '->0'
                routes_data.append({
                    'vehicle_id': vehicle_idx + 1,
                    'route': route_str,
                    'num_customers': len(route),
                    'total_demand': sum(self.demands[c] for c in route)
                })
        
        routes_df = pd.DataFrame(routes_data)
        routes_file = save_path / f"sa_solution_routes_{timestamp}.csv"
        routes_df.to_csv(routes_file, index=False, encoding='utf-8-sig')
        logger.info(f"解决方案已保存: {routes_file}")
        
        # 2. 保存评估结果
        eval_data = {
            'algorithm': evaluation['algorithm'],
            'total_cost': evaluation['total_cost'],
            'total_distance': evaluation['total_distance'],
            'total_violation': evaluation['total_violation'],
            'vehicle_penalty': evaluation.get('vehicle_penalty', 0),
            'coverage_rate': evaluation['coverage_rate'],
            'vehicles_used': evaluation['vehicles_used'],
            'unserved_customers': evaluation['unserved_customers'],
            'runtime': evaluation.get('runtime', 0),
            'iterations': evaluation.get('iterations', 0)
        }
        
        eval_df = pd.DataFrame([eval_data])
        eval_file = save_path / f"sa_evaluation_{timestamp}.csv"
        eval_df.to_csv(eval_file, index=False, encoding='utf-8-sig')
        logger.info(f"评估结果已保存: {eval_file}")
        
        # 3. 保存详细评估
        details_df = pd.DataFrame(evaluation['details'])
        details_file = save_path / f"sa_details_{timestamp}.csv"
        details_df.to_csv(details_file, index=False, encoding='utf-8-sig')
        logger.info(f"详细评估已保存: {details_file}")
        
        # 4. 保存收敛数据
        if 'best_cost_history' in evaluation:
            convergence_data = {
                'iteration': list(range(len(evaluation['best_cost_history']))),
                'temperature': evaluation.get('temperature_history', []),
                'best_cost': evaluation['best_cost_history'],
                'current_cost': evaluation.get('current_cost_history', []),
                'acceptance_rate': evaluation.get('acceptance_rate_history', []),
                'vehicle_count': evaluation.get('vehicle_count_history', [])
            }
            
            convergence_df = pd.DataFrame(convergence_data)
            convergence_file = save_path / f"sa_convergence_data_{timestamp}.csv"
            convergence_df.to_csv(convergence_file, index=False, encoding='utf-8-sig')
            logger.info(f"收敛数据已保存: {convergence_file}")
        
        # 5. 生成可视化
        self.plot_solution(routes, save_path, timestamp, evaluation)
        
        # 6. 生成报告
        self.generate_report(evaluation, save_path, timestamp)
        
        return save_path
    
    def plot_solution(self, routes: List[List[int]], save_path: Path, timestamp: str, evaluation: Dict[str, Any]):
        """绘制解决方案图"""
        try:
            plt.figure(figsize=(12, 10))
            
            # 绘制仓库
            plt.scatter(self.depot_x, self.depot_y, c='red', s=200, marker='*', 
                       label='仓库', zorder=5, edgecolors='black')
            
            # 绘制客户点
            customer_coords = self.coords[1:]
            plt.scatter(customer_coords[:, 0], customer_coords[:, 1], 
                       c='gray', s=50, alpha=0.6, label='客户点')
            
            # 定义颜色
            colors = plt.cm.Set3(np.linspace(0, 1, len(routes)))
            
            # 绘制路径
            for vehicle_idx, route in enumerate(routes):
                if not route:
                    continue
                
                # 获取路径坐标
                path_coords = [self.coords[0]]
                for cust_idx in route:
                    path_coords.append(self.coords[cust_idx])
                path_coords.append(self.coords[0])
                
                path_coords = np.array(path_coords)
                
                # 绘制路径
                plt.plot(path_coords[:, 0], path_coords[:, 1], 
                        marker='o', markersize=4, linewidth=2,
                        color=colors[vehicle_idx], 
                        label=f'车辆{vehicle_idx+1} ({len(route)}个客户)')
            
            # 标记未服务客户
            all_customers = set(range(1, self.n_customers + 1))
            served_customers = set()
            for route in routes:
                served_customers.update(route)
            unserved_customers = all_customers - served_customers
            
            if unserved_customers and evaluation['unserved_customers'] > 0:
                unserved_coords = [self.coords[c] for c in unserved_customers]
                unserved_coords = np.array(unserved_coords)
                plt.scatter(unserved_coords[:, 0], unserved_coords[:, 1], 
                           c='red', s=100, marker='x', label=f'未服务客户 ({len(unserved_customers)}个)', zorder=6)
            
            plt.title(f'C101解决方案 - 模拟退火算法\n总距离: {evaluation["total_distance"]:.2f}, 总成本: {evaluation["total_cost"]:.2f}, 车辆: {evaluation["vehicles_used"]}/{self.n_vehicles}')
            plt.xlabel('X坐标')
            plt.ylabel('Y坐标')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # 保存图片
            plot_file = save_path / f"sa_solution_plot_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger = logging.getLogger(__name__)
            logger.info(f"解决方案图已保存: {plot_file}")
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"绘制解决方案图失败: {e}")
    
    def generate_report(self, evaluation: Dict[str, Any], save_path: Path, timestamp: str):
        """生成实验报告"""
        report_content = f"""# C101 数据集模拟退火算法实验报告

## 实验信息
- 实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 算法: 模拟退火算法
- 数据集: C101标准数据集

## 问题参数
- 客户点数量: {self.n_customers}
- 车辆数量: {self.n_vehicles}
- 车辆容量: {self.vehicle_capacity}
- 仓库坐标: ({self.depot_x}, {self.depot_y})
- 仓库时间窗: [{self.depot_ready_time}, {self.depot_due_date}]
- 总需求: {self.demands[1:].sum()}

## 算法参数
- 初始温度: {self.initial_temperature}
- 终止温度: {self.final_temperature}
- 降温系数: {self.cooling_rate}
- 每个温度迭代次数: {self.iterations_per_temperature}
- 最大迭代次数: {self.max_iterations}
- 时间窗违反惩罚权重: {self.time_penalty_weight}
- 容量违反惩罚权重: {self.capacity_penalty_weight}
- 车辆数惩罚权重: {self.vehicle_penalty_weight}

## 算法性能
- 总成本: {evaluation['total_cost']:.2f}
- 总距离: {evaluation['total_distance']:.2f}
- 总违反惩罚: {evaluation['total_violation']:.2f}
- 车辆数惩罚: {evaluation.get('vehicle_penalty', 0):.2f}
- 客户覆盖率: {evaluation['coverage_rate']:.1f}%
- 使用车辆数: {evaluation['vehicles_used']}/{self.n_vehicles}
- 未服务客户: {evaluation['unserved_customers']}
- 总运行时间: {evaluation.get('runtime', 0):.3f}秒
- 总迭代次数: {evaluation.get('iterations', 0)}

## 各车辆性能详情
| 车辆 | 服务客户数 | 行驶距离 | 载重量 | 时间窗违反 | 容量违反 |
|------|------------|----------|--------|------------|----------|
"""
        
        for detail in evaluation['details']:
            report_content += f"| {detail['vehicle']} | {detail['customers_served']} | {detail['distance']:.2f} | {detail['load']} | {detail['time_window_violations']} | {detail['capacity_violations']} |\n"
        
        report_content += f"""
## 算法总结
模拟退火算法在C101数据集上表现如下:
1. **优化效果**: 最终成本为 {evaluation['total_cost']:.2f}
2. **覆盖情况**: 服务了{evaluation['coverage_rate']:.1f}%的客户
3. **距离表现**: 总行驶距离为{evaluation['total_distance']:.2f}
4. **约束满足**: 总违反惩罚为{evaluation['total_violation']:.2f}
5. **资源利用**: 使用了{evaluation['vehicles_used']}辆车，利用率{100*evaluation['vehicles_used']/self.n_vehicles:.1f}%
6. **计算效率**: 运行{evaluation.get('iterations', 0)}次迭代，用时{evaluation.get('runtime', 0):.2f}秒

## 模拟退火算法特点
1. **局部搜索能力强**: 通过温度控制接受劣解的概率，有助于跳出局部最优
2. **参数敏感**: 初始温度、降温系数对结果影响较大
3. **收敛性好**: 随着温度降低，逐渐收敛到优质解
4. **灵活性高**: 多种邻域操作（2-opt、重定位、交换等）

## 输出文件
- 解决方案: sa_solution_routes_{timestamp}.csv
- 评估结果: sa_evaluation_{timestamp}.csv
- 详细评估: sa_details_{timestamp}.csv
- 收敛数据: sa_convergence_data_{timestamp}.csv
- 可视化图: sa_solution_plot_{timestamp}.png
- 收敛曲线: sa_convergence_plot_{timestamp}.png
"""
        
        report_file = save_path / f"sa_experiment_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger = logging.getLogger(__name__)
        logger.info(f"实验报告已保存: {report_file}")
    
    def run_simulated_annealing_experiment(self) -> Tuple[List[List[int]], Dict[str, Any]]:
        """运行模拟退火算法实验"""
        logger = logging.getLogger(__name__)
        logger.info("\n" + "="*60)
        logger.info("开始模拟退火算法实验")
        logger.info("="*60)
        
        # 运行模拟退火算法
        routes, evaluation = self.simulated_annealing()
        
        # 保存结果
        results_dir = self.save_results(routes, evaluation)
        
        logger.info("\n" + "="*60)
        logger.info("模拟退火算法实验完成!")
        logger.info("="*60)
        logger.info(f"结果保存在: {results_dir}")
        
        return routes, evaluation

def main():
    """主函数"""
    print("C101 数据集VRP求解 - 模拟退火算法")
    print("="*60)
    
    try:
        # 创建模拟退火求解器实例
        sa_solver = C101SimulatedAnnealingVRP()
        
        # 运行模拟退火算法实验
        routes, evaluation = sa_solver.run_simulated_annealing_experiment()
        
        print(f"\n实验完成!")
        print(f"总成本: {evaluation['total_cost']:.2f}")
        print(f"总距离: {evaluation['total_distance']:.2f}")
        print(f"使用车辆: {evaluation['vehicles_used']}/{sa_solver.n_vehicles}")
        print(f"运行时间: {evaluation.get('runtime', 0):.2f}秒")
        print(f"迭代次数: {evaluation.get('iterations', 0)}")
        print(f"\n详细结果见日志文件和输出目录。")
        
    except Exception as e:
        print(f"实验执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()