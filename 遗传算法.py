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
            customers_path = CLEANED_DIR / "c102_customers.csv"
            self.df_customers = pd.read_csv(customers_path)
            
            info_path = CLEANED_DIR / "c102_info.csv"
            self.df_info = pd.read_csv(info_path)
            
            distance_path = DISTANCE_DIR / "c102_距离矩阵.csv"
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
class Chromosome:
    """染色体类，表示一个解决方案"""
    
    def __init__(self, routes: List[List[int]], fitness: float = None):
        self.routes = routes  # 直接存储路径，而不是基因序列
        self.fitness = fitness
        self.total_distance = 0
        self.total_violation = 0
        self.total_cost = 0
        self.crowding_distance = 0  # 用于多样性保持
        
    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __repr__(self):
        return f"Chromosome(fitness={self.fitness:.2f}, vehicles={len(self.routes)}, cost={self.total_cost:.2f})"

class C101GeneticAlgorithmVRP:
    """基于C101标准数据集的VRP遗传算法 - 深度优化版"""
    
    def __init__(self, data_dir="data"):
        """初始化"""
        self.data_dir = Path(data_dir)
        self.setup_logging()
        
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("C101 数据集VRP求解 - 遗传算法（深度优化版）")
        logger.info("=" * 60)
        
        self.load_data()
        self.init_parameters()
        
    def setup_logging(self):
        """设置日志系统"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"c101_genetic_algorithm_optimized_{timestamp}.log"
        
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
            customers_path = CLEANED_DIR / "c101_customers.csv"
            self.df_customers = pd.read_csv(customers_path)
            
            info_path = CLEANED_DIR / "c101_info.csv"
            self.df_info = pd.read_csv(info_path)
            
            distance_path = DISTANCE_DIR / "c101_距离矩阵.csv"
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
            
            # 预计算客户间的时间矩阵
            self.time_matrix = self.distance_matrix / 1.0
            
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
        
        # 遗传算法参数
        self.population_size = 100
        self.generations = 500
        self.crossover_rate = 0.9
        self.mutation_rate = 0.3
        self.elitism_count = 5
        
        # 车辆数惩罚权重 - 动态调整
        self.base_vehicle_penalty = 5000
        self.vehicle_penalty_weight = self.base_vehicle_penalty
        
        # 局部搜索参数
        self.local_search_prob = 0.5
        self.max_local_search_iter = 10
        
        # 自适应参数
        self.adaptive_crossover = True
        self.adaptive_mutation = True
        
        # 提前停止参数
        self.early_stop_patience = 50
        self.early_stop_threshold = 1e-6
        
        # 多样性保持参数
        self.niche_radius = 10.0
        self.use_niching = True
        
        logger = logging.getLogger(__name__)
        logger.info("\n遗传算法参数:")
        logger.info(f"  种群大小: {self.population_size}")
        logger.info(f"  迭代代数: {self.generations}")
        logger.info(f"  交叉率: {self.crossover_rate}")
        logger.info(f"  变异率: {self.mutation_rate}")
        logger.info(f"  精英保留数: {self.elitism_count}")
        logger.info(f"  基础车辆惩罚: {self.base_vehicle_penalty}")
        logger.info(f"  局部搜索概率: {self.local_search_prob}")
    
    def generate_greedy_solution(self) -> List[List[int]]:
        """生成贪心解 - 改进的节约算法"""
        unvisited = set(range(1, self.n_customers + 1))
        routes = []
        
        # 计算节约值
        savings = []
        for i in range(1, self.n_customers + 1):
            for j in range(i + 1, self.n_customers + 1):
                saving = self.distance_matrix[0, i] + self.distance_matrix[0, j] - self.distance_matrix[i, j]
                savings.append((-saving, i, j))  # 负号用于升序排序
        
        savings.sort()
        
        # 初始化每个客户为一条路径
        route_map = {}
        for cust in unvisited:
            route_map[cust] = [cust]
        
        # 合并路径
        for saving, i, j in savings:
            if i not in route_map or j not in route_map:
                continue
            
            route_i = route_map[i]
            route_j = route_map[j]
            
            if route_i is route_j:
                continue
            
            # 检查容量约束
            total_demand = sum(self.demands[c] for c in route_i) + sum(self.demands[c] for c in route_j)
            if total_demand > self.vehicle_capacity:
                continue
            
            # 合并路径
            new_route = route_i + route_j
            
            # 检查时间窗约束
            if self.check_route_feasibility(new_route):
                # 更新路由映射
                for cust in new_route:
                    route_map[cust] = new_route
        
        # 收集所有唯一路径
        unique_routes = []
        seen = set()
        for route in route_map.values():
            route_id = id(route)
            if route_id not in seen:
                seen.add(route_id)
                unique_routes.append(route)
        
        # 对路径进行优化
        for route in unique_routes:
            self.optimize_route_order(route)
        
        # 处理剩余客户（如果有）
        all_served = set()
        for route in unique_routes:
            all_served.update(route)
        
        remaining = unvisited - all_served
        if remaining:
            for cust in remaining:
                if len(unique_routes) < self.n_vehicles:
                    unique_routes.append([cust])
        
        # 限制车辆数不超过最大值
        if len(unique_routes) > self.n_vehicles:
            unique_routes = unique_routes[:self.n_vehicles]
        
        return unique_routes
    
    def check_route_feasibility(self, route: List[int]) -> bool:
        """检查路径是否满足时间窗和容量约束"""
        current_time = self.depot_ready_time
        current_load = 0
        current_node = 0
        
        for cust in route:
            # 容量检查
            current_load += self.demands[cust]
            if current_load > self.vehicle_capacity:
                return False
            
            # 时间检查
            travel_time = self.time_matrix[current_node, cust]
            arrival_time = current_time + travel_time
            
            if arrival_time > self.due_times[cust]:
                return False
            
            # 更新时间
            if arrival_time < self.ready_times[cust]:
                arrival_time = self.ready_times[cust]
            
            current_time = arrival_time + self.service_times[cust]
            current_node = cust
        
        # 检查返回仓库时间
        return_time = current_time + self.time_matrix[current_node, 0]
        if return_time > self.depot_due_date:
            return False
        
        return True
    
    def optimize_route_order(self, route: List[int]) -> None:
        """优化路径内客户的顺序（2-opt）"""
        improved = True
        while improved:
            improved = False
            best_distance = self.calculate_route_distance(route)
            
            for i in range(len(route)):
                for j in range(i + 1, len(route)):
                    # 反转i到j的部分
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    
                    if self.check_route_feasibility(new_route):
                        new_distance = self.calculate_route_distance(new_route)
                        if new_distance < best_distance:
                            route[:] = new_route
                            best_distance = new_distance
                            improved = True
                            break
                if improved:
                    break
    
    def calculate_route_distance(self, route: List[int]) -> float:
        """计算路径的总距离"""
        if not route:
            return 0
        
        distance = 0
        current_node = 0
        
        for cust in route:
            distance += self.distance_matrix[current_node, cust]
            current_node = cust
        
        distance += self.distance_matrix[current_node, 0]
        return distance
    
    def generate_random_solution(self) -> List[List[int]]:
        """生成随机解 - 改进版，考虑时间窗"""
        customers = list(range(1, self.n_customers + 1))
        random.shuffle(customers)
        
        routes = []
        current_route = []
        current_load = 0
        current_time = self.depot_ready_time
        current_node = 0
        
        for cust in customers:
            demand = self.demands[cust]
            
            # 检查容量和时间窗
            if (current_load + demand <= self.vehicle_capacity and 
                len(routes) < self.n_vehicles):
                
                travel_time = self.time_matrix[current_node, cust]
                arrival_time = current_time + travel_time
                
                if arrival_time <= self.due_times[cust]:
                    current_route.append(cust)
                    current_load += demand
                    
                    if arrival_time < self.ready_times[cust]:
                        arrival_time = self.ready_times[cust]
                    
                    current_time = arrival_time + self.service_times[cust]
                    current_node = cust
                    continue
            
            # 不能加入当前路径，创建新路径
            if current_route:
                routes.append(current_route)
            
            current_route = [cust]
            current_load = demand
            
            arrival_time = self.depot_ready_time + self.time_matrix[0, cust]
            if arrival_time < self.ready_times[cust]:
                arrival_time = self.ready_times[cust]
            
            current_time = arrival_time + self.service_times[cust]
            current_node = cust
        
        if current_route:
            routes.append(current_route)
        
        # 限制车辆数
        if len(routes) > self.n_vehicles:
            routes = routes[:self.n_vehicles]
        
        return routes
    
    def generate_initial_population(self) -> List[Chromosome]:
        """生成初始种群 - 混合贪心和随机解"""
        logger = logging.getLogger(__name__)
        logger.info("生成初始种群...")
        
        population = []
        
        # 生成10%的贪心解
        n_greedy = max(1, int(self.population_size * 0.1))
        for _ in range(n_greedy):
            routes = self.generate_greedy_solution()
            chromosome = Chromosome(routes)
            self.evaluate_chromosome(chromosome)
            population.append(chromosome)
        
        # 生成剩余的随机解
        for _ in range(self.population_size - n_greedy):
            routes = self.generate_random_solution()
            chromosome = Chromosome(routes)
            self.evaluate_chromosome(chromosome)
            population.append(chromosome)
        
        # 按适应度排序
        population.sort()
        
        logger.info(f"初始种群生成完成，最佳适应度: {population[0].fitness:.6f}, 最佳成本: {population[0].total_cost:.2f}")
        return population
    
    def evaluate_chromosome(self, chromosome: Chromosome) -> Chromosome:
        """评估染色体 - 改进版，更精确的惩罚计算"""
        total_distance = 0
        total_violation = 0
        
        for route in chromosome.routes:
            if not route:
                continue
            
            # 计算路径距离
            distance = self.calculate_route_distance(route)
            total_distance += distance
            
            # 计算违反惩罚
            violation = self.calculate_route_violation(route)
            total_violation += violation
        
        # 车辆数惩罚 - 动态调整
        vehicle_excess = max(0, len(chromosome.routes) - self.n_vehicles)
        vehicle_penalty = vehicle_excess * self.vehicle_penalty_weight
        
        # 空路径惩罚
        empty_routes = sum(1 for route in chromosome.routes if not route)
        empty_penalty = empty_routes * 1000
        
        chromosome.total_distance = total_distance
        chromosome.total_violation = total_violation
        chromosome.total_cost = total_distance + total_violation + vehicle_penalty + empty_penalty
        
        # 适应度是成本的倒数（成本越低，适应度越高）
        chromosome.fitness = 1.0 / (chromosome.total_cost + 1.0)
        
        return chromosome
    
    def calculate_route_violation(self, route: List[int]) -> float:
        """计算单条路径的违反惩罚 - 改进版"""
        violation = 0
        current_time = self.depot_ready_time
        current_node = 0
        current_load = 0
        
        for cust in route:
            # 距离和时间
            travel_time = self.time_matrix[current_node, cust]
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
        return_time = current_time + self.time_matrix[current_node, 0]
        if return_time > self.depot_due_date:
            time_violation = return_time - self.depot_due_date
            violation += time_violation * self.time_penalty_weight
        
        return violation
    
    def tournament_selection(self, population: List[Chromosome], tournament_size: int = 3) -> Chromosome:
        """锦标赛选择 - 比轮盘赌更稳定"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def selection(self, population: List[Chromosome]) -> Chromosome:
        """选择操作 - 使用锦标赛选择"""
        return self.tournament_selection(population)
    
    def order_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """顺序交叉(OX) - 更适合VRP的交叉算子"""
        # 将父代转换为基因序列（客户编号列表）
        gene1 = []
        for route in parent1.routes:
            gene1.extend(route)
        
        gene2 = []
        for route in parent2.routes:
            gene2.extend(route)
        
        # 确保基因长度相同
        if len(gene1) != self.n_customers or len(gene2) != self.n_customers:
            # 如果有缺失客户，使用父代1
            return parent1
        
        # 随机选择交叉点
        start = random.randint(0, self.n_customers - 2)
        end = random.randint(start + 1, self.n_customers - 1)
        
        # 创建子代基因
        child_gene = [None] * self.n_customers
        
        # 复制父代1的中间部分
        child_gene[start:end+1] = gene1[start:end+1]
        
        # 从父代2复制剩余部分
        ptr = 0
        for cust in gene2:
            if cust not in child_gene:
                while child_gene[ptr] is not None:
                    ptr += 1
                child_gene[ptr] = cust
        
        # 将基因序列转换为路径
        child_routes = self.gene_to_routes(child_gene)
        
        return Chromosome(child_routes)
    
    def gene_to_routes(self, gene: List[int]) -> List[List[int]]:
        """将基因序列转换为路径列表"""
        routes = []
        current_route = []
        current_load = 0
        current_time = self.depot_ready_time
        current_node = 0
        
        for cust in gene:
            demand = self.demands[cust]
            
            # 检查容量和时间窗
            if (current_load + demand <= self.vehicle_capacity and 
                len(routes) < self.n_vehicles):
                
                travel_time = self.time_matrix[current_node, cust]
                arrival_time = current_time + travel_time
                
                if arrival_time <= self.due_times[cust]:
                    current_route.append(cust)
                    current_load += demand
                    
                    if arrival_time < self.ready_times[cust]:
                        arrival_time = self.ready_times[cust]
                    
                    current_time = arrival_time + self.service_times[cust]
                    current_node = cust
                    continue
            
            # 不能加入当前路径，创建新路径
            if current_route:
                routes.append(current_route)
            
            current_route = [cust]
            current_load = demand
            
            arrival_time = self.depot_ready_time + self.time_matrix[0, cust]
            if arrival_time < self.ready_times[cust]:
                arrival_time = self.ready_times[cust]
            
            current_time = arrival_time + self.service_times[cust]
            current_node = cust
        
        if current_route:
            routes.append(current_route)
        
        # 限制车辆数
        if len(routes) > self.n_vehicles:
            routes = routes[:self.n_vehicles]
        
        return routes
    
    def path_based_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """基于路径的交叉 - 保留好的路径结构"""
        if random.random() > self.crossover_rate:
            return parent1
        
        routes1 = parent1.routes.copy()
        routes2 = parent2.routes.copy()
        
        # 随机选择一些路径从父代1
        child_routes = []
        used_customers = set()
        
        # 从父代1选择最好的几条路径
        routes1_with_cost = []
        for route in routes1:
            cost = self.calculate_route_distance(route) + self.calculate_route_violation(route)
            routes1_with_cost.append((cost, route))
        
        routes1_with_cost.sort()
        
        # 选择前30%的路径
        n_select = max(1, int(len(routes1_with_cost) * 0.3))
        for cost, route in routes1_with_cost[:n_select]:
            if len(child_routes) < self.n_vehicles:
                child_routes.append(route.copy())
                used_customers.update(route)
        
        # 从父代2取剩余的客户
        for route in routes2:
            new_route = []
            for cust in route:
                if cust not in used_customers:
                    new_route.append(cust)
                    used_customers.add(cust)
            
            if new_route:
                # 尝试合并到现有路径
                merged = False
                for i, existing_route in enumerate(child_routes):
                    route_demand = sum(self.demands[c] for c in existing_route)
                    new_route_demand = sum(self.demands[c] for c in new_route)
                    
                    if route_demand + new_route_demand <= self.vehicle_capacity:
                        # 尝试合并并检查可行性
                        merged_route = existing_route + new_route
                        if self.check_route_feasibility(merged_route):
                            child_routes[i] = merged_route
                            merged = True
                            break
                
                if not merged and len(child_routes) < self.n_vehicles:
                    child_routes.append(new_route)
        
        # 处理遗漏的客户
        all_customers = set(range(1, self.n_customers + 1))
        missing_customers = all_customers - used_customers
        
        for cust in missing_customers:
            # 尝试插入到现有路径的最佳位置
            best_route_idx = -1
            best_position = -1
            best_cost = float('inf')
            
            for route_idx, route in enumerate(child_routes):
                for pos in range(len(route) + 1):
                    temp_route = route.copy()
                    temp_route.insert(pos, cust)
                    
                    route_demand = sum(self.demands[c] for c in temp_route)
                    if route_demand > self.vehicle_capacity:
                        continue
                    
                    if self.check_route_feasibility(temp_route):
                        cost = self.calculate_route_distance(temp_route)
                        if cost < best_cost:
                            best_cost = cost
                            best_route_idx = route_idx
                            best_position = pos
            
            if best_route_idx != -1:
                child_routes[best_route_idx].insert(best_position, cust)
            elif len(child_routes) < self.n_vehicles:
                child_routes.append([cust])
        
        # 移除空路径
        child_routes = [route for route in child_routes if route]
        
        return Chromosome(child_routes)
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """交叉操作 - 混合使用多种交叉算子"""
        # 70%概率使用基于路径的交叉，30%使用顺序交叉
        if random.random() < 0.7:
            return self.path_based_crossover(parent1, parent2)
        else:
            return self.order_crossover(parent1, parent2)
    
    def swap_mutation(self, chromosome: Chromosome) -> Chromosome:
        """交换变异 - 同路径内交换两个客户"""
        routes = [route.copy() for route in chromosome.routes]
        
        # 选择一个有至少2个客户的路径
        valid_routes = [i for i, r in enumerate(routes) if len(r) >= 2]
        if not valid_routes:
            return chromosome
        
        route_idx = random.choice(valid_routes)
        route = routes[route_idx]
        
        # 随机选择两个位置交换
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
        
        # 检查可行性
        if self.check_route_feasibility(route):
            return Chromosome(routes)
        else:
            # 恢复原状
            route[i], route[j] = route[j], route[i]
            return chromosome
    
    def relocate_mutation(self, chromosome: Chromosome) -> Chromosome:
        """重定位变异 - 将一个客户移到另一个位置"""
        routes = [route.copy() for route in chromosome.routes]
        
        # 选择一个有客户的路径
        valid_routes = [i for i, r in enumerate(routes) if r]
        if not valid_routes:
            return chromosome
        
        from_route_idx = random.choice(valid_routes)
        from_route = routes[from_route_idx]
        
        # 选择要移动的客户
        cust_idx = random.randint(0, len(from_route) - 1)
        customer = from_route.pop(cust_idx)
        
        # 如果源路径为空，移除它
        if not from_route:
            routes.pop(from_route_idx)
        
        # 尝试插入到最佳位置
        best_route_idx = -1
        best_position = -1
        best_cost = float('inf')
        
        for route_idx, route in enumerate(routes):
            for pos in range(len(route) + 1):
                temp_route = route.copy()
                temp_route.insert(pos, customer)
                
                route_demand = sum(self.demands[c] for c in temp_route)
                if route_demand > self.vehicle_capacity:
                    continue
                
                if self.check_route_feasibility(temp_route):
                    cost = self.calculate_route_distance(temp_route)
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = route_idx
                        best_position = pos
        
        # 如果找到好的位置，插入
        if best_route_idx != -1:
            routes[best_route_idx].insert(best_position, customer)
        elif len(routes) < self.n_vehicles:
            # 创建新路径
            routes.append([customer])
        else:
            # 无法插入，恢复原状
            if from_route_idx < len(routes):
                routes[from_route_idx].insert(cust_idx, customer)
            else:
                routes.insert(from_route_idx, [customer])
        
        return Chromosome(routes)
    
    def inversion_mutation(self, chromosome: Chromosome) -> Chromosome:
        """反转变异 - 反转路径中的一段"""
        routes = [route.copy() for route in chromosome.routes]
        
        # 选择一个有至少2个客户的路径
        valid_routes = [i for i, r in enumerate(routes) if len(r) >= 2]
        if not valid_routes:
            return chromosome
        
        route_idx = random.choice(valid_routes)
        route = routes[route_idx]
        
        # 随机选择一段反转
        start = random.randint(0, len(route) - 2)
        end = random.randint(start + 1, len(route) - 1)
        
        new_route = route[:start] + route[start:end+1][::-1] + route[end+1:]
        
        # 检查可行性
        if self.check_route_feasibility(new_route):
            routes[route_idx] = new_route
        
        return Chromosome(routes)
    
    def merge_routes_mutation(self, chromosome: Chromosome) -> Chromosome:
        """合并路径变异 - 尝试合并两条路径以减少车辆数"""
        routes = [route.copy() for route in chromosome.routes]
        
        if len(routes) < 2:
            return chromosome
        
        # 随机选择两条路径
        route1_idx, route2_idx = random.sample(range(len(routes)), 2)
        route1 = routes[route1_idx]
        route2 = routes[route2_idx]
        
        # 检查容量
        total_demand = sum(self.demands[c] for c in route1) + sum(self.demands[c] for c in route2)
        if total_demand > self.vehicle_capacity:
            return chromosome
        
        # 尝试合并
        merged_route = route1 + route2
        
        # 优化合并后的路径
        self.optimize_route_order(merged_route)
        
        if self.check_route_feasibility(merged_route):
            # 移除两条旧路径，添加新路径
            new_routes = []
            for i, route in enumerate(routes):
                if i != route1_idx and i != route2_idx:
                    new_routes.append(route)
            new_routes.append(merged_route)
            
            return Chromosome(new_routes)
        
        return chromosome
    
    def mutation(self, chromosome: Chromosome) -> Chromosome:
        """变异操作 - 混合使用多种变异算子"""
        if random.random() > self.mutation_rate:
            return chromosome
        
        # 随机选择变异类型
        mutation_types = ['swap', 'relocate', 'inversion', 'merge']
        weights = [0.2, 0.3, 0.2, 0.3]  # 合并变异权重更高，以减少车辆数
        
        mutation_type = random.choices(mutation_types, weights=weights)[0]
        
        if mutation_type == 'swap':
            return self.swap_mutation(chromosome)
        elif mutation_type == 'relocate':
            return self.relocate_mutation(chromosome)
        elif mutation_type == 'inversion':
            return self.inversion_mutation(chromosome)
        elif mutation_type == 'merge':
            return self.merge_routes_mutation(chromosome)
        
        return chromosome
    
    def local_search_2opt(self, chromosome: Chromosome) -> Chromosome:
        """2-opt局部搜索"""
        routes = [route.copy() for route in chromosome.routes]
        improved = True
        
        while improved:
            improved = False
            
            for route_idx, route in enumerate(routes):
                if len(route) < 2:
                    continue
                
                best_route = route.copy()
                best_distance = self.calculate_route_distance(route)
                
                for i in range(len(route)):
                    for j in range(i + 1, len(route)):
                        # 反转i到j的部分
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        
                        if self.check_route_feasibility(new_route):
                            new_distance = self.calculate_route_distance(new_route)
                            if new_distance < best_distance - 1e-6:
                                best_route = new_route
                                best_distance = new_distance
                                improved = True
                
                if improved:
                    routes[route_idx] = best_route
                    break
        
        return Chromosome(routes)
    
    def local_search(self, chromosome: Chromosome) -> Chromosome:
        """局部搜索主函数"""
        if random.random() > self.local_search_prob:
            return chromosome
        
        # 应用2-opt局部搜索
        improved = self.local_search_2opt(chromosome)
        
        # 重新评估
        self.evaluate_chromosome(improved)
        
        return improved
    
    def calculate_crowding_distance(self, population: List[Chromosome]) -> None:
        """计算拥挤度 - 用于多样性保持"""
        if len(population) <= 2:
            for ch in population:
                ch.crowding_distance = float('inf')
            return
        
        # 初始化拥挤度
        for ch in population:
            ch.crowding_distance = 0
        
        # 按成本排序
        sorted_by_cost = sorted(population, key=lambda x: x.total_cost)
        
        # 边界点的拥挤度设为无穷大
        sorted_by_cost[0].crowding_distance = float('inf')
        sorted_by_cost[-1].crowding_distance = float('inf')
        
        # 计算中间点的拥挤度
        cost_range = sorted_by_cost[-1].total_cost - sorted_by_cost[0].total_cost
        if cost_range > 0:
            for i in range(1, len(sorted_by_cost) - 1):
                sorted_by_cost[i].crowding_distance += (
                    sorted_by_cost[i+1].total_cost - sorted_by_cost[i-1].total_cost
                ) / cost_range
        
        # 按车辆数排序
        sorted_by_vehicles = sorted(population, key=lambda x: len(x.routes))
        
        # 边界点的拥挤度设为无穷大
        sorted_by_vehicles[0].crowding_distance = float('inf')
        sorted_by_vehicles[-1].crowding_distance = float('inf')
        
        # 计算中间点的拥挤度
        vehicle_range = len(sorted_by_vehicles[-1].routes) - len(sorted_by_vehicles[0].routes)
        if vehicle_range > 0:
            for i in range(1, len(sorted_by_vehicles) - 1):
                sorted_by_vehicles[i].crowding_distance += (
                    len(sorted_by_vehicles[i+1].routes) - len(sorted_by_vehicles[i-1].routes)
                ) / vehicle_range
    
    def niche_selection(self, population: List[Chromosome]) -> List[Chromosome]:
        """小生境选择 - 保持种群多样性"""
        if not self.use_niching:
            return population
        
        # 计算拥挤度
        self.calculate_crowding_distance(population)
        
        # 按适应度和拥挤度排序
        population.sort(key=lambda x: (-x.fitness, -x.crowding_distance))
        
        # 选择前population_size个个体
        return population[:self.population_size]
    
    def update_adaptive_parameters(self, generation: int, best_cost_history: List[float]) -> None:
        """更新自适应参数"""
        if not self.adaptive_crossover and not self.adaptive_mutation:
            return
        
        # 计算改进率
        if len(best_cost_history) > 10:
            recent_improvement = (
                best_cost_history[-10] - best_cost_history[-1]
            ) / best_cost_history[-10] if best_cost_history[-10] > 0 else 0
            
            # 如果改进率低，增加变异率，降低交叉率
            if recent_improvement < 0.001:
                self.mutation_rate = min(0.5, self.mutation_rate + 0.01)
                self.crossover_rate = max(0.7, self.crossover_rate - 0.01)
            else:
                # 如果改进率高，降低变异率，增加交叉率
                self.mutation_rate = max(0.1, self.mutation_rate - 0.01)
                self.crossover_rate = min(0.95, self.crossover_rate + 0.01)
        
        # 动态调整车辆惩罚权重
        if generation % 50 == 0 and generation > 0:
            # 检查当前最佳解的车辆数
            current_best_vehicles = len(self.best_chromosome.routes)
            
            if current_best_vehicles > self.n_vehicles:
                # 车辆数超标，增加惩罚
                self.vehicle_penalty_weight = min(20000, self.vehicle_penalty_weight * 1.5)
            elif current_best_vehicles < self.n_vehicles:
                # 车辆数低于限制，降低惩罚
                self.vehicle_penalty_weight = max(1000, self.vehicle_penalty_weight * 0.8)
    
    def genetic_algorithm(self) -> Tuple[List[List[int]], Dict[str, Any]]:
        """遗传算法主函数 - 优化版"""
        logger = logging.getLogger(__name__)
        logger.info("\n开始遗传算法优化...")
        start_time = time.time()
        
        # 生成初始种群
        population = self.generate_initial_population()
        
        # 记录进化过程
        best_cost_history = []
        avg_cost_history = []
        vehicle_count_history = []
        
        # 初始化最佳解
        self.best_chromosome = population[0]
        best_generation = 0
        
        # 遗传算法主循环
        for generation in range(self.generations):
            # 创建新一代种群
            new_population = []
            
            # 精英保留
            new_population.extend(population[:self.elitism_count])
            
            # 生成剩余个体
            while len(new_population) < self.population_size:
                # 选择
                parent1 = self.selection(population)
                parent2 = self.selection(population)
                
                # 交叉
                child = self.crossover(parent1, parent2)
                
                # 变异
                child = self.mutation(child)
                
                # 局部搜索
                child = self.local_search(child)
                
                # 评估
                self.evaluate_chromosome(child)
                
                new_population.append(child)
            
            # 更新种群
            population = new_population
            
            # 小生境选择 - 保持多样性
            population = self.niche_selection(population)
            
            # 按适应度排序
            population.sort()
            
            # 更新最佳解
            if population[0].total_cost < self.best_chromosome.total_cost - 1e-6:
                self.best_chromosome = population[0]
                best_generation = generation
                logger.info(f"代 {generation:3d}: 发现新最佳解! 成本: {self.best_chromosome.total_cost:.2f}, 车辆数: {len(self.best_chromosome.routes)}")
            
            # 记录统计信息
            avg_cost = np.mean([ch.total_cost for ch in population])
            
            best_cost_history.append(self.best_chromosome.total_cost)
            avg_cost_history.append(avg_cost)
            vehicle_count_history.append(len(self.best_chromosome.routes))
            
            # 输出进度
            if generation % 20 == 0 or generation == self.generations - 1:
                logger.info(f"代 {generation:3d}: "
                            f"最佳成本 {self.best_chromosome.total_cost:.2f}, "
                            f"平均成本 {avg_cost:.2f}, "
                            f"车辆数 {len(self.best_chromosome.routes)}, "
                            f"最佳代数 {best_generation}")
            
            # 更新自适应参数
            self.update_adaptive_parameters(generation, best_cost_history)
            
            # 提前停止检查
            if generation - best_generation > self.early_stop_patience:
                logger.info(f"提前停止: 连续{self.early_stop_patience}代没有改进")
                break
        
        runtime = time.time() - start_time
        
        # 最终评估
        evaluation = self.evaluate_solution(self.best_chromosome.routes, "遗传算法（优化版）")
        evaluation.update({
            'runtime': runtime,
            'generations': generation + 1,
            'best_generation': best_generation,
            'best_cost_history': best_cost_history,
            'avg_cost_history': avg_cost_history,
            'vehicle_count_history': vehicle_count_history
        })
        
        logger.info("\n遗传算法完成:")
        logger.info(f"  总运行时间: {runtime:.3f} 秒")
        logger.info(f"  最终成本: {evaluation['total_cost']:.2f}")
        logger.info(f"  总距离: {evaluation['total_distance']:.2f}")
        logger.info(f"  总违反惩罚: {evaluation['total_violation']:.2f}")
        logger.info(f"  使用车辆数: {evaluation['vehicles_used']}/{self.n_vehicles}")
        logger.info(f"  最佳解发现于第 {best_generation} 代")
        
        # 绘制收敛曲线
        self.plot_convergence_curve(evaluation)
        
        return self.best_chromosome.routes, evaluation
    
    def evaluate_solution(self, routes: List[List[int]], algorithm_name: str) -> Dict[str, Any]:
        """评估解决方案的质量"""
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
        
        # 总成本
        total_cost = total_distance + total_violation
        
        # 计算覆盖率
        served_count = len(served_customers)
        coverage_rate = served_count / self.n_customers * 100
        
        result = {
            'algorithm': algorithm_name,
            'total_cost': total_cost,
            'total_distance': total_distance,
            'total_violation': total_violation,
            'coverage_rate': coverage_rate,
            'vehicles_used': len([r for r in routes if r]),
            'unserved_customers': self.n_customers - served_count,
            'details': details
        }
        
        return result
    
    def plot_convergence_curve(self, evaluation: Dict[str, Any]):
        """绘制收敛曲线"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            generations = list(range(len(evaluation['best_cost_history'])))
            
            # 1. 成本收敛曲线
            axes[0, 0].plot(generations, evaluation['best_cost_history'], 'b-', linewidth=2, label='最佳成本')
            axes[0, 0].plot(generations, evaluation['avg_cost_history'], 'r-', alpha=0.7, label='平均成本')
            axes[0, 0].set_xlabel('迭代代数')
            axes[0, 0].set_ylabel('成本')
            axes[0, 0].set_title('成本收敛曲线')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 车辆数变化曲线
            axes[0, 1].plot(generations, evaluation['vehicle_count_history'], 'g-')
            axes[0, 1].axhline(y=self.n_vehicles, color='r', linestyle='--', label=f'车辆限制 ({self.n_vehicles})')
            axes[0, 1].set_xlabel('迭代代数')
            axes[0, 1].set_ylabel('车辆数')
            axes[0, 1].set_title('车辆数变化')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 成本改进率
            if len(evaluation['best_cost_history']) > 1:
                improvement_rate = []
                for i in range(1, len(evaluation['best_cost_history'])):
                    if evaluation['best_cost_history'][i-1] > 0:
                        rate = (evaluation['best_cost_history'][i-1] - evaluation['best_cost_history'][i]) / evaluation['best_cost_history'][i-1] * 100
                        improvement_rate.append(rate)
                    else:
                        improvement_rate.append(0)
                
                axes[1, 0].plot(generations[1:], improvement_rate, 'b-')
                axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                axes[1, 0].set_xlabel('迭代代数')
                axes[1, 0].set_ylabel('改进率 (%)')
                axes[1, 0].set_title('成本改进率')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 成本分布
            final_generation = min(50, len(evaluation['best_cost_history']))
            axes[1, 1].hist(evaluation['best_cost_history'][-final_generation:], bins=20, alpha=0.7)
            axes[1, 1].set_xlabel('成本')
            axes[1, 1].set_ylabel('频率')
            axes[1, 1].set_title('最后50代成本分布')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'遗传算法优化过程 (最终成本: {evaluation["total_cost"]:.2f}, 车辆数: {evaluation["vehicles_used"]}/{self.n_vehicles})', fontsize=12)
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            convergence_plot = RESULTS_DIR / f"convergence_plot_optimized_{timestamp}.png"
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
            save_path = RESULTS_DIR / f"c101_genetic_algorithm_optimized_{timestamp}"
        
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
        routes_file = save_path / f"solution_routes_{timestamp}.csv"
        routes_df.to_csv(routes_file, index=False, encoding='utf-8-sig')
        logger.info(f"解决方案已保存: {routes_file}")
        
        # 2. 保存评估结果
        eval_data = {
            'algorithm': evaluation['algorithm'],
            'total_cost': evaluation['total_cost'],
            'total_distance': evaluation['total_distance'],
            'total_violation': evaluation['total_violation'],
            'coverage_rate': evaluation['coverage_rate'],
            'vehicles_used': evaluation['vehicles_used'],
            'unserved_customers': evaluation['unserved_customers'],
            'runtime': evaluation.get('runtime', 0),
            'best_generation': evaluation.get('best_generation', 0)
        }
        
        eval_df = pd.DataFrame([eval_data])
        eval_file = save_path / f"evaluation_{timestamp}.csv"
        eval_df.to_csv(eval_file, index=False, encoding='utf-8-sig')
        logger.info(f"评估结果已保存: {eval_file}")
        
        # 3. 保存详细评估
        details_df = pd.DataFrame(evaluation['details'])
        details_file = save_path / f"details_{timestamp}.csv"
        details_df.to_csv(details_file, index=False, encoding='utf-8-sig')
        logger.info(f"详细评估已保存: {details_file}")
        
        # 4. 保存收敛数据
        if 'best_cost_history' in evaluation:
            convergence_data = {
                'generation': list(range(len(evaluation['best_cost_history']))),
                'best_cost': evaluation['best_cost_history'],
                'avg_cost': evaluation.get('avg_cost_history', []),
                'vehicle_count': evaluation.get('vehicle_count_history', [])
            }
            
            convergence_df = pd.DataFrame(convergence_data)
            convergence_file = save_path / f"convergence_data_{timestamp}.csv"
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
            
            plt.title(f'C101解决方案 - 遗传算法（优化版）\n总距离: {evaluation["total_distance"]:.2f}, 总成本: {evaluation["total_cost"]:.2f}, 车辆: {evaluation["vehicles_used"]}/{self.n_vehicles}')
            plt.xlabel('X坐标')
            plt.ylabel('Y坐标')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # 保存图片
            plot_file = save_path / f"solution_plot_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger = logging.getLogger(__name__)
            logger.info(f"解决方案图已保存: {plot_file}")
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"绘制解决方案图失败: {e}")
    
    def generate_report(self, evaluation: Dict[str, Any], save_path: Path, timestamp: str):
        """生成实验报告"""
        report_content = f"""# C101 数据集遗传算法实验报告（优化版）

## 实验信息
- 实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 算法: 遗传算法（深度优化版）
- 数据集: C101标准数据集

## 问题参数
- 客户点数量: {self.n_customers}
- 车辆数量: {self.n_vehicles}
- 车辆容量: {self.vehicle_capacity}
- 仓库坐标: ({self.depot_x}, {self.depot_y})
- 仓库时间窗: [{self.depot_ready_time}, {self.depot_due_date}]
- 总需求: {self.demands[1:].sum()}

## 算法参数
- 种群大小: {self.population_size}
- 迭代代数: {self.generations}
- 交叉率: {self.crossover_rate}
- 变异率: {self.mutation_rate}
- 精英保留数: {self.elitism_count}
- 时间窗违反惩罚权重: {self.time_penalty_weight}
- 容量违反惩罚权重: {self.capacity_penalty_weight}
- 基础车辆惩罚: {self.base_vehicle_penalty}
- 局部搜索概率: {self.local_search_prob}
- 提前停止耐心值: {self.early_stop_patience}

## 算法性能
- 总成本: {evaluation['total_cost']:.2f}
- 总距离: {evaluation['total_distance']:.2f}
- 总违反惩罚: {evaluation['total_violation']:.2f}
- 客户覆盖率: {evaluation['coverage_rate']:.1f}%
- 使用车辆数: {evaluation['vehicles_used']}/{self.n_vehicles}
- 未服务客户: {evaluation['unserved_customers']}
- 总运行时间: {evaluation.get('runtime', 0):.3f}秒
- 最佳解发现于第 {evaluation.get('best_generation', 0)} 代

## 各车辆性能详情
| 车辆 | 服务客户数 | 行驶距离 | 载重量 | 时间窗违反 | 容量违反 |
|------|------------|----------|--------|------------|----------|
"""
        
        for detail in evaluation['details']:
            report_content += f"| {detail['vehicle']} | {detail['customers_served']} | {detail['distance']:.2f} | {detail['load']} | {detail['time_window_violations']} | {detail['capacity_violations']} |\n"
        
        report_content += f"""
## 算法改进点
1. **初始解质量**: 混合了贪心节约算法和随机算法，提高了初始种群质量
2. **交叉算子**: 实现了基于路径的交叉和顺序交叉，更好地保留了好的路径结构
3. **变异算子**: 增加了路径合并变异，有效减少车辆数
4. **局部搜索**: 加入了2-opt局部优化，大幅提升了解的质量
5. **选择策略**: 使用锦标赛选择代替轮盘赌，更加稳定
6. **多样性保持**: 实现了拥挤度计算和小生境技术，避免早熟收敛
7. **自适应参数**: 交叉率和变异率会根据进化情况自动调整
8. **提前停止**: 当连续多代没有改进时自动停止，节省计算时间

## 输出文件
- 解决方案: solution_routes_{timestamp}.csv
- 评估结果: evaluation_{timestamp}.csv
- 详细评估: details_{timestamp}.csv
- 收敛数据: convergence_data_{timestamp}.csv
- 可视化图: solution_plot_{timestamp}.png
- 收敛曲线: convergence_plot_optimized_{timestamp}.png
"""
        
        report_file = save_path / f"experiment_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger = logging.getLogger(__name__)
        logger.info(f"实验报告已保存: {report_file}")
    
    def run_genetic_algorithm_experiment(self) -> Tuple[List[List[int]], Dict[str, Any]]:
        """运行遗传算法实验"""
        logger = logging.getLogger(__name__)
        logger.info("\n" + "="*60)
        logger.info("开始遗传算法实验（优化版）")
        logger.info("="*60)
        
        # 运行遗传算法
        routes, evaluation = self.genetic_algorithm()
        
        # 保存结果
        results_dir = self.save_results(routes, evaluation)
        
        logger.info("\n" + "="*60)
        logger.info("遗传算法实验完成!")
        logger.info("="*60)
        logger.info(f"结果保存在: {results_dir}")
        
        return routes, evaluation

def main():
    """主函数"""
    print("C101 数据集VRP求解 - 遗传算法（深度优化版）")
    print("="*60)
    
    try:
        # 创建遗传算法求解器实例
        ga_solver = C101GeneticAlgorithmVRP()
        
        # 运行遗传算法实验
        routes, evaluation = ga_solver.run_genetic_algorithm_experiment()
        
        print(f"\n实验完成!")
        print(f"总成本: {evaluation['total_cost']:.2f}")
        print(f"总距离: {evaluation['total_distance']:.2f}")
        print(f"使用车辆: {evaluation['vehicles_used']}/{ga_solver.n_vehicles}")
        print(f"运行时间: {evaluation.get('runtime', 0):.2f}秒")
        print(f"最佳解发现于第 {evaluation.get('best_generation', 0)} 代")
        print(f"\n详细结果见日志文件和输出目录。")
        
    except Exception as e:
        print(f"实验执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()