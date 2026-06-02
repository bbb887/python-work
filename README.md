[readme(1).md](https://github.com/user-attachments/files/28507509/readme.1.md)
# 无人机集群配送路径规划 (Multi-UAV Delivery Routing)

**基于多种智能优化算法的带时间窗车辆路径问题（VRPTW）求解器**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Data](https://img.shields.io/badge/data-Solomon_C101-orange)](http://web.cba.neu.edu/~msolomon/problems.htm)

本项目以 **Solomon C101 标准数据集** 为测试实例，模拟多架无人机（无人机）从中央仓库出发，在 **载重、航程、时间窗** 三重约束下为所有客户提供配送服务，最终返回仓库。  
优化目标为最小化 **综合成本 = 飞行总距离 + 时间窗超时惩罚 + 硬约束违规惩罚**。

---

##  项目亮点

- **丰富的算法库**
  实现了贪心算法（最近邻、节约算法、最便宜插入）、遗传算法、混合遗传算法（融合 Solomon I1 插入启发式）以及模拟退火 + 神经网络辅助四种主流求解器。
- **完全可复现**  
  所有改进版算法均输出统一的日志文件、指标 CSV、文本摘要和路径可视化图，方便对比不同算法的性能。
- **数据全流程处理**  
  提供一键式脚本，将原始 Solomon 格式的 `.txt` 文件转换为结构化 CSV，并自动生成欧氏距离矩阵。
- **探索性数据分析（EDA）**  
  自带 EDA 脚本，可生成空间分布热力图、客户聚类、特征相关性分析等 6 张专业图表和一份详细分析报告。
- **硬约束严格满足**  
  所有算法均内嵌容错机制和修复算子，优先保证解可行性，并通过惩罚函数量化不可行度。

---

##  项目结构

```
drone_delivery_eda/
├── data_clean.py                # 数据预处理：Solomon TXT → CSV
├── 距离矩阵.py                   # 批量生成欧氏距离矩阵 CSV
├── c101_eda.py                  # 探索性数据分析与可视化报告
├── requirements.txt             # Python 依赖清单
│
├── c101_customers.csv           # 客户数据（由预处理生成，或直接提供）
├── c101_info.csv                # 问题参数（车辆数、容量等）
├── c101_距离矩阵.csv             # 距离矩阵（可选，算法可自动计算）
│
├── 贪心算法改(1).py              # 贪心算法（改进版，统一输出）
├── 遗传算法改(1).py              # 遗传算法（改进版，统一输出）
├── 混合遗传算法改(1).py          # 混合遗传算法（改进版，统一输出）
├── 退火算法改.py                # 模拟退火+神经网络（改进版，统一输出）
│
├── 贪心算法.py                  # 贪心算法（原始版，多策略对比）
├── 遗传算法.py                  # 遗传算法（原始版，硬编码数据集）
├── Mixed GA.py                  # 混合遗传算法（另一版本）
├── greedy_vrp.py               # 贪心算法（另一版本）
├── 模拟退火.py                  # 模拟退火（原始版）
│
├── results/                     # 输出结果（自动创建）
├── log/                         # 运行日志（自动创建）
└── README.md
```

---

##  环境要求与安装

**基础环境：Python 3.8+**

### 核心依赖
所有求解器均依赖以下库，可通过一条命令安装：
```bash
pip install pandas numpy matplotlib scikit-learn
```

### 可选依赖（仅运行 EDA 时需要）
```bash
pip install seaborn scipy plotly jupyter ipython
```

或直接根据 `requirements.txt` 安装全部：
```bash
pip install -r requirements.txt
```

推荐使用虚拟环境：
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

---

##  快速上手（5分钟跑通）

### 步骤1：准备 C101 数据集
1. 从 [Solomon VRPTW 官网](http://web.cba.neu.edu/~msolomon/problems.htm) 下载 `C101.txt` 文件。
2. 将文件放入 `data/raw/Solomon(1)/` 文件夹（可在 `data_clean.py` 中修改路径）。
3. 运行数据预处理：
   ```bash
   python data_clean.py
   ```
   执行后在 `data/cleaned/` 下生成：
   - `c101_customers.csv` — 100 个客户的坐标、需求、时间窗、服务时间
   - `c101_info.csv`    — 仓库坐标、车辆数、容量等全局参数

4. （可选）生成距离矩阵，加快后续读入速度：
   ```bash
   python 距离矩阵.py
   ```
   得到的 `c101_距离矩阵.csv` 将保存仓库+客户两两之间的欧氏距离。

### 步骤2：运行求解算法
将上述生成的三个 CSV 文件（客户、信息、距离矩阵）拷贝到各算法脚本所在目录 **或** 修改脚本头部的文件路径变量。  
然后依次执行：
```bash
python 贪心算法改(1).py      # 贪心算法（三种策略对比）
python 遗传算法改(1).py      # 遗传算法
python 混合遗传算法改(1).py  # 混合遗传算法
python 退火算法改.py        # 模拟退火+神经网络
```

### 步骤3：查看结果
每个算法运行完毕后，在 `results/` 下会生成以下文件：
- `{算法名}_metrics.csv` — 关键指标（总成本、飞行距离、超时惩罚、违规等情况）
- `{算法名}_summary.txt` — 文本摘要，包含各无人机任务详情及约束违反表
- `{算法名}_routes.png` — 路径可视化图（JPG 高清输出）
- `{算法名}_routes.csv` — 每条路线的客户序列和具体载重/航程

详细运行日志保存在 `log/` 文件夹中。

---

##  算法详解

### 1. 贪心算法 (Greedy)
**核心思想**：依赖局部最优选择逐步构造解，之后用 2-opt 交换进行局部优化。

**内置三种构造策略**：
- **最近邻 (Nearest Neighbor)**  
  从仓库出发，每次选择距离当前节点最近且满足容量/时间窗的客户，直到路线无法扩展。
- **节约算法 (Clarke-Wright Savings)**  
  计算合并两条路线的“节约值”，从大到小合并，产生更紧凑的路线簇。
- **最便宜插入 (Cheapest Insertion)**  
  从最近仓库客户开始，逐步将剩余客户插入到现有路线中，插入成本最小的位置。

**后处理**：对每条路线执行 2-opt 优化，消除自交叉路径。  
**特点**：速度极快（<1秒），适合生成决策参考或用作元启发式算法的初始种群。

---

### 2. 遗传算法 (Genetic Algorithm, GA)
**核心思想**：模拟自然选择与遗传进化，通过交叉和变异产生多样化解，逐代收敛至高质量解。

**关键组件**：
- **编码**：客户序号排列（Giant Tour）。
- **解码**：顺序扫描排列，遇到载重超限则开启新路线（简单容量分割）。
- **适应度函数**：同综合成本（距离 + 超时惩罚 + 硬约束惩罚）。
- **选择**：锦标赛选择（Tournament Selection）。
- **交叉**：有序交叉（OX）保留父代的相对顺序。
- **变异**：交换变异或相邻子串反转。
- **精英保留**：直接传递最优个体至下一代。
- **局部搜索**：每 10 代对精英个体执行 2-opt 局部优化。

**参数可控**：种群大小、进化代数、交叉率、变异率、精英比例。  
**优点**：全局搜索能力强，不易陷入局部最优。

---

### 3. 混合遗传算法 (Mixed GA)
**核心改进**：前代遗传算法在时间窗约束严格时，解码后的解往往包含大量超时惩罚，导致搜索效率低。  
混合遗传算法在两个方面进行强化：

1. **Solomon I1 插入启发式构建初始种群**  
   在构造初始解时即考虑时间窗可行性，优先将紧迫客户提前服务，大幅降低初始解的违规惩罚。
2. **2-opt 使用新目标函数**  
   局部搜索时直接评估“距离 + 超时 + 硬约束”的综合成本，确保每次交换都使综合成本下降。

**适用场景**：时间窗非常严格、要求解不可行度极低的问题。

---

### 4. 模拟退火 + 神经网络 (Simulated Annealing + NN)
**双层加速**：  
- **模拟退火**：通过温度控制，允许以一定概率接受劣解，从而跳出局部最优。  
- **神经网络辅助**：训练一个多层感知机（MLP），预测将某个客户插入到某路线的具体位置所带来的距离增量，快速筛选邻域操作，避免从头评估完整路线。

**邻域结构**：2-opt、客户重定位（relocate）、客户交换（exchange）、路线分割与合并。  
**严格约束**：强制无人机的使用数量不超过预设上限（默认 25 架），并在违反时施加高额惩罚。  
**优点**：收敛速度快，适合超参数调优和实时性要求较高的场景。

---

##  参数配置

所有改进版算法均从 `c101_info.csv` 读取以下参数，可通过修改该文件或脚本内全局变量调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VEHICLE_CAPACITY` | 200 | 每架无人机的最大载重 |
| `VEHICLE_NUMBER` | 25 | 最大可用无人机数量 |
| `DEPOT_READY_TIME` | 0 | 仓库开门时间 |
| `DEPOT_DUE_DATE` | 1236 | 仓库关门时间 |
| `TIME_PENALTY_WEIGHT` | 20 | 超时惩罚系数（单位/时间） |
| `HARD_PENALTY_WEIGHT` | 1000 | 载重/航程超限惩罚系数 |

**算法超参数**（可在各脚本顶部修改）：
- **GA**：`pop_size`, `generations`, `cx_prob`, `mut_prob`, `elite_size`
- **SA**：`initial_temperature`, `cooling_rate`, `max_iterations`
- **NN**：`hidden_layer_sizes`, 训练样本数

---

##  输出指标说明

以 `{算法}_metrics.csv` 为例：

| 指标 | 含义 |
|----- |-----|
| 总成本 | 距离 + 超时惩罚 + 硬约束惩罚（越小越好）|
| 飞行距离成本 | 所有无人机飞行总距离（即纯路径长度） |
| 原始距离(无惩罚) | 不含任何惩罚的物理飞行距离 |
| 时间窗超时惩罚 | 所有客户和仓库的超时累计 × 惩罚系数 |
| 硬约束违规惩罚 | 载重超限 + 航程超限的累计惩罚 |
| Makespan | 所有无人机完成任务的最后时刻 |
| 超时总时长 | 实际超时的总时间（未乘系数）|
| 使用的无人机数 | 实际出动无人机数量 |
| 算法运行时间(秒) | 求解耗时 |

约束违反表（`summary.txt` 内）会逐条列出每架无人机的 **载重、航程、客户超时、仓库超时** 以及总体可行性判断。

---

##  可视化示例

运行 `c101_eda.py` 将生成以下分析图表，帮助理解数据特征：
- **多变量联合分布图** – 发现需求、时间窗的分布模式
- **空间-需求热力图** – 识别需求密集区域
- **客户空间聚类（K-Means）** – 为分区规划提供参考
- **时间窗口分布直方图** – 观察服务时间紧迫性
- **特征相关性热力图** – 探索需求、坐标与时间的关联

求解器生成的路线图示例：  
（此处可插入 `results/routes.png` 的截图链接）

---

##  贡献指南

欢迎提交 Issue 或 Pull Request 来改进代码、增加新算法或修复 bug。  
建议的开发流程：
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingAlgorithm`)
3. 提交更改 (`git commit -m 'Add some AmazingAlgorithm'`)
4. 推送到分支 (`git push origin feature/AmazingAlgorithm`)
5. 开启 Pull Request

请确保新增的算法脚本与现有输出格式保持一致，并附带简要的使用说明。

---

##  参考文献

- Solomon, M. M. (1987). Algorithms for the vehicle routing and scheduling problems with time window constraints. *Operations Research*, 35(2), 254‑265.
- VRPTW Benchmark: [http://web.cba.neu.edu/~msolomon/problems.htm](http://web.cba.neu.edu/~msolomon/problems.htm)
- Oliver, I. M., Smith, D. J., & Holland, J. R. C. (1987). A study of permutation crossover operators on the traveling salesman problem. *ICGA*.
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671‑680.

---

##  许可

本项目代码遵循 MIT 许可证，自由使用、修改和分发。  
数据集（Solomon 实例）版权归属原作者，仅供学术研究使用。

---
