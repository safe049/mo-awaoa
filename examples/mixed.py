import numpy as np
from moawaoa.algorithm import MOAWOA
from moawaoa.utils import plot_pareto_front

# ========================
# 1. 公交调度路线优化
# 目标：最小化总等待时间或班次间隔不均
# 决策变量：每趟车的发车时间点
# ========================
def bus_schedule(x):
    x_sorted = np.sort(x)
    intervals = np.diff(x_sorted)
    if np.any(intervals < 5):  # 强制最小间隔
        return float('inf')     # 不合法解惩罚
    imbalance = np.std(intervals)
    return imbalance

print("=== 公交调度路线优化 ===")
moawoa_bus = MOAWOA(
    obj_func=bus_schedule,
    bounds=(0, 60),  # 发车时间在 0~60 分钟之间
    dim=6,          # 6 个班次
    num_objs=1,
    pop_size=30,
    max_iter=100,
    verbose=False
)
best_bus = moawoa_bus.optimize()[0]
print("最佳发车时间点:", np.sort(best_bus.x))
print("不平衡度:", best_bus.f)

# ========================
# 2. 多目标旅行路径规划（TSP）
# 目标：最短路径 + 最少时间
# 假设城市坐标已知，x是城市的访问顺序索引
# ========================
cities_coords = np.random.rand(10, 2) * 100  # 10个城市坐标

def tsp_multi_obj(x):
    order = np.argsort(x)  # 将连续值映射为城市顺序
    if len(np.unique(order)) != len(order):  # 检查是否重复访问城市
        return [float('inf'), float('inf')]
    path = cities_coords[order]
    dist = 0
    time = 0
    for i in range(len(path) - 1):
        d = np.linalg.norm(path[i] - path[i+1])
        dist += d
        time += d / np.random.uniform(30, 80)  # 不同路段速度不同
    return [dist, time]

print("\n=== 多目标旅行路径优化 ===")
moawoa_tsp = MOAWOA(
    obj_func=tsp_multi_obj,
    bounds=(0, 1),
    dim=10,
    num_objs=2,
    pop_size=50,
    max_iter=100,
    verbose=False
)
pareto_solutions = moawoa_tsp.optimize()
print(f"帕累托前沿解数量: {len(pareto_solutions)}")
print(f"解集:")
for sol in pareto_solutions:
    print(np.argsort(sol.x), "  ", sol.f)
for sol in pareto_solutions[:3]:
    print("路径长度:", sol.f[0], "时间:", sol.f[1])

# ========================
# 3. 最优超参数搜索（如用于ML模型）
# 参数：学习率、batch size、层数、神经元数等
# 目标：最大化准确率（负值表示最小化）
# ========================
def hyperparam_search(x):
    lr = x[0] * (0.1 - 0.0001) + 0.0001
    batch_size = int(x[1] * 250) + 10
    layers = int(x[2] * 4) + 1
    units = int(x[3] * 200) + 32
    
    if batch_size < 8 or not (1 <= layers <= 5):
        return float('inf')
    
    accuracy = -(lr * 0.7 + np.sin(batch_size/32) + np.log(units) + layers * 0.1)
    return accuracy

print("\n=== 最优超参数搜索 ===")
moawoa_hp = MOAWOA(
    obj_func=hyperparam_search,
    bounds=(0, 1),
    dim=4,
    num_objs=1,
    pop_size=30,
    max_iter=50,
    verbose=False
)
best_hp = moawoa_hp.optimize()[0]
print("最佳参数向量:", best_hp.x)
print("模拟准确率:", -np.array(best_hp.f).item())

# ========================
# 4. 最优家电调度时间
# 决策变量：各家电启动时间
# 目标：最小化高峰用电时段重叠
# ========================
def appliance_scheduling(x):
    # 约束：洗衣机(索引0)和热水器(索引1)不能同时运行
    if abs(x[0] - x[1]) < 1:
        return float('inf')

    # 约束：洗碗机(索引2)必须在 20:00 - 22:00 运行
    if not (20 <= x[2] <= 22):
        return float('inf')

    peak_start, peak_end = 18, 20
    overlaps = sum(1 for t in x if peak_start <= t <= peak_end)
    return overlaps

print("\n=== 最优家电调度时间 ===")
moawoa_app = MOAWOA(
    obj_func=appliance_scheduling,
    bounds=(0, 24),
    dim=5,  # 5个家电
    num_objs=1,
    pop_size=30,
    max_iter=50,
    verbose=False
)
best_app = moawoa_app.optimize()[0]
print("最佳调度时间:", np.round(best_app.x))
print("高峰重叠次数:", best_app.f)

# ========================
# 5. 最优家电调度时间 + 总路径长度（多目标）
# 多目标：最小化高峰用电 + 家电移动距离总和
# ========================
locations = np.random.rand(5, 2) * 10  # 每个家电的位置

def multi_obj_appliance(x):
    times = x[:5]
    order = np.argsort(x[5:])
    
    # 家电不能同时运行
    if any(abs(times[i] - times[j]) < 1 for i in range(5) for j in range(i+1,5)):
        return [float('inf'), float('inf')]
    
    # 计算高峰用电
    overlaps = sum(1 for t in times if 18 <= t <= 20)

    # 计算路径长度
    total_dist = 0
    for i in range(len(order)-1):
        total_dist += np.linalg.norm(locations[order[i]] - locations[order[i+1]])

    return [overlaps, total_dist]

print("\n=== 多目标家电调度时间+路径优化 ===")
moawoa_app_multi = MOAWOA(
    obj_func=multi_obj_appliance,
    bounds=(0, 24),
    dim=10,  # 5个时间 + 5个路径决策
    num_objs=2,
    pop_size=50,
    max_iter=100,
    verbose=False
)
pareto_app_multi = moawoa_app_multi.optimize()
print(f"帕累托前沿解数量: {len(pareto_app_multi)}")
print(f"解集:")
for sol in pareto_app_multi:
    print(np.round(sol.x[:5]), "  ", np.argsort(sol.x[5:]), "  ", sol.f)
for sol in pareto_app_multi[:3]:
    print("高峰重叠:", sol.f[0], "路径总长:", sol.f[1])

# ========================
# 6. 最优货架布局顺序
# 目标：最小化取货路径长度
# 决策变量：商品排列顺序（通过排序映射）
# ========================
product_coords = np.array([
    [0, 0], [1, 0], [2, 0],
    [0, 1], [1, 1], [2, 1],
    [0, 2], [1, 2], [2, 2]
])  # 9个货架位置

def shelf_layout_order(x):
    order = np.argsort(x)
    
    # 商品 0 和 1 必须相邻
    if abs(np.where(order == 0)[0][0] - np.where(order == 1)[0][0]) != 1:
        return float('inf')
    
    # 商品 2 必须最后一个取
    if order[-1] != 2:
        return float('inf')
    
    path = product_coords[order]
    total_length = 0
    for i in range(len(path) - 1):
        total_length += np.linalg.norm(path[i] - path[i+1])
    return total_length

print("\n=== 最优货架布局顺序 ===")
moawoa_shelf = MOAWOA(
    obj_func=shelf_layout_order,
    bounds=(0, 1),
    dim=9,
    num_objs=1,
    pop_size=50,
    max_iter=100,
    verbose=False
)
best_shelf = moawoa_shelf.optimize()[0]
print("最佳货架访问顺序索引:", np.argsort(best_shelf.x))
print("路径总长度:", best_shelf.f)