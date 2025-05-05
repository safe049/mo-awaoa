import numpy as np
from moawaoa.algorithm import MOAWOA
from moawaoa.utils import (
    plot_pareto_front,
    plot_parallel_coordinates,
    plot_radar_chart,
    animate_pareto_evolution,
    animate_abstract_search_process
)

# 定义一个双目标测试函数（ZDT1）
def zdt1(x):
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    f1 = x[0]
    f2 = g * (1 - np.sqrt(f1 / g))
    return [f1, f2]

# 设置参数
dim = 5
bounds = (0, 1)
pop_size = 30
max_iter = 50

# 初始化多目标优化器
moawoa = MOAWOA(
    obj_func=zdt1,
    bounds=bounds,
    dim=dim,
    num_objs=2,
    pop_size=pop_size,
    max_iter=max_iter,
    verbose=True
)

# 存储历史数据
pareto_history = []
population_history = []

# 执行优化并记录历史
for iter in range(max_iter):
    moawoa.optimize()
    pareto_front = moawoa.archive.members.copy()
    population = moawoa.population.solutions.copy()
    population_history.append(population)

# 最终解集
final_solutions = moawoa.archive.members

# 绘制最终帕累托前沿
plot_pareto_front(final_solutions, title="Final Pareto Front", filename="pareto_final.png")

# 绘制平行坐标图
plot_parallel_coordinates(final_solutions, title="Parallel Coordinates of Final Solutions", filename="parallel_coords.png")

# 绘制雷达图
plot_radar_chart(final_solutions, title="Radar Chart of Final Solutions", filename="radar_chart.png")


# 动画：抽象搜索过程
animate_abstract_search_process(
    population_history=population_history,
    pareto_history=pareto_history,
    bounds=bounds,
    title="Algorithm Search Process",
    filename="search_process.gif"
)

print("✅ 所有绘图已完成，并保存为文件。")