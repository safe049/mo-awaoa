import numpy as np
from moawaoa.algorithm import MOAWOA
from moawaoa.utils import plot_pareto_front

def zdt1(x):
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    f1 = x[0]
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.array([f1, f2])

# 设置参数
bounds = (0, 1)
dim = 30
num_objs = 2
pop_size = 50
max_iter = 100

# 初始化算法
optimizer = MOAWOA(zdt1, bounds, dim, num_objs, pop_size, max_iter, verbose=True)

# 执行优化
pareto_front = optimizer.optimize()

# 输出结果
print(f"Final Pareto Front Size: {len(pareto_front)}")

# 绘图
plot_pareto_front(pareto_front) #filename="pareto_front.png")