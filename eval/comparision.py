import numpy as np
import time
import statistics
from typing import List, Tuple, Callable, Optional, Union
import matplotlib.pyplot as plt
from woa_optimizer import WOAOptimizer
from moawaoa.algorithm import MOAWOA    # 新增 MO-AWAOA

import numpy as np
import time
import statistics
from typing import List, Tuple, Callable, Optional, Union
import matplotlib.pyplot as plt

from qgbwoa import QGBWOA



# 测试函数
def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) + (1 - x[:-1]) ** 2))

def ackley(x: np.ndarray) -> float:
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    return float(-a * np.exp(-b * np.sqrt((1/n) * np.sum(x ** 2))) - 
                np.exp((1/n) * np.sum(np.cos(c * x))) + a + np.e)

def rastrigin(x: np.ndarray) -> float:
    return float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10))

def griewank(x: np.ndarray) -> float:
    product = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
    return float(1 + np.sum(x ** 2) / 4000 - product)

def schwefel(x: np.ndarray) -> float:
    return float(-np.sum(x * np.sin(np.sqrt(np.abs(x)))))

def levy(x: np.ndarray) -> float:
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return float(term1 + term2 + term3)

def zakharov(x: np.ndarray) -> float:
    idx = np.arange(1, len(x)+1)
    return float(np.sum(x ** 2) + np.sum(0.5 * idx * x) ** 2 + np.sum(0.5 * idx * x) ** 4)

def michalewicz(x: np.ndarray, m: int = 10) -> float:
    return float(-np.sum(np.sin(x) * np.sin((np.arange(1, len(x)+1) * x ** 2) / np.pi) ** (2 * m)))

def schwefel_12(x: np.ndarray) -> float:
    sum_ = 0
    for i in range(len(x)):
        sum_ += abs(x[:i+1]).sum()
    return sum_

# 设置随机种子以保证实验可复现性
def set_seed(seed=42):
    np.random.seed(seed)

# 对齐历史长度
def align_histories(histories):
    max_len = max(len(h) for h in histories)
    aligned = []
    for h in histories:
        if len(h) < max_len:
            last_val = h[-1]
            h += [last_val] * (max_len - len(h))
        aligned.append(h)
    return aligned


if __name__ == "__main__":
    DIMENSIONS = 10
    MAX_ITER = 100
    BOUNDS = [(-5, 5)] * DIMENSIONS
    N_RUNS = 5

    def michalewicz_wrapper(x):
        return michalewicz(x)

    test_functions = [
        ('Sphere', sphere),
        ('Rosenbrock', rosenbrock),
        ('Ackley', ackley),
        ('Rastrigin', rastrigin),
        ('Griewank', griewank),
        ('Schwefel', schwefel),
        ('Levy', levy),
        ('Zakharov', zakharov),
        ('Michalewicz', michalewicz_wrapper),
        ('Schwefel_1.2', schwefel_12)
    ]

    results_table = []
    all_woa_histories = []
    all_moawaoa_histories = []

    for func_name, func in test_functions:
        print(f"\n=== Testing on {func_name} Function ===")
        woa_fitnesses = []
        moawaoa_fitnesses = []
        qgbwoa_fitnesses = []

        woa_histories = []
        moawaoa_histories = []
        qgbwoa_histories = []

        for run in range(N_RUNS):
            seed = run * 100
            set_seed(seed)
            print(f"\nRun {run+1}/{N_RUNS} (Seed={seed})")

            # SEDO


            # WOA
            print("\nRunning WOA...")
            woa_opt = WOAOptimizer(objective_func=func, problem_dim=DIMENSIONS, n_particles=30, bounds=BOUNDS)
            woa_opt.optimize(MAX_ITER)
            woa_fitnesses.append(woa_opt.best_fitness)
            woa_histories.append(woa_opt.get_convergence_history())

            print("\nRunning QGBWOA...")
            qgbwoa_opt = QGBWOA(
                obj_func=func,
                bounds=(-5, 5),
                dim=DIMENSIONS,
                pop_size=30,
                max_iter=MAX_ITER,
                verbose=False
            )
            best_sol, best_fit, history = qgbwoa_opt.optimize()
            qgbwoa_fitnesses.append(best_fit)
            qgbwoa_histories.append(history)            

            # MO-AWAOA
            print("\nRunning MO-AWAOA...")
            moawaoa_opt = MOAWOA(obj_func=func, bounds=(-5, 5), dim=DIMENSIONS, num_objs=1,
                                 pop_size=30, max_iter=MAX_ITER, verbose=False)
            pareto_front = moawaoa_opt.optimize()
            best_sol = min(pareto_front, key=lambda s: sum(s.f))  # 选择综合最优解
            moawaoa_fitnesses.append(sum(best_sol.f))
            moawaoa_histories.append([sum(s.f) for s in pareto_front])  # 模拟历史记录

        # 计算平均、标准差、成功率等
        avg_woa = statistics.mean(woa_fitnesses)
        std_woa = statistics.stdev(woa_fitnesses) if len(woa_fitnesses) > 1 else 0
        avg_moawaoa = statistics.mean(moawaoa_fitnesses)
        std_moawaoa = statistics.stdev(moawaoa_fitnesses) if len(moawaoa_fitnesses) > 1 else 0
        avg_qgbwoa = statistics.mean(qgbwoa_fitnesses)
        std_qgbwoa = statistics.stdev(qgbwoa_fitnesses) if len(qgbwoa_fitnesses) > 1 else 0
        

        success_thresholds = {
            'Sphere': 1e-4,
            'Rosenbrock': 1e-2,
            'Ackley': 1e-2,
            'Rastrigin': 1e-1,
            'Griewank': 1e-3,
            'Zakharov': 1e-3,
            'Michalewicz': 1e-1
        }

        sr_woa = sum(1 for f in woa_fitnesses if f <= success_thresholds.get(func_name, float('inf'))) / N_RUNS
        sr_moawaoa = sum(1 for f in moawaoa_fitnesses if f <= success_thresholds.get(func_name, float('inf'))) / N_RUNS
        sr_qgbwoa = sum(1 for f in qgbwoa_fitnesses if f <= success_thresholds.get(func_name, float('inf'))) / N_RUNS

        results_table.extend([
            {"Function": func_name, "Algorithm": "WOA", "Avg Fitness": avg_woa, "Std Dev": std_woa, "Success Rate": sr_woa},
            {"Function": func_name, "Algorithm": "MO-AWAOA", "Avg Fitness": avg_moawaoa, "Std Dev": std_moawaoa, "Success Rate": sr_moawaoa},
            {"Function": func_name, "Algorithm": "QGBWOA", "Avg Fitness": avg_qgbwoa, "Std Dev": std_qgbwoa, "Success Rate": sr_qgbwoa}
        ])


        all_woa_histories.append(np.mean(align_histories(woa_histories), axis=0).tolist())
        all_moawaoa_histories.append(np.mean(align_histories(moawaoa_histories), axis=0).tolist())

        # 可选：绘制每个函数的收敛曲线
        # plot_convergence(...)

    # 输出结果表格
    print("\nAll Tests Complete!")
    print("=" * 80)
    print(f"{'Function':<15} {'Algorithm':<10} {'Avg Fitness':<15} {'Std Dev':<15} {'SR (%)':<10}")
    print("-" * 80)
    for row in results_table:
        print(f"{row['Function']:<15} {row['Algorithm']:<10} "
              f"{row['Avg Fitness']:<15.6f} {row['Std Dev']:<15.6f} {row['Success Rate']*100:<10.2f}")
    print("=" * 80)

    # 可选：绘制所有函数的平均收敛曲线