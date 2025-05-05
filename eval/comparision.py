import numpy as np
import time
import statistics
from typing import List, Tuple, Callable, Optional, Union
import matplotlib.pyplot as plt
import argparse

# 引入优化器
from woa_optimizer import WOAOptimizer
from moawaoa.algorithm import MOAWOA
from qgbwoa import QGBWOA

import opfunu

# ==================== 测试函数定义 ====================

def wrap_cec_function(cec_func_class, ndim=20):
    func = cec_func_class(ndim=ndim)
    def wrapper(x):
        return float(func.evaluate(x))
    wrapper.__name__ = func.__class__.__name__
    return wrapper

# 获取所有 CEC2017 函数
cec2022_funcs = opfunu.get_functions_based_classname("2022")

# 选择前20个用于测试（F1~F10），这里假设我们选择第一个函数作为示例
cec2022_test_funcs = [
    (func_class.__name__, wrap_cec_function(func_class)) for func_class in cec2022_funcs[:20]
]

# 🔹 单峰函数 Unimodal
def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))

def schwefel_12(x: np.ndarray) -> float:
    sum_ = 0
    for i in range(len(x)):
        sum_ += abs(x[:i+1]).sum()
    return sum_

def step(x: np.ndarray) -> float:
    return float(np.sum(np.floor(x)))

def schwefel_220(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))

def schwefel_221(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))

# 🔸 多模态函数 Multimodal
def rastrigin(x: np.ndarray) -> float:
    return float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10))

def ackley(x: np.ndarray) -> float:
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    return float(-a * np.exp(-b * np.sqrt((1/n) * np.sum(x ** 2))) -
                np.exp((1/n) * np.sum(np.cos(c * x))) + a + np.e)

def griewank(x: np.ndarray) -> float:
    product = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
    return float(1 + np.sum(x ** 2) / 4000 - product)

def levy(x: np.ndarray) -> float:
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return float(term1 + term2 + term3)

def michalewicz(x: np.ndarray, m: int = 10) -> float:
    return float(-np.sum(np.sin(x) * np.sin((np.arange(1, len(x)+1) * x ** 2) / np.pi) ** (2 * m)))

def alpine(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x * np.sin(x) + 0.1 * x)))

# 🔹 非凸/复杂结构函数 Complex Structure
def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) + (1 - x[:-1]) ** 2))

def zakharov(x: np.ndarray) -> float:
    idx = np.arange(1, len(x)+1)
    return float(np.sum(x ** 2) + np.sum(0.5 * idx * x) ** 2 + np.sum(0.5 * idx * x) ** 4)

def happy_cat(x: np.ndarray, alpha: float = 0.125) -> float:
    n = len(x)
    return float(((np.sum(x**2) - n)**2)**alpha + (0.5 * np.sum(x**2) + np.sum(x)) / n + 0.5)

# 🔸 不可分离函数 Non-separable
def brown(x: np.ndarray) -> float:
    return float(np.sum((x[:-1]**2)**(x[1:]**2 + 1) + (x[1:]**2)**(x[:-1]**2 + 1)))

# 🔹 特殊结构函数 Special Form
def exponential(x: np.ndarray) -> float:
    return float(-np.exp(-0.5 * np.sum(x**2)))

def quartic(x: np.ndarray) -> float:
    return float(np.sum((np.arange(1, len(x)+1) * x**4) + np.random.rand()))

# ==================== 测试函数分类配置 ====================
test_function_groups = {
    "Unimodal": [
        ('Sphere', sphere),
        ('Schwefel_1.2', schwefel_12),
        ('Step', step),
        ('Schwefel_2.20', schwefel_220),
        ('Schwefel_2.21', schwefel_221)
    ],
    "Multimodal": [
        ('Rastrigin', rastrigin),
        ('Ackley', ackley),
        ('Griewank', griewank),
        ('Levy', levy),
        ('Michalewicz', lambda x: michalewicz(x)),
        ('Alpine', alpine)
    ],
    "ComplexStructure": [
        ('Rosenbrock', rosenbrock),
        ('Zakharov', zakharov),
        ('HappyCat', happy_cat)
    ],
    "NonSeparable": [
        ('Brown', brown),
        ('Levy', levy)
    ],
    "SpecialForm": [
        ('Exponential', exponential),
        ('Quartic', quartic)
    ],
    "CEC2022": cec2022_test_funcs  # 新增 CEC2017 分类
}

# ==================== 主程序逻辑 ====================
DIMENSIONS = 20
MAX_ITER = 100
BOUNDS = [(-100, 100)] * DIMENSIONS
N_RUNS = 5

set_seed = lambda seed=42: np.random.seed(seed)

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
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run optimization benchmark tests.")
    parser.add_argument('--categories', nargs='+', type=str, default=None,
                        help='List of function categories to test (e.g., Unimodal Multimodal). Default: all.')

    args = parser.parse_args()

    # 如果未指定类别，则运行所有分类
    selected_categories = args.categories if args.categories is not None else list(test_function_groups.keys())
    results_table = []

    # 按照分类运行测试
    for category in selected_categories:
        if category not in test_function_groups:
            print(f"Warning: Category '{category}' not found in test_function_groups. Skipping.")
            continue

        functions = test_function_groups[category]
        print(f"\n{'='*60}")
        print(f"Running tests for category: {category}")
        print(f"{'='*60}")

        for func_name, func in functions:
            print(f"\n=== Testing on {func_name} Function ===")
            woa_fitnesses = []
            moawaoa_fitnesses = []
            qgbwoa_fitnesses = []

            for run in range(N_RUNS):
                seed = run * 100
                set_seed(seed)
                print(f"\nRun {run+1}/{N_RUNS} (Seed={seed})")

                # WOA
                woa_opt = WOAOptimizer(objective_func=func, problem_dim=DIMENSIONS, n_particles=30, bounds=BOUNDS)
                woa_opt.optimize(MAX_ITER)
                woa_fitnesses.append(woa_opt.best_fitness)

                # QGBWOA
                qgbwoa_opt = QGBWOA(obj_func=func, bounds=(-5, 5), dim=DIMENSIONS,
                                    pop_size=30, max_iter=MAX_ITER, verbose=False)
                best_sol, best_fit, _ = qgbwoa_opt.optimize()
                qgbwoa_fitnesses.append(best_fit)

                # MO-AWAOA
                moawaoa_opt = MOAWOA(obj_func=func, bounds=(-5, 5), dim=DIMENSIONS, num_objs=1,
                                     pop_size=30, max_iter=MAX_ITER, verbose=False)
                pareto_front = moawaoa_opt.optimize()
                best_sol = min(pareto_front, key=lambda s: sum(s.f))
                moawaoa_fitnesses.append(sum(best_sol.f))

            # 计算统计指标
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
                {"Category": category, "Function": func_name, "Algorithm": "WOA", "Avg Fitness": avg_woa, "Std Dev": std_woa, "Success Rate": sr_woa},
                {"Category": category, "Function": func_name, "Algorithm": "MO-AWAOA", "Avg Fitness": avg_moawaoa, "Std Dev": std_moawaoa, "Success Rate": sr_moawaoa},
                {"Category": category, "Function": func_name, "Algorithm": "QGBWOA", "Avg Fitness": avg_qgbwoa, "Std Dev": std_qgbwoa, "Success Rate": sr_qgbwoa}
            ])

    # 输出结果表格
    print("\nAll Tests Complete!")
    print("=" * 90)
    print(f"{'Category':<15} {'Function':<15} {'Algorithm':<10} {'Avg Fitness':<15} {'Std Dev':<15} {'SR (%)':<10}")
    print("-" * 90)
    for row in results_table:
        print(f"{row['Category']:<15} {row['Function']:<15} {row['Algorithm']:<10} "
              f"{row['Avg Fitness']:<15.6f} {row['Std Dev']:<15.6f} {row['Success Rate']*100:<10.2f}")
    print("=" * 90)