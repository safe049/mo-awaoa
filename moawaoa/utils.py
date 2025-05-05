import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

def dominates(obj1, obj2):
    better = np.less(obj1, obj2)
    at_least_one_better = np.any(better)
    all_better_or_equal = np.all(np.less_equal(obj1, obj2))
    return at_least_one_better and all_better_or_equal

def crowding_distance_assignment(solutions):
    n = len(solutions)
    m = len(solutions[0].f)

    for sol in solutions:
        sol.crowding_dist = 0

    for obj_idx in range(m):
        sorted_sols = sorted(solutions, key=lambda s: s.f[obj_idx])
        f_min = sorted_sols[0].f[obj_idx]
        f_max = sorted_sols[-1].f[obj_idx]
        if f_max == f_min:
            continue
        sorted_sols[0].crowding_dist = float('inf')
        sorted_sols[-1].crowding_dist = float('inf')
        for i in range(1, n - 1):
            sorted_sols[i].crowding_dist += (sorted_sols[i + 1].f[obj_idx] - sorted_sols[i - 1].f[obj_idx]) / (f_max - f_min)

def hypervolume(ref_point, solutions):
    from deap.benchmarks.tools import hypervolume
    return hypervolume([s.f.tolist() for s in solutions], ref_point)

def igd(ref_set, approx_set):
    d = pairwise_distances(ref_set, [s.f for s in approx_set])
    return np.mean(np.min(d, axis=1))

def plot_pareto_front(solutions, title="Pareto Front", filename=None):
    fig = plt.figure()
    if len(solutions[0].f) == 2:
        plt.scatter([s.f[0] for s in solutions], [s.f[1] for s in solutions])
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
    elif len(solutions[0].f) == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter([s.f[0] for s in solutions], [s.f[1] for s in solutions], [s.f[2] for s in solutions])
    else:
        print("Only support 2D or 3D objective space.")
        return
    plt.title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def save_checkpoint(population, archive, iteration, filename="checkpoint.json"):
    data = {
        "iteration": iteration,
        "population": [{"x": list(sol.x), "f": list(sol.f)} for sol in population.solutions],
        "archive": [{"x": list(sol.x), "f": list(sol.f)} for sol in archive.members]
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_checkpoint(problem, filename="checkpoint.json"):
    with open(filename, 'r') as f:
        data = json.load(f)
    population = Population(problem, len(data["population"]))
    for i, item in enumerate(data["population"]):
        x = np.array(item["x"])
        f = np.array(item["f"])
        population.solutions[i] = Solution(x, f)
    archive = Archive(max_size=len(data["archive"]))
    for item in data["archive"]:
        x = np.array(item["x"])
        f = np.array(item["f"])
        archive.add(Solution(x, f))
    return population, archive, data["iteration"]

def crowding_distance_assignment(solutions):
    """计算拥挤距离"""
    n = len(solutions)
    m = len(solutions[0].f)

    for sol in solutions:
        sol.crowding_dist = 0

    for obj_idx in range(m):
        sorted_sols = sorted(solutions, key=lambda s: s.f[obj_idx])
        f_min = sorted_sols[0].f[obj_idx]
        f_max = sorted_sols[-1].f[obj_idx]
        if f_max == f_min:
            continue
        sorted_sols[0].crowding_dist = float('inf')
        sorted_sols[-1].crowding_dist = float('inf')
        for i in range(1, n - 1):
            sorted_sols[i].crowding_dist += (sorted_sols[i + 1].f[obj_idx] - sorted_sols[i - 1].f[obj_idx]) / (f_max - f_min)

def plot_parallel_coordinates(solutions, title="Parallel Coordinates", filename=None):
    import pandas as pd
    df = pd.DataFrame([s.f for s in solutions])
    plt.figure()
    for i in range(len(df)):
        plt.plot(df.columns, df.iloc[i], alpha=0.6)
    plt.title(title)
    plt.xlabel("Objective Index")
    plt.ylabel("Objective Value")
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def plot_radar_chart(solutions, title="Radar Chart of Objectives", filename=None):
    from math import pi
    n_obj = len(solutions[0].f)
    labels = [f'Obj {i+1}' for i in range(n_obj)]
    angles = [n * (2 * pi / n_obj) for n in range(n_obj)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for sol in solutions:
        values = list(sol.f) + [sol.f[0]]
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1], labels)
    ax.set_title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

from matplotlib.animation import FuncAnimation

def animate_abstract_search_process(population_history, pareto_history, bounds, title="Algorithm Search Process", filename=None):
    fig, ax = plt.subplots()
    lb, ub = bounds
    ax.set_xlim(lb, ub)
    ax.set_ylim(lb, ub)
    ax.set_title(title)
    scat = ax.scatter([], [], c=[], cmap='viridis', alpha=0.7, edgecolors='k')

    def update(frame):
        pop = population_history[frame]
        xs = [s.x[0] for s in pop]
        ys = [s.x[1] for s in pop]
        colors = [s.f[0] for s in pop]  # 可改为多个目标聚合值
        scat.set_offsets(np.c_[xs, ys])
        scat.set_array(np.array(colors))
        return scat,

    ani = FuncAnimation(fig, update, frames=len(population_history), interval=100, blit=True)

    if filename:
        ani.save(filename, writer='pillow', fps=10)
    else:
        plt.show()