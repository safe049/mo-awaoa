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