import numpy as np


def dominates(obj1, obj2):
    """判断目标向量 obj1 是否支配 obj2"""
    better = np.less(obj1, obj2)
    at_least_one_better = np.any(better)
    all_better_or_equal = np.all(np.less_equal(obj1, obj2))
    return at_least_one_better and all_better_or_equal


class Solution:
    def __init__(self, x, f):
        self.x = x
        if isinstance(f, (int, float)):
            self.f = [float(f)]  # 自动转为列表
        else:
            self.f = list(f)     # 确保是列表
        self.rank = 0
        self.crowding_dist = 0

    def dominates(self, other):
        """判断当前解是否支配另一个解（other 是另一个 Solution）"""
        return dominates(self.f, other.f)


class Problem:
    def __init__(self, obj_func, bounds, dim, num_objs):
        self.obj_func = obj_func
        self.bounds = bounds      # (lb, ub)
        self.dim = dim
        self.num_objs = num_objs

    def evaluate(self, x):
        return self.obj_func(x)


class Population:
    def __init__(self, problem: Problem, size: int, initial_solutions=None):
        self.problem = problem
        self.size = size
        if initial_solutions is not None:
            # 使用给定初始解
            self.solutions = [Solution(x, self.problem.evaluate(x)) for x in initial_solutions]
        else:
            # 随机初始化
            self.solutions = self._initialize()

    def _initialize(self):
        lb, ub = self.problem.bounds
        pop_x = np.random.uniform(lb, ub, size=(self.size, self.problem.dim))
        return [Solution(x, self.problem.evaluate(x)) for x in pop_x]

    def update(self, idx, solution: Solution):
        self.solutions[idx] = solution


class Archive:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.members = []

    def add(self, solution: Solution):
        # 添加非支配解到外部存档，并删除被支配的旧成员
        dominated = []
        for i, s in enumerate(self.members):
            if solution.dominates(s):  # ✅ 这里调用的是实例方法
                dominated.append(i)
        for i in reversed(dominated):
            self.members.pop(i)
        if not any(solution.dominates(s) for s in self.members):
            self.members.append(solution)
            if len(self.members) > self.max_size:
                self._limit_size()

    def _limit_size(self):
        from moawaoa.utils import crowding_distance_assignment
        crowding_distance_assignment(self.members)
        self.members.sort(key=lambda s: s.crowding_dist, reverse=True)
        self.members = self.members[:self.max_size]

    def get_centers(self):
        return np.array([s.x for s in self.members])