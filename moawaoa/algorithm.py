import numpy as np
from .base import Solution, Problem, Population, Archive
from .utils import dominates, crowding_distance_assignment
from scipy.special import gamma
from matplotlib.animation import FuncAnimation


def levy_flight(beta=1.5):
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, size=1)
    v = np.random.normal(0, 1, size=1)
    step = u / (abs(v) ** (1 / beta))
    return step


def quasi_opposite_learning(x, lb, ub):
    """准反向学习：生成当前解的对立解"""
    return lb + ub - x


class MOAWOA:
    def __init__(self, obj_func, bounds, dim, num_objs=1, pop_size=50, max_iter=100,
                 archive_size=100, verbose=False, checkpoint_interval=10):
        self.problem = Problem(obj_func, bounds, dim, num_objs)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.num_objs = num_objs
        self.verbose = verbose
        self.checkpoint_interval = checkpoint_interval
        self.population = None
        self.best_solution = None
        self.archive = None

        # 根据目标数量决定是否启用多目标机制
        if self.num_objs == 1:
            self.best_solution = None
        else:
            self.archive = Archive(max_size=archive_size)

    def _is_better(self, f1, f2):
        """通用比较函数：支持单/多目标"""
        if self.num_objs == 1:
            return f1[0] < f2[0]
        else:
            return dominates(f1, f2)

    def optimize(self):
        # 初始化种群并加入 QOBL 机制
        self.population = self._initialize_population_with_qobl()
        
        # 初始化全局最优
        if self.num_objs == 1:
            self.best_solution = min(self.population.solutions, key=lambda s: s.f[0])
        else:
            for sol in self.population.solutions:
                self.archive.add(sol)

        stagnation_counter = 0
        prev_best_fitness = float('inf')

        for iter in range(self.max_iter):
            a = 2 - iter * (2 / self.max_iter)
            p = np.random.rand()

            # 获取引导点
            if self.num_objs == 1:
                leader = self.best_solution.x
            else:
                leader = self.archive.get_centers()[0] if len(self.archive.members) > 0 else self.population.solutions[0].x

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.problem.dim), np.random.rand(self.problem.dim)
                A = 2 * a * r1 - a
                C = 2 * r2

                # WOA 主体逻辑
                if p < 0.5:
                    if abs(A).max() < 1:
                        D = abs(C * leader - self.population.solutions[i].x)
                        new_x = leader - A * D
                    else:
                        rand_idx = np.random.randint(0, self.pop_size)
                        rand_leader = self.population.solutions[rand_idx].x
                        D = abs(C * rand_leader - self.population.solutions[i].x)
                        new_x = rand_leader - A * D
                else:
                    D = abs(leader - self.population.solutions[i].x)
                    b = 1
                    l = np.random.uniform(-1, 1)
                    new_x = D * np.exp(b * l) * np.cos(2 * np.pi * l)

                new_x = np.clip(new_x, *self.problem.bounds)
                new_f = self.problem.evaluate(new_x)
                new_sol = Solution(new_x, new_f)

                # 更新个体
                if self._is_better(new_sol.f, self.population.solutions[i].f):
                    self.population.update(i, new_sol)

                    # 更新全局最优
                    if self.num_objs == 1:
                        if self._is_better(new_sol.f, self.best_solution.f):
                            self.best_solution = new_sol
                            stagnation_counter = 0
                        else:
                            stagnation_counter += 1
                    else:
                        self.archive.add(new_sol)
                        stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # Lévy 飞行变异
                if np.random.rand() < 0.1:
                    step = levy_flight()
                    mutated_x = new_x + step * np.random.randn(self.problem.dim)
                    mutated_x = np.clip(mutated_x, *self.problem.bounds)
                    mutated_f = self.problem.evaluate(mutated_x)
                    mutated_sol = Solution(mutated_x, mutated_f)
                    if self._is_better(mutated_sol.f, new_sol.f):
                        self.population.update(i, mutated_sol)
                        if self.num_objs == 1:
                            if self._is_better(mutated_sol.f, self.best_solution.f):
                                self.best_solution = mutated_sol
                        else:
                            self.archive.add(mutated_sol)

                # Gaussian Barebone 变异
                if np.random.rand() < 0.05:
                    mu = self.best_solution.x if self.num_objs == 1 else leader
                    sigma = abs(mu - new_x) + 1e-6
                    mutated_x = np.random.normal(mu, sigma, size=self.problem.dim)
                    mutated_x = np.clip(mutated_x, *self.problem.bounds)
                    mutated_f = self.problem.evaluate(mutated_x)
                    mutated_sol = Solution(mutated_x, mutated_f)
                    if self._is_better(mutated_sol.f, new_sol.f):
                        self.population.update(i, mutated_sol)
                        if self.num_objs == 1:
                            if self._is_better(mutated_sol.f, self.best_solution.f):
                                self.best_solution = mutated_sol
                        else:
                            self.archive.add(mutated_sol)

            # 检查停滞并触发重启
            current_best_fitness = self.best_solution.f[0] if self.num_objs == 1 else min(s.f[0] for s in self.archive.members)
            if abs(current_best_fitness - prev_best_fitness) < 1e-6:
                self._restart_population()
            prev_best_fitness = current_best_fitness

            # 打印日志
            if self.verbose and (iter % self.checkpoint_interval == 0):
                if self.num_objs == 1:
                    print(f"Iteration {iter}: Best Fitness = {self.best_solution.f[0]:.6f}")
                else:
                    print(f"Iteration {iter}: Archive size = {len(self.archive.members)}")

        # 返回结果统一格式
        if self.num_objs == 1:
            return [self.best_solution]
        else:
            self.archive.members = self._remove_duplicate_solutions(self.archive.members)
            return self.archive.members

    def _initialize_population_with_qobl(self):
        """使用 QOBL 初始化种群"""
        lb, ub = self.problem.bounds
        half_pop = self.pop_size // 2
        pop_x = np.random.uniform(lb, ub, size=(half_pop, self.problem.dim))
        opposite_pop = quasi_opposite_learning(pop_x, lb, ub)
        full_pop = np.vstack([pop_x, opposite_pop])[:self.pop_size]
        return Population(self.problem, self.pop_size, initial_solutions=full_pop)

    def _restart_population(self):
        """当陷入局部最优时，重启部分种群"""
        lb, ub = self.problem.bounds
        n_restart = int(0.3 * self.pop_size)
        restart_x = np.random.uniform(lb, ub, size=(n_restart, self.problem.dim))
        for i in range(n_restart):
            new_sol = Solution(restart_x[i], self.problem.evaluate(restart_x[i]))
            self.population.update(i, new_sol)


    def _remove_duplicate_solutions(self, solutions):
        seen = set()
        unique_solutions = []
        for sol in solutions:
            key = (tuple(np.round(sol.x, 5)), tuple(np.round(sol.f, 5)))
            if key not in seen:
                seen.add(key)
                unique_solutions.append(sol)
        return unique_solutions

