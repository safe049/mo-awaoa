import numpy as np
from .base import Solution, Problem, Population, Archive
from .utils import dominates, crowding_distance_assignment
from scipy.special import gamma
from scipy.stats import qmc
import math
import time

# ———————————————————————————————— 工具函数 ————————————————————————————————

def levy_flight(beta=1.5):
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, size=1)
    v = np.random.normal(0, 1, size=1)
    step = u / (abs(v) ** (1 / beta))
    return step

def quasi_opposite_learning(x, lb, ub):
    return lb + ub - x

def de_mutation(population, F=0.5):
    indices = np.random.choice(len(population), size=3, replace=False)
    a, b, c = population[indices[0]], population[indices[1]], population[indices[2]]
    return a.x + F * (b.x - c.x)

def local_search(x, lb, ub, sigma=0.1):
    perturbation = np.random.normal(0, sigma, size=len(x)) + \
                   np.random.uniform(-sigma, sigma, size=len(x))  # 高斯+均匀混合扰动
    return np.clip(x + perturbation, lb, ub)

# ———————————————————————————————— 主类 ————————————————————————————————

class MOAWOA:
    def __init__(self, obj_func, bounds, dim, num_objs=1, pop_size=50, max_iter=100,
                 archive_size=100, verbose=False, checkpoint_interval=10, permutation=False):
        self.problem = Problem(obj_func, bounds, dim, num_objs)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.num_objs = num_objs
        self.verbose = verbose
        self.checkpoint_interval = checkpoint_interval
        self.permutation = permutation
        self.population = None
        self.best_solution = None
        self.archive = None

        if self.num_objs == 1:
            self.best_solution = None
        else:
            self.archive = Archive(max_size=archive_size)

        # 变异参数（动态调整）
        self.levy_rate = 0.2
        self.de_rate = 0.3
        self.local_search_rate = 0.1
        self.restart_threshold = 5  # 停滞多少次后重启
        self.stagnation_counter = 0
        self.hall_of_fame = []

    def _is_better(self, f1, f2):
        if self.num_objs == 1:
            return f1[0] < f2[0]
        else:
            return dominates(f1, f2)

    def optimize(self):
        start_time = time.time()
        # 初始化种群（QOBL + LHS + Sobol）
        self.population = self._initialize_population_with_qobl_lhs_sobol()
        if not self.population.solutions:
            print("⚠️ 初始化失败：种群为空，请检查目标函数和边界设置")
            return []

        # 初始化全局最优
        if self.num_objs == 1:
            self.best_solution = min(self.population.solutions, key=lambda s: s.f[0])
        else:
            for sol in self.population.solutions:
                self.archive.add(sol)

        prev_best_fitness = float('inf')
        stagnation_counter = 0

        # 自适应参数
        elite_pool = []
        diversity_factor = 1.0

        for iter in range(self.max_iter):
            # 自适应参数调整
            a = 2 * (1 - (iter / self.max_iter)) ** 2.5  # 非线性衰减
            p = np.random.rand()

            # 获取引导点（精英池）
            if self.num_objs == 1:
                leader = self.best_solution.x
            else:
                if len(self.archive.members) > 0:
                    leaders = self.archive.get_centers(k=min(5, len(self.archive.members)))
                    leader = leaders[np.random.randint(len(leaders))]
                else:
                    leader = self.population.solutions[0].x

            # 动态调整变异率
            self._adjust_mutation_rates(iter)

            # 主循环
            improved = False
            for i in range(self.pop_size):
                sol = self.population.solutions[i]
                r1, r2 = np.random.rand(self.problem.dim), np.random.rand(self.problem.dim)
                A = 2 * a * r1 - a
                C = 2 * r2

                # WOA 核心逻辑
                if p < 0.6:
                    if abs(A).max() < 1:
                        D = abs(C * leader - sol.x)
                        new_x = leader - A * D
                    else:
                        rand_idx = np.random.randint(0, self.pop_size)
                        rand_leader = self.population.solutions[rand_idx].x
                        D = abs(C * rand_leader - sol.x)
                        new_x = rand_leader - A * D
                else:
                    D = abs(leader - sol.x)
                    l = np.random.uniform(-1, 1)
                    new_x = D * np.exp(l) * np.cos(2 * np.pi * l)

                new_x = np.clip(new_x, *self.problem.bounds)
                new_f = self.problem.evaluate(new_x)
                new_sol = Solution(new_x, new_f)

                # 更新个体
                if self._is_better(new_sol.f, sol.f):
                    self.population.update(i, new_sol)
                    improved = True
                    if self.num_objs == 1 and self._is_better(new_sol.f, self.best_solution.f):
                        self.best_solution = new_sol
                        stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # 多样化变异
                if np.random.rand() < self.de_rate:
                    mutant_x = de_mutation(self.population.solutions)
                    mutant_x = np.clip(mutant_x, *self.problem.bounds)
                    mutant_f = self.problem.evaluate(mutant_x)
                    mutant_sol = Solution(mutant_x, mutant_f)
                    if self._is_better(mutant_sol.f, new_sol.f):
                        self.population.update(i, mutant_sol)
                        if self.num_objs == 1 and self._is_better(mutant_sol.f, self.best_solution.f):
                            self.best_solution = mutant_sol

                if np.random.rand() < self.levy_rate:
                    step = levy_flight()
                    mutated_x = new_x + step * np.random.randn(self.problem.dim)
                    mutated_x = np.clip(mutated_x, *self.problem.bounds)
                    mutated_f = self.problem.evaluate(mutated_x)
                    mutated_sol = Solution(mutated_x, mutated_f)
                    if self._is_better(mutated_sol.f, new_sol.f):
                        self.population.update(i, mutated_sol)
                        if self.num_objs == 1 and self._is_better(mutated_sol.f, self.best_solution.f):
                            self.best_solution = mutated_sol

                if np.random.rand() < self.local_search_rate:
                    ls_x = local_search(new_x, *self.problem.bounds)
                    ls_f = self.problem.evaluate(ls_x)
                    ls_sol = Solution(ls_x, ls_f)
                    if self._is_better(ls_sol.f, new_sol.f):
                        self.population.update(i, ls_sol)
                        if self.num_objs == 1 and self._is_better(ls_sol.f, self.best_solution.f):
                            self.best_solution = ls_sol

            # 更新档案
            if self.num_objs > 1:
                for sol in self.population.solutions:
                    self.archive.add(sol)

            # 停滞检测与重启
            try:
                current_best_fitness = self.best_solution.f[0] if self.num_objs == 1 else min(s.f[0] for s in self.archive.members)
            except Exception:
                print("⚠️ 当前最优解无效，正在重启种群...")
                self._restart_population()
                continue

            if abs(current_best_fitness - prev_best_fitness) < 1e-8:
                stagnation_counter += 1
                if stagnation_counter >= self.restart_threshold:
                    self._restart_population()
                    stagnation_counter = 0
            else:
                stagnation_counter = 0

            prev_best_fitness = current_best_fitness

            # 日志输出
            if self.verbose and (iter % self.checkpoint_interval == 0):
                elapsed = time.time() - start_time
                if self.num_objs == 1:
                    print(f"Iteration {iter}: Best Fitness = {current_best_fitness:.6f}, Time Elapsed = {elapsed:.2f}s")
                else:
                    print(f"Iteration {iter}: Archive size = {len(self.archive.members)}, Time Elapsed = {elapsed:.2f}s")

        # 返回结果
        if self.num_objs == 1:
            return [self.best_solution] if self.best_solution else []
        else:
            try:
                self.archive.members = self._remove_duplicate_solutions(self.archive.members)
                return self.archive.members
            except Exception as e:
                print(f"❌ 存档处理失败: {e}")
                return []

    def _adjust_mutation_rates(self, iter):
        """动态调整变异率"""
        rate_decay = 1 - iter / self.max_iter
        self.levy_rate = 0.1 + 0.3 * rate_decay
        self.de_rate = 0.2 + 0.4 * rate_decay
        self.local_search_rate = 0.05 + 0.2 * (1 - rate_decay)

    def _initialize_population_with_qobl_lhs_sobol(self):
        lb, ub = self.problem.bounds
        half_pop = self.pop_size // 2
        third_pop = self.pop_size // 3

        # QOBL 初始化
        pop_x = np.random.uniform(lb, ub, size=(half_pop, self.problem.dim))
        opposite_pop = quasi_opposite_learning(pop_x, lb, ub)
        full_pop = np.vstack([pop_x, opposite_pop])[:third_pop]

        # Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.problem.dim)
        lhs_samples = sampler.random(n=third_pop)
        lhs_samples = qmc.scale(lhs_samples, lb, ub)

        # Sobol 序列补充
        sobol_sampler = qmc.Sobol(d=self.problem.dim)
        sobol_samples = sobol_sampler.random(self.pop_size - len(full_pop) - len(lhs_samples))
        sobol_samples = qmc.scale(sobol_samples, lb, ub)

        full_pop = np.vstack([full_pop, lhs_samples, sobol_samples])
        return Population(self.problem, self.pop_size, initial_solutions=full_pop)

    def _restart_population(self):
        lb, ub = self.problem.bounds
        n_restart = int(0.5 * self.pop_size)
        restart_x = np.random.uniform(lb, ub, size=(n_restart, self.problem.dim))
        for i in range(n_restart):
            new_sol = Solution(restart_x[i], self.problem.evaluate(restart_x[i]))
            self.population.update(i, new_sol)

    def _remove_duplicate_solutions(self, solutions):
        seen = set()
        unique_solutions = []
        for sol in solutions:
            key = tuple(np.round(sol.x, 5)), tuple(np.round(sol.f, 5))
            if key not in seen:
                seen.add(key)
                unique_solutions.append(sol)
        return unique_solutions