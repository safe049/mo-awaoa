import numpy as np
from typing import Callable

class QGBWOA:
    def __init__(self, obj_func, bounds, dim, pop_size=30, max_iter=100, verbose=False):
        self.obj_func = obj_func
        self.bounds = bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.verbose = verbose
        # 初始化种群（结合 QOBL）
        self.population = self._initialize_population()
        self.fitness = np.array([self.obj_func(ind) for ind in self.population])
        self.gbest = self.population[np.argmin(self.fitness)]
        self.gbest_fitness = min(self.fitness)

    def _initialize_population(self):
        # 使用 QOBL 初始化
        pop = np.random.uniform(*self.bounds, size=(self.pop_size // 2, self.dim))
        opposite_pop = self.bounds[0] + self.bounds[1] - pop
        return np.vstack((pop, opposite_pop))

    def optimize(self):
        history = []
        for iter in range(self.max_iter):
            a = 2 - iter * (2 / self.max_iter)
            for i in range(self.pop_size):
                r1 = np.random.rand()
                r2 = np.random.rand()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = np.random.rand()

                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * self.gbest - self.population[i])
                        new_pos = self.gbest - A * D
                    else:
                        rand_idx = np.random.randint(0, self.pop_size)
                        rand_leader = self.population[rand_idx]
                        D = abs(C * rand_leader - self.population[i])
                        new_pos = rand_leader - A * D
                else:
                    D = abs(self.gbest - self.population[i])
                    b = 1
                    l = np.random.uniform(-1, 1)
                    new_pos = D * np.exp(b * l) * np.cos(2 * np.pi * l)

                # Gaussian Barebone Mutation
                if np.random.rand() < 0.1:
                    new_pos = np.random.normal(loc=self.gbest, scale=abs(new_pos - self.gbest), size=self.dim)

                new_pos = np.clip(new_pos, *self.bounds)
                new_fit = self.obj_func(new_pos)

                if new_fit < self.fitness[i]:
                    self.population[i] = new_pos
                    self.fitness[i] = new_fit
                    if new_fit < self.gbest_fitness:
                        self.gbest = new_pos
                        self.gbest_fitness = new_fit

            history.append(self.gbest_fitness)
            if self.verbose and iter % 10 == 0:
                print(f"Iteration {iter}, Best Fitness: {self.gbest_fitness:.6f}")

        return self.gbest, self.gbest_fitness, history