# woa_optimizer.py
import numpy as np
from typing import List, Tuple, Callable, Optional, Union
from concurrent.futures import ProcessPoolExecutor
import time

class WOAOptimizer:
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        problem_dim: int,
        n_particles: int = 30,
        bounds: Optional[List[Tuple[float, float]]] = None,
        use_parallel: bool = False
    ):
        self.objective_func = objective_func
        self.problem_dim = problem_dim
        self.n_particles = n_particles
        self.bounds = bounds or [(-5, 5)] * problem_dim
        self.use_parallel = use_parallel

        self.positions = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(n_particles, problem_dim)
        )
        self.best_position = None
        self.best_fitness = float('inf')
        self.history = []

    def _evaluate(self, x):
        return self.objective_func(x)

    def optimize(self, max_iter: int):
        start_time = time.time()

        for iter in range(max_iter):
            a = 2 - iter * (2 / max_iter)
            l = np.random.uniform(-1, 1, self.n_particles)

            if self.use_parallel:
                with ProcessPoolExecutor() as executor:
                    fits = list(executor.map(self._evaluate, self.positions))
            else:
                fits = [self._evaluate(pos) for pos in self.positions]

            # 更新最优解
            for i in range(self.n_particles):
                if fits[i] < self.best_fitness:
                    self.best_fitness = fits[i]
                    self.best_position = self.positions[i].copy()

            # 更新位置
            for i in range(self.n_particles):
                p = np.random.rand()
                if p < 0.5:
                    if abs(a) < 1:
                        # 包围猎物
                        r = np.random.rand(self.problem_dim)
                        d = abs(2 * r * self.best_position - self.positions[i])
                        new_pos = self.best_position - a * d
                    else:
                        # 随机搜索
                        rand_index = np.random.randint(0, self.n_particles)
                        rand_pos = self.positions[rand_index]
                        r = np.random.rand(self.problem_dim)
                        d = abs(2 * r * rand_pos - self.positions[i])
                        new_pos = rand_pos - a * d
                else:
                    # 气泡网攻击
                    r = np.random.rand(self.problem_dim)
                    d = abs(self.best_position - self.positions[i])
                    b = 1
                    new_pos = d * np.exp(b * l[i]) * np.cos(2 * np.pi * l[i])

                self.positions[i] = np.clip(new_pos, [b[0] for b in self.bounds], [b[1] for b in self.bounds])

            self.history.append(self.best_fitness)
            print(f"Iteration {iter+1}/{max_iter}, Best Fitness: {self.best_fitness:.6f}")

        print(f"WOA Optimization Complete! Time: {time.time() - start_time:.2f}s")

    def get_best_solution(self) -> np.ndarray:
        return self.best_position

    def get_convergence_history(self):
        return self.history