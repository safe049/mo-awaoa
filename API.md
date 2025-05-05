# MO-AWAOA API Documentation

[[README]](README.md)

The **Multi-Objective Adaptive Whale Optimization Algorithm (MO-AWAOA)** is a Python-based optimization framework that supports both single- and multi-objective optimization problems. It implements an adaptive variant of the Whale Optimization Algorithm (WOA) with enhanced exploration and exploitation mechanisms, such as Lévy flight mutation, Gaussian Barebone mutation, and Quasi-Opposition-Based Learning (QOBL).

---

## Table of Contents

1. [Installation](#installation)
2. [Overview](#overview)
3. [Core Classes](#core-classes)
   - [Problem](#problem)
   - [Solution](#solution)
   - [Population](#population)
   - [Archive](#archive)
4. [Optimization Algorithm](#optimization-algorithm-moawoa)
5. [Utils Module](#utils-module)
6. [Example Usage](#example-usage)

---

## Installation

To use this package, ensure you have Python 3.8+ installed along with the following dependencies:

```bash
pip install numpy matplotlib scikit-learn deap scipy
```

Clone or download the source code and import the necessary modules into your project.

---

## Overview

This package provides a modular architecture for defining optimization problems and solving them using the MO-AWAOA algorithm. The main components are:

- `Problem`: Defines the objective function and search space.
- `Solution`: Represents a candidate solution to the problem.
- `Population`: Manages a group of solutions.
- `Archive`: Maintains non-dominated solutions in multi-objective optimization.
- `MOAWOA`: The core optimizer class implementing the Whale Optimization Algorithm.

---

## Core Classes

### Problem

#### Class Definition:
```python
class Problem:
    def __init__(self, obj_func, bounds, dim, num_objs):
```

#### Parameters:
- `obj_func`: Callable function that evaluates a solution vector and returns its objective value(s).
- `bounds`: Tuple `(lb, ub)` representing lower and upper bounds for each decision variable.
- `dim`: Integer dimensionality of the problem.
- `num_objs`: Number of objectives (1 for single-objective, >1 for multi-objective).

#### Methods:
- `evaluate(x)`: Evaluates the solution `x` using the objective function.

---

### Solution

#### Class Definition:
```python
class Solution:
    def __init__(self, x, f):
```

#### Parameters:
- `x`: Numpy array representing the decision variables.
- `f`: Objective value(s) — either float or list of floats.

#### Attributes:
- `x`: Decision variables.
- `f`: Objective values.
- `rank`: Pareto rank (used in NSGA-II-style sorting).
- `crowding_dist`: Crowding distance (used in diversity preservation).

#### Methods:
- `dominates(other: Solution) -> bool`: Returns `True` if this solution dominates the other solution.

---

### Population

#### Class Definition:
```python
class Population:
    def __init__(self, problem: Problem, size: int, initial_solutions=None):
```

#### Parameters:
- `problem`: Instance of `Problem`.
- `size`: Number of individuals in the population.
- `initial_solutions`: Optional list of pre-defined solution vectors.

#### Methods:
- `_initialize()`: Randomly initializes the population within bounds.
- `update(idx, solution)`: Updates the individual at index `idx`.

---

### Archive

#### Class Definition:
```python
class Archive:
    def __init__(self, max_size=100):
```

#### Parameters:
- `max_size`: Maximum number of non-dominated solutions the archive can store.

#### Methods:
- `add(solution: Solution)`: Adds a new solution to the archive and removes any dominated solutions.
- `_limit_size()`: Limits the archive size by removing least crowded individuals based on crowding distance.
- `get_centers()`: Returns the decision vectors of all archived solutions.

---

## Optimization Algorithm (MOAWOA)

### Class Definition:
```python
class MOAWOA:
    def __init__(self, obj_func, bounds, dim, num_objs=1, pop_size=50, max_iter=100,
                 archive_size=100, verbose=False, checkpoint_interval=10):
```

### Parameters:
- `obj_func`: Objective function accepting a NumPy array input and returning scalar or list output.
- `bounds`: Tuple `(lb, ub)` specifying variable bounds.
- `dim`: Dimensionality of the problem.
- `num_objs`: Number of objectives (1 for single-objective).
- `pop_size`: Size of the population.
- `max_iter`: Maximum number of iterations.
- `archive_size`: Maximum number of non-dominated solutions stored (for multi-objective).
- `verbose`: If `True`, prints progress during optimization.
- `checkpoint_interval`: Interval (in iterations) to print logs or save checkpoints.

### Methods:

#### `optimize() -> List[Solution]`
Runs the optimization process and returns:
- For single-objective: A list containing the best solution.
- For multi-objective: A list of non-dominated solutions from the archive.

#### `_is_better(f1, f2)`
Compares two fitness vectors and returns `True` if `f1` dominates or is better than `f2`.

#### `_initialize_population_with_qobl()`
Initializes the population using Quasi-Opposition-Based Learning for faster convergence.

#### `_restart_population()`
Restarts part of the population when stagnation is detected.

---

## Utils Module

The `utils.py` module contains utility functions used across the framework, including dominance checking, performance metrics, visualization tools, and checkpointing functionality.

---

### 🔧 Core Utility Functions

#### `dominates(obj1, obj2) -> bool`
Checks whether one objective vector dominates another.

**Parameters:**
- `obj1`: Objective vector of the first solution.
- `obj2`: Objective vector of the second solution.

**Returns:**
- `True` if `obj1` dominates `obj2`, otherwise `False`.

---

#### `crowding_distance_assignment(solutions)`
Calculates and assigns crowding distances to a list of solutions. This is used in multi-objective optimization algorithms like NSGA-II to maintain diversity in the population.

**Parameters:**
- `solutions`: A list of `Solution` objects.

**Modifies:**
- Each `Solution` object's `crowding_dist` attribute is updated.

---

#### `hypervolume(ref_point, solutions) -> float`
Computes the hypervolume metric given a reference point and a set of non-dominated solutions.

**Parameters:**
- `ref_point`: Reference point (list or array) for computing hypervolume.
- `solutions`: List of `Solution` objects representing the approximation front.

**Returns:**
- The computed hypervolume value.

**Note:** Requires `DEAP`'s hypervolume implementation.

---

#### `igd(ref_set, approx_set) -> float`
Calculates the Inverted Generational Distance (IGD), which measures the average distance from the true Pareto front (`ref_set`) to the approximated front (`approx_set`).

**Parameters:**
- `ref_set`: True Pareto front (numpy array of objective vectors).
- `approx_set`: Approximation set (list of `Solution` objects).

**Returns:**
- The IGD value.

**Dependencies:**
- Uses `sklearn.metrics.pairwise_distances`.

---

### 📊 Visualization Tools

#### `plot_pareto_front(solutions, title="Pareto Front", filename=None)`
Plots the Pareto front for 2D or 3D objective spaces.

**Parameters:**
- `solutions`: List of `Solution` objects.
- `title`: Title of the plot.
- `filename`: If provided, saves the figure to this file path; otherwise displays it.

---

####  `plot_parallel_coordinates(solutions, title="Parallel Coordinates", filename=None)`
Plots a parallel coordinates chart to visualize high-dimensional objective space.

**Parameters:**
- `solutions`: List of `Solution` objects.
- `title`: Plot title.
- `filename`: Optional output file path.

---

####  `plot_radar_chart(solutions, title="Radar Chart of Objectives", filename=None)`
Plots a radar chart (spider plot) showing each solution's objective values.

**Parameters:**
- `solutions`: List of `Solution` objects.
- `title`: Plot title.
- `filename`: Optional output file path.

---

####  `animate_abstract_search_process(population_history, pareto_history, bounds, title="Algorithm Search Process", filename=None)`
Abstractly visualizes how the algorithm explores the search space.

**Parameters:**
- `population_history`: List of populations at each iteration.
- `pareto_history`: List of Pareto fronts at each iteration.
- `bounds`: Tuple `(lb, ub)` representing lower and upper bounds of the search space.
- `title`: Animation title.
- `filename`: If provided, saves animation as GIF; otherwise displays it.

---

### 💾 Checkpointing Functions

#### `save_checkpoint(population, archive, iteration, filename="checkpoint.json")`
Saves the current state of the population, archive, and iteration count to a JSON file.

**Parameters:**
- `population`: Current `Population` object.
- `archive`: Current `Archive` object (for multi-objective algorithms).
- `iteration`: Current iteration number.
- `filename`: File path to save the checkpoint.

---

#### `load_checkpoint(problem, filename="checkpoint.json") -> Tuple[Population, Archive, int]`
Loads a saved state from a JSON file.

**Parameters:**
- `problem`: Problem definition used to recreate solutions.
- `filename`: Path to the checkpoint file.

**Returns:**
- A tuple `(population, archive, iteration)`.

---

### 📌 Example Usage - Utils Module

```python
# Save state during optimization
save_checkpoint(population, archive, iteration, "checkpoint.json")

# Load state later
population, archive, iteration = load_checkpoint(problem, "checkpoint.json")
```

```python
# Plotting example
plot_pareto_front(archive.members, title="Final Pareto Front", filename="pareto_final.png")
plot_parallel_coordinates(archive.members, title="Objective Space Comparison")
plot_radar_chart(archive.members)
```

```python
# Animation example
animate_abstract_search_process(population_history, history, bounds=(-5, 5), filename="search_process.gif")
```

---

### 🧠 Tips

- Use `plot_parallel_coordinates` and `plot_radar_chart` when working with 4+ objectives.
- Use animations to demonstrate convergence behavior in reports or presentations.
- You can extend these plotting functions to include color coding by generation or fitness value.


---

## Example Usage

### Single-Objective Optimization

```python
def sphere(x):
    return np.sum(x**2)

moawoa = MOAWOA(
    obj_func=sphere,
    bounds=(-5.12, 5.12),
    dim=30,
    num_objs=1,
    pop_size=50,
    max_iter=200,
    verbose=True
)

best_solution = moawoa.optimize()[0]
print("Best solution:", best_solution.x)
print("Fitness:", best_solution.f)
```

### Multi-Objective Optimization (ZDT1)

```python
def zdt1(x):
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    f1 = x[0]
    f2 = g * (1 - np.sqrt(f1 / g))
    return [f1, f2]

moawoa = MOAWOA(
    obj_func=zdt1,
    bounds=(0, 1),
    dim=30,
    num_objs=2,
    pop_size=100,
    max_iter=200,
    archive_size=100,
    verbose=True
)

pareto_front = moawoa.optimize()
from moawaoa.utils import plot_pareto_front
plot_pareto_front(pareto_front)
```

---

## Notes

- The algorithm automatically switches between single- and multi-objective modes based on `num_objs`.
- You can load and resume from saved checkpoints using `save_checkpoint()` and `load_checkpoint()`.
- Metrics like Hypervolume and IGD are available for performance evaluation.

---

## License

MIT License — see the LICENSE file for details.

---

