# Multi-Objective Adaptive Whale Optimization Algorithm (MO-AWAOA)

[[API Documentation]](API.md)

This repository contains the implementation of the Multi-Objective Adaptive Whale Optimization Algorithm (MO-AWAOA), a nature-inspired optimization algorithm designed to solve both single-objective and multi-objective optimization problems. The algorithm integrates several enhancements, including quasi-oppositional learning, Lévy flight mutation, Gaussian Barebone mutation, and population restart mechanisms to improve convergence and diversity in the solution set.

---

## 🔧 Features

- **Multi-Objective Support**: Uses an external archive to store non-dominated solutions and applies crowding distance for diversity maintenance.
- **Quasi-Oppositional Learning (QOBL)**: Enhances exploration during initialization by generating opposite solutions.
- **Lévy Flight Mutation**: Introduces long-range jumps to escape local optima.
- **Gaussian Barebone Mutation**: Exploits promising regions using a normal distribution centered around the best solution.
- **Population Restart Strategy**: Reinitializes part of the population when stagnation is detected.

---

## 📁 File Structure

- `base.py`: Core classes (`Solution`, `Problem`, `Population`, `Archive`) representing individuals, objective functions, populations, and external archives.
- `utils.py`: Utility functions for dominance checks, crowding distance calculation, performance metrics (hypervolume, IGD), visualization, and checkpointing.
- `algorithm.py`: Main MO-AWAOA implementation with support for single and multi-objective optimization.

---

## 🧠 Classes and Functions

### Base Classes
- `Solution`: Encapsulates decision variables (`x`), objective values (`f`), rank, and crowding distance.
- `Problem`: Wraps the objective function and problem bounds.
- `Population`: Manages a collection of `Solution` instances.
- `Archive`: Maintains a set of non-dominated solutions for multi-objective optimization.

### Utilities
- `dominates`: Determines whether one solution dominates another.
- `crowding_distance_assignment`: Assigns crowding distances to solutions for diversity preservation.
- `hypervolume`, `igd`: Performance indicators for evaluating multi-objective algorithms.
- `plot_pareto_front`: Visualizes the Pareto front in 2D or 3D.
- `save_checkpoint`, `load_checkpoint`: Supports saving and restoring algorithm states.

### Algorithm
- `MOAWOA`: Main optimizer class supporting both single and multi-objective problems.
  - `_initialize_population_with_qobl`: Initializes population using QOBL strategy.
  - `_restart_population`: Restarts part of the population to avoid stagnation.
  - `levy_flight`, `quasi_opposite_learning`: Mutation strategies for exploration and exploitation.

---

## 🚀 Usage Example

```python
import numpy as np
from moawaoa.algorithm import MOAWOA

# Define your objective function
def obj_func(x):
    # Example: ZDT1 test function
    g = 1 + 9 * np.sum(x[1:]) / len(x)
    f1 = x[0]
    f2 = g * (1 - np.sqrt(f1 / g))
    return [f1, f2]

# Set up the problem
bounds = (0, 1)
dim = 30
num_objs = 2
pop_size = 50
max_iter = 100

# Run the optimizer
optimizer = MOAWOA(obj_func, bounds, dim, num_objs, pop_size, max_iter, verbose=True)
pareto_front = optimizer.optimize()

# Plot results
from moawaoa.utils import plot_pareto_front
plot_pareto_front(pareto_front)
```

---

## 📈 Performance Metrics

- **Hypervolume**: Measures the volume of the objective space dominated by the approximation set.
- **Inverted Generational Distance (IGD)**: Evaluates both convergence and diversity of the obtained solutions compared to a reference set.

---

## 📦 Dependencies

- `numpy`
- `matplotlib`
- `scikit-learn` (for IGD computation)
- `deap` (for hypervolume calculation)

Install dependencies using:

```bash
pip install numpy matplotlib scikit-learn deap
```

---

## 📝 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For questions, suggestions, or contributions, feel free to open an issue or contact us directly.

--- 

