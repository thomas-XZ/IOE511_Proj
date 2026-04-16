# File Structure

- **`optSolver.py`**
  The main entry point for our package. It contains the `optSolver_Fire_Horse(problem, method, options)` function. It uses functions in algorithms.py to solve problems.

- **`algorithms.py`**
  It has the step computations for the 10 algorithms (e.g., `GradientDescent`, `Newton`, `BPGS`, `TRNewtonCG`).

- **`project_problems.py`**
  The to-be-computed functions (problems) are defined here.

- **`helper_func.py`**
  Helper functions used in algorithms.py
