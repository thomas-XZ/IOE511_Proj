import numpy as np
from types import SimpleNamespace
from project_problems import rosenbrock_2_func, rosenbrock_2_grad, rosenbrock_2_Hess
from optSolver import optSolver_Fire_Horse

problem = SimpleNamespace(
    name="rosenbrock_2",
    x0=np.array([-1.2, 1.0]),
    compute_f=rosenbrock_2_func,
    compute_g=rosenbrock_2_grad,
    compute_H=rosenbrock_2_Hess,
)
method = SimpleNamespace(
    name="GradientDescent",
    options={
        "step_type": "Backtracking",
        "backtracking_alpha": 1.0,
        "backtracking_beta": 1e-4,
    },
)

options = SimpleNamespace(term_tol=1e-6, max_iterations=1000)

x_final, f_final = optSolver_Fire_Horse(problem, method, options)

print(f"Final Iterate: {x_final}, Final Function Value: {f_final}")
