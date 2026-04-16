"""IOE 511/MATH 562, University of Michigan
Code template written by: Albert S. Berahas & Jiahao Shi
Implementations by: Kayla Huang, Wuhao Cao, & Thomas Xu ZHANG
"""

import numpy as np
import algorithms
import functions


def optSolver_Fire_Horse(problem, method, options):
    # Initialization
    x = problem.x0
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    norm_g = np.linalg.norm(g, ord=np.inf)

    n = len(x)
    k = 0

    H = (
        np.eye(n)
        if method.name in ["BFGS", "BFGSW", "DFP", "DFPW", "TRSR1CG"]
        else None
    )
    x_old, g_old = None, None

    # trust region initialization
    Delta = (
        getattr(options, "initial_radius", 1.0)
        if method.name in ["TRNewtonCG", "TRSR1CG"]
        else None
    )

    # Fetch tolerances
    term_tol = getattr(options, "term_tol", 1e-6)
    max_iter = getattr(options, "max_iterations", 1000)

    # Main Optimization Loop
    while norm_g > term_tol and k < max_iter:

        match method.name:
            case "GradientDescent":
                x_new, f_new, g_new, d, alpha = algorithms.GradientDescent(
                    x, f, g, problem, method, options
                )

            case "GradientDescentW":
                x_new, f_new, g_new, d, alpha = algorithms.GradientDescentW(
                    x, f, g, problem, method, options
                )

            case "Newton":
                x_new, f_new, g_new, d, alpha = algorithms.Newton(
                    x, f, g, problem, method, options
                )

            case "NewtonW":
                x_new, f_new, g_new, d, alpha = algorithms.NewtonW(
                    x, f, g, problem, method, options
                )

            case "TRNewtonCG":
                x_new, f_new, g_new, d, Delta = algorithms.TRNewtonCG(
                    x, f, g, Delta, problem, method, options
                )

            case "TRSR1CG":
                x_new, f_new, g_new, H, d, Delta = algorithms.TRSR1CG(
                    x, x_old, f, g, g_old, H, Delta, k, problem, method, options
                )

            case "BFGS":
                x_new, f_new, g_new, H, d, alpha = algorithms.BFGS(
                    x, x_old, f, g, g_old, H, k, problem, method, options
                )

            case "BFGSW":
                x_new, f_new, g_new, H, d, alpha = algorithms.BFGSW(
                    x, x_old, f, g, g_old, H, k, problem, method, options
                )

            case "DFP":
                x_new, f_new, g_new, H, d, alpha = algorithms.DFP(
                    x, x_old, f, g, g_old, H, k, problem, method, options
                )

            case "DFPW":
                x_new, f_new, g_new, H, d, alpha = algorithms.DFPW(
                    x, x_old, f, g, g_old, H, k, problem, method, options
                )

            case _:
                raise ValueError(f"Method '{method.name}' is not recognized.")

        # Update tracking variables
        x_old = x
        f_old = f
        g_old = g

        x = x_new
        f = f_new
        g = g_new
        norm_g = np.linalg.norm(g, ord=np.inf)

        k += 1

    return x, f
