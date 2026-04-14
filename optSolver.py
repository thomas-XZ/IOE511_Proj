"""IOE 511/MATH 562, University of Michigan
Code written by: Albert S. Berahas & Jiahao Shi
"""

import numpy as np

import algorithms
import functions


def optSolver_Huang_Kayla(problem, method, options):
    """Function that runs a chosen algorithm on a chosen problem

    Inputs:
        problem, method, options (structs)
    Outputs:
        final iterate (x) and final function value (f)
    """

    # compute initial function/gradient/Hessian
    x = problem.x0
    f = problem.compute_f(x)
    g = problem.compute_g(x)  # TODO: Add code!
    # H = (optional in some algorithms)
    # print("x shape:", x.shape)
    # print("g shape:", g.shape)
    norm_g = np.linalg.norm(g, ord=np.inf) #norm of the gradient, used for termination condition
    
    # Initialization for BFGS and L-BFGS
    n = len(x)
    if method.name == "BFGS":
        # Initial inverse Hessian approximation H_0 = I
        H = np.eye(n) 
    elif method.name in ["L-BFGS", "LBFGS"]:
        # Memory lists for L-BFGS
        s_list = []
        y_list = []
        # Assume memory 'm' is stored in method (default to 5 if not found)
        m = getattr(method, 'm', 5) 

    # set initial iteration counter
    k = 0

    while norm_g > options.term_tol and k < options.max_iterations:  # TODO: ADD TERMINATION CONDITION
        match method.name:
            case "GradientDescent": #this is computed in algorithms.py 
                x_new, f_new, g_new, d, alpha = algorithms.GDStep(
                    x, f, g, problem, method, options
                )

            case "Newton":
                H = problem.compute_H(x)
                d = -np.linalg.solve(H, g)
                alpha = 1.0  # constant step size for Newton's method
                x_new = x + alpha * d
                f_new = problem.compute_f(x_new)
                g_new = problem.compute_g(x_new)
            
            case "BFGS":
                if k == 0:
                    H = np.eye(len(x))  # initialize Hessian approximation as identity
                else:
                    s = x - x_old  # change in x
                    y = g - g_old  # change in gradient
                    rho = 1.0 / (y.T @ s)  # scaling factor
                    H = (np.eye(len(x)) - rho * np.outer(s, y)) @ H @ (np.eye(len(x)) - rho * np.outer(y, s)) + rho * np.outer(s, s)  # BFGS update
                d = -H @ g  # search direction
                alpha = 1.0  # constant step size for BFGS
                x_new = x + alpha * d
                f_new = problem.compute_f(x_new)
                g_new = problem.compute_g(x_new)

            case "L-BFGS":
                # 1. search direction using two-loop recursion
                # H_0 is initialized to Identity, so H_0 * q = 1.0 * q. 
                d = -algorithms.lbfgs_two_loop(g, s_list, y_list, H0=1.0, m=m)
                
                # 2. Backtracking line search
                alpha = getattr(options, 'alpha_bar', 1.0)
                c1 = getattr(options, 'c1', 1e-4)
                tau = getattr(options, 'tau', 0.5)
                
                while problem.compute_f(x + alpha * d) > f + c1 * alpha * np.dot(g, d):
                    alpha = tau * alpha
                    
                x_new = x + alpha * d
                f_new = problem.compute_f(x_new)
                g_new = problem.compute_g(x_new)
                
                # 3. Update Memory
                s_k = x_new - x
                y_k = g_new - g
                
                # skip update if s_k^T y_k is not sufficiently positive
                epsilon_sy = getattr(options, 'epsilon_sy', 1e-6)
                if np.dot(s_k, y_k) >= epsilon_sy * np.linalg.norm(s_k) * np.linalg.norm(y_k):
                    # if memory is full, remove the oldest elements
                    if len(s_list) == m:
                        s_list.pop(0)
                        y_list.pop(0)
                    # Append new curvature pairs
                    s_list.append(s_k)
                    y_list.append(y_k)

            case _:
                raise ValueError("method is not implemented yet")

        # update old and new function values
        x_old = x
        f_old = f
        g_old = g
        norm_g_old = norm_g
        x = x_new
        f = f_new
        g = g_new
        norm_g = np.linalg.norm(g, ord=np.inf)

        # increment iteration counter
        k = k + 1
    return x, f #return final iterate and final function value
