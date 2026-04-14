""" IOE 511/MATH 562, University of Michigan
Code written by: Albert S. Berahas & Jiahao Shi
"""
import numpy as np

def GDStep(x, f, g, problem, method, options): #KLH added tau as an input argument
    """Function that: (1) computes the GD step; (2) updates the iterate; and,
         (3) computes the function and gradient at the new iterate

    Inputs:
        x, f, g, problem, method, options
    Outputs:
        x_new, f_new, g_new, d, alpha
    """
    # Set the search direction d to be -g
    d = -g  # TODO: Add code (please replace None into its correct formula)
    # determine step size
    match method.options["step_type"]:
        case "Constant":
            alpha = method.options["constant_step_size"]
            x_new = x + alpha * d
            f_new = problem.compute_f(x_new)
            g_new = (
                problem.compute_g(x_new)  # TODO: Add code (please replace None into its correct formula)
            )

        case "Backtracking":
            # TODO: Add code!
            alpha = method.options["backtracking_alpha"]
            tau = 0.5 #KLH added according to HW2
            beta = method.options["backtracking_beta"]

            while problem.compute_f(x + alpha * d) > f + beta * alpha * np.dot(g, d): #g.T @ d
                alpha = tau * alpha
            x_new = x + alpha * d
            f_new = problem.compute_f(x_new)
            g_new = problem.compute_g(x_new)


        case _:
            raise ValueError("step type is not defined")

    return x_new, f_new, g_new, d, alpha
