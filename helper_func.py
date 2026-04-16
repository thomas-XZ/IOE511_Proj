# Here we put functions helping implementations in algorithms.py
import numpy as np


def weak_wolfe_line_search(x, f, g, d, problem, options):
    """
    Weak Wolfe line search based on pdf file on Wolfe line search
    """
    # [0] Inputs: (c1, c2) line search parameters
    c1 = getattr(options, "c1", 1e-4)
    c2 = getattr(options, "c2", 0.9)

    # [0] Inputs: (alpha, alpha_high, alpha_low, c) subroutine parameters
    alpha = getattr(options, "initial_alpha", 1.0)
    alpha_low = getattr(options, "alpha_low", 0.0)
    alpha_high = getattr(options, "alpha_high", 1000.0)
    c = getattr(options, "c", 0.5)

    max_ls_iter = getattr(options, "max_ls_iter", 100)
    g_dot_d = np.dot(g, d)

    # [1] While 1 (implemented with a safety max_iteration break)
    for _ in range(max_ls_iter):

        # [2] Evaluate f at point x_k + alpha * d_k
        x_new = x + alpha * d
        f_new = problem.compute_f(x_new)

        # Condition (3): f(x_k + alpha*d_k) <= f(x_k) + c1 * alpha * g^T d
        cond3_holds = f_new <= f + c1 * alpha * g_dot_d

        # [3] If (3) holds for alpha_k = alpha
        if cond3_holds:

            # [4] Evaluate nabla f at point x_k + alpha * d_k
            g_new = problem.compute_g(x_new)

            # Condition (4): nabla f(x_k + alpha*d_k)^T d_k >= c2 * nabla f(x_k)^T d_k
            cond4_holds = np.dot(g_new, d) >= c2 * g_dot_d

            # [5] If (4) holds for alpha_k = alpha
            if cond4_holds:
                # [6] Break
                break

        # [9] If (3) holds for alpha_k = alpha
        if cond3_holds:
            # [10]
            alpha_low = alpha
        # [11] Else
        else:
            # [12]
            alpha_high = alpha

        # [14] Set alpha = c * alpha_low + (1 - c) * alpha_high
        alpha = c * alpha_low + (1 - c) * alpha_high

    # Catch for the return if max_ls_iter is hit before condition 4 is evaluated
    if "g_new" not in locals():
        g_new = problem.compute_g(x_new)

    return alpha, x_new, f_new, g_new
