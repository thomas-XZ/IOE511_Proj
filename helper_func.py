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


def cg_steihaug(g, H, Delta, options):
    """
    Steihaug-Toint Conjugate Gradient method for solving the Trust Region subproblem.
    """
    n = len(g)
    z = np.zeros(n)
    r = g.copy()
    d = -r

    term_tol_CG = getattr(options, "term_tol_CG", 1e-5 * np.linalg.norm(g))
    max_iter_CG = getattr(options, "max_iterations_CG", n * 2)

    if np.linalg.norm(r) < term_tol_CG:
        return z

    for _ in range(max_iter_CG):
        Hd = H @ d
        kappa = np.dot(d, Hd)  # Curvature

        # negative curvature
        if kappa <= 0:
            a = np.dot(d, d)
            b = 2 * np.dot(z, d)
            c = np.dot(z, z) - Delta**2
            # Find root tau >= 0
            tau = (-b + np.sqrt(max(0, b**2 - 4 * a * c))) / (2 * a)
            return z + tau * d

        alpha = np.dot(r, r) / kappa
        z_next = z + alpha * d

        # exceed trust region
        if np.linalg.norm(z_next) >= Delta:
            a = np.dot(d, d)
            b = 2 * np.dot(z, d)
            c = np.dot(z, z) - Delta**2
            # Find root tau >= 0
            tau = (-b + np.sqrt(max(0, b**2 - 4 * a * c))) / (2 * a)
            return z + tau * d

        r_next = r + alpha * Hd

        # convergence test
        if np.linalg.norm(r_next) < term_tol_CG:
            return z_next

        beta = np.dot(r_next, r_next) / np.dot(r, r)
        d = -r_next + beta * d
        z = z_next
        r = r_next

    return z
