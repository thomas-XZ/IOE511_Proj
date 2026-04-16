"""IOE 511/MATH 562, University of Michigan
Code written by: Albert S. Berahas & Jiahao Shi
"""

import numpy as np
import helper_func


def GradientDescent(x, f, g, problem, method, options):
    d = -g

    alpha = getattr(options, "backtracking_alpha", 1.0)
    tau = getattr(options, "tau", 0.5)
    c1 = getattr(options, "c1", 1e-4)

    while problem.compute_f(x + alpha * d) > f + c1 * alpha * np.dot(g, d):
        alpha = tau * alpha

    x_new = x + alpha * d
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    return x_new, f_new, g_new, d, alpha


def GradientDescentW(x, f, g, problem, method, options):
    d = -g

    # weak Wolfe line search helper function
    alpha, x_new, f_new, g_new = helper_func.weak_wolfe_line_search(
        x, f, g, d, problem, options
    )

    return x_new, f_new, g_new, d, alpha


def Newton(x, f, g, problem, method, options):
    H = problem.compute_H(x)
    # TODO: Add modification for nonconvexity if H is not positive definite
    d = -np.linalg.solve(H, g)

    alpha = getattr(options, "backtracking_alpha", 1.0)
    tau = getattr(options, "tau", 0.5)
    c1 = getattr(options, "c1", 1e-4)

    while problem.compute_f(x + alpha * d) > f + c1 * alpha * np.dot(g, d):
        alpha = tau * alpha

    x_new = x + alpha * d
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    return x_new, f_new, g_new, d, alpha


def NewtonW(x, f, g, problem, method, options):
    H = problem.compute_H(x)  # Hessian
    d = -np.linalg.solve(H, g)

    # weak Wolfe line search
    alpha, x_new, f_new, g_new = helper_func.weak_wolfe_line_search(
        x, f, g, d, problem, options
    )

    return x_new, f_new, g_new, d, alpha


def TRNewtonCG(x, f, g, Delta, problem, method, options):
    H = problem.compute_H(x)

    # CG Steihaug
    p = helper_func.cg_steihaug(g, H, Delta, options)

    # move trial step
    x_trial = x + p
    f_trial = problem.compute_f(x_trial)

    # rho
    actual_reduction = f - f_trial
    predicted_reduction = -(np.dot(g, p) + 0.5 * np.dot(p, H @ p))

    if predicted_reduction > 1e-12:
        rho = actual_reduction / predicted_reduction
    else:
        rho = 0

    # TR radius
    rho_c1 = getattr(options, "rho_c1", 0.25)
    rho_c2 = getattr(options, "rho_c2", 0.75)
    gamma1 = getattr(options, "gamma1", 0.5)
    gamma2 = getattr(options, "gamma2", 2.0)
    max_Delta = getattr(options, "max_Delta", 10.0)

    if rho < rho_c1:
        # bad step
        Delta = gamma1 * Delta
    elif rho > rho_c2 and np.abs(np.linalg.norm(p) - Delta) < 1e-8:
        # good step
        Delta = min(gamma2 * Delta, max_Delta)

    # step acceptance test
    eta0 = getattr(options, "eta0", 0.0)
    if rho > eta0:
        # Accept
        x_new = x_trial
        f_new = f_trial
        g_new = problem.compute_g(x_new)
    else:
        # Reject: stay at current point, try again next iteration with smaller Delta
        x_new = x
        f_new = f
        g_new = g

    return x_new, f_new, g_new, p, Delta


def TRSR1CG(x, x_old, f, g, g_old, H, Delta, k, problem, method, options):
    # SR1 Hessian approx
    if k == 0:
        H = np.eye(len(x))
    else:
        s = x - x_old
        y = g - g_old
        Hs = H @ s
        diff = y - Hs

        # SR1 denominator safeguard
        denom = np.dot(diff, s)
        if np.abs(denom) >= 1e-8 * np.linalg.norm(s) * np.linalg.norm(diff):
            H = H + np.outer(diff, diff) / denom

    # CG Steihaug
    d = helper_func.cg_steihaug(g, H, Delta, options)

    # trail step
    x_trial = x + d
    f_trial = problem.compute_f(x_trial)

    # rho
    actual_reduction = f - f_trial
    predicted_reduction = -(np.dot(g, d) + 0.5 * np.dot(d, H @ d))

    if predicted_reduction > 1e-12:
        rho = actual_reduction / predicted_reduction
    else:
        rho = 0.0

    # get parameters
    c1 = getattr(options, "TR_c1", 0.25)
    c2 = getattr(options, "TR_c2", 0.75)
    max_Delta = getattr(options, "max_Delta", 10.0)

    # Step test
    if rho > c1:
        # Accept step
        x_new = x_trial
        f_new = f_trial
        g_new = problem.compute_g(x_new)

        if rho > c2:
            # Expand radius
            Delta = min(2.0 * Delta, max_Delta)
        else:
            # Maintain radius
            pass

    else:
        # Reject step and shrink radius
        x_new = x
        f_new = f
        g_new = g
        Delta = 0.5 * Delta

    return x_new, f_new, g_new, H, d, Delta


def BFGS(x, x_old, f, g, g_old, H, k, problem, method, options):
    if k == 0:
        H_new = np.eye(len(x))
    else:
        s = x - x_old
        y = g - g_old
        if np.dot(y, s) > 1e-10:
            rho = 1.0 / np.dot(y, s)
            I = np.eye(len(x))
            H_new = (I - rho * np.outer(s, y)) @ H @ (
                I - rho * np.outer(y, s)
            ) + rho * np.outer(s, s)
        else:
            H_new = H

    d = -H_new @ g

    alpha = getattr(options, "backtracking_alpha", 1.0)
    tau = getattr(options, "tau", 0.5)
    c1 = getattr(options, "c1", 1e-4)

    while problem.compute_f(x + alpha * d) > f + c1 * alpha * np.dot(g, d):
        alpha = tau * alpha

    x_new = x + alpha * d
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    return x_new, f_new, g_new, H_new, d, alpha


def BFGSW(x, x_old, f, g, g_old, H, k, problem, method, options):
    # Hessian approximaton
    if k == 0:
        H_new = np.eye(len(x))
    else:
        s = x - x_old
        y = g - g_old
        if np.dot(y, s) > 1e-10:
            rho = 1.0 / np.dot(y, s)
            I = np.eye(len(x))
            H_new = (I - rho * np.outer(s, y)) @ H @ (
                I - rho * np.outer(y, s)
            ) + rho * np.outer(s, s)
        else:
            H_new = H

    # search direction
    d = -H_new @ g

    # weak Wolfe line search
    alpha, x_new, f_new, g_new = helper_func.weak_wolfe_line_search(
        x, f, g, d, problem, options
    )

    return x_new, f_new, g_new, H_new, d, alpha


def DFP(x, x_old, f, g, g_old, H, k, problem, method, options):
    # Hessian approximation
    if k == 0:
        H_new = np.eye(len(x))
    else:
        s = x - x_old
        y = g - g_old

        # Check curvature condition
        ys = np.dot(y, s)
        if ys > 1e-10:
            Hy = H @ y
            yHy = np.dot(y, Hy)

            # DFP update formula
            term1 = np.outer(Hy, Hy) / yHy
            term2 = np.outer(s, s) / ys
            H_new = H - term1 + term2
        else:
            H_new = H

    # search direction
    d = -H_new @ g

    # backtracking line search
    alpha = getattr(options, "backtracking_alpha", 1.0)
    tau = getattr(options, "tau", 0.5)
    c1 = getattr(options, "c1", 1e-4)

    while problem.compute_f(x + alpha * d) > f + c1 * alpha * np.dot(g, d):
        alpha = tau * alpha

    x_new = x + alpha * d
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    return x_new, f_new, g_new, H_new, d, alpha


def DFPW(x, x_old, f, g, g_old, H, k, problem, method, options):
    # Hessian approximation
    if k == 0:
        H_new = np.eye(len(x))
    else:
        s = x - x_old
        y = g - g_old

        # Check curvature condition
        ys = np.dot(y, s)
        if ys > 1e-10:
            Hy = H @ y
            yHy = np.dot(y, Hy)

            # DFP update formula
            term1 = np.outer(Hy, Hy) / yHy
            term2 = np.outer(s, s) / ys
            H_new = H - term1 + term2
        else:
            H_new = H

    # search direction
    d = -H_new @ g

    # weak Wolfe line search
    alpha, x_new, f_new, g_new = helper_func.weak_wolfe_line_search(
        x, f, g, d, problem, options
    )

    return x_new, f_new, g_new, H_new, d, alpha
