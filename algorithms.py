"""IOE 511/MATH 562, University of Michigan
Code written by: Albert S. Berahas & Jiahao Shi
Extended and corrected by: Kayla Huang, Wuhao Cao, & Thomas Xu ZHANG

Implements the 10 optimisation algorithms:
  1.  GradientDescent   – steepest descent + Armijo backtracking
  2.  GradientDescentW  – steepest descent + weak-Wolfe line search
  3.  Newton            – modified Newton  + Armijo backtracking
  4.  NewtonW           – modified Newton  + weak-Wolfe line search
  5.  TRNewtonCG        – trust-region Newton with CG (Steihaug) sub-solver
  6.  TRSR1CG           – trust-region SR1  with CG (Steihaug) sub-solver
  7.  BFGS              – BFGS quasi-Newton + Armijo backtracking
  8.  BFGSW             – BFGS quasi-Newton + weak-Wolfe line search
  9.  DFP               – DFP  quasi-Newton + Armijo backtracking
  10. DFPW              – DFP  quasi-Newton + weak-Wolfe line search
"""

import numpy as np
import helper_func


# ---------------------------------------------------------------------------
# Internal helper: spectral Hessian modification for nonconvexity
# ---------------------------------------------------------------------------
def _modify_hessian(H, beta_min=1e-3):
    """Return H + shift*I where shift ensures all eigenvalues >= beta_min.

    This is the standard 'spectral modification' used in modified Newton
    methods to handle nonconvex regions: if the Hessian is not sufficiently
    positive definite, we add a scalar multiple of the identity to make it so.
    """
    eigvals = np.linalg.eigvalsh(H)  # eigvalsh is faster/stable for symmetric H
    min_eig = eigvals.min()
    if min_eig < beta_min:
        shift = abs(min_eig) + beta_min   # push smallest eigenvalue to beta_min
        return H + shift * np.eye(len(H))
    return H                              # H is already sufficiently PD


# ---------------------------------------------------------------------------
# Internal helper: safe Armijo backtracking line search
# ---------------------------------------------------------------------------
def _armijo_backtrack(x, f, g, d, problem, options):
    """Armijo (sufficient-decrease) backtracking line search.

    Returns the accepted step size alpha and the corresponding new point/value.
    A max-iteration guard prevents infinite loops when d is a near-flat direction.

    Parameters
    ----------
    c1  : Armijo constant   (default 1e-4)
    tau : reduction factor  (default 0.5)
    backtracking_alpha : initial step size (default 1.0)
    max_backtrack : max backtracking iterations (default 50)
    """
    alpha       = getattr(options, "backtracking_alpha", 1.0)
    tau         = getattr(options, "tau", 0.5)
    c1          = getattr(options, "c1", 1e-4)
    max_bt      = getattr(options, "max_backtrack", 50)

    # Safety: if d is not a descent direction, fall back to –g
    gd = np.dot(g, d)
    if gd >= 0:
        d  = -g
        gd = -np.dot(g, g)

    for _ in range(max_bt):
        if problem.compute_f(x + alpha * d) <= f + c1 * alpha * gd:
            break
        alpha *= tau

    x_new = x + alpha * d
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)
    return x_new, f_new, g_new, alpha


# ===========================================================================
# 1. Gradient Descent – Armijo backtracking
# ===========================================================================
def GradientDescent(x, f, g, problem, method, options):
    """Steepest descent step with Armijo backtracking line search.

    Direction: d = -g
    Step size: chosen by Armijo (sufficient-decrease) backtracking.
    Nonconvexity: not applicable (first-order method).
    """
    d = -g   # steepest-descent direction

    x_new, f_new, g_new, alpha = _armijo_backtrack(x, f, g, d, problem, options)
    return x_new, f_new, g_new, d, alpha


# ===========================================================================
# 2. Gradient Descent – weak-Wolfe line search
# ===========================================================================
def GradientDescentW(x, f, g, problem, method, options):
    """Steepest descent step with weak-Wolfe (sufficient-decrease + curvature) line search.

    Direction: d = -g
    Step size: weak-Wolfe line search (see helper_func.weak_wolfe_line_search).
    Nonconvexity: not applicable (first-order method).
    """
    d = -g   # steepest-descent direction

    alpha, x_new, f_new, g_new = helper_func.weak_wolfe_line_search(
        x, f, g, d, problem, options)
    return x_new, f_new, g_new, d, alpha


# ===========================================================================
# 3. Newton – Armijo backtracking  (modified to handle nonconvexity)
# ===========================================================================
def Newton(x, f, g, problem, method, options):
    """Modified Newton step with Armijo backtracking line search.

    Direction: d = -(H_mod)^{-1} g  where H_mod is the spectrally-shifted
               Hessian that is guaranteed positive definite.
    Nonconvexity: if H has negative/near-zero eigenvalues, we add shift*I
                  (spectral modification) to ensure positive definiteness.
    """
    H_raw = problem.compute_H(x)
    H_mod = _modify_hessian(H_raw)             # handle nonconvex regions
    d     = -np.linalg.solve(H_mod, g)         # Newton direction

    x_new, f_new, g_new, alpha = _armijo_backtrack(x, f, g, d, problem, options)
    return x_new, f_new, g_new, d, alpha


# ===========================================================================
# 4. Newton – weak-Wolfe line search  (modified for nonconvexity)
# ===========================================================================
def NewtonW(x, f, g, problem, method, options):
    """Modified Newton step with weak-Wolfe line search.

    Direction: d = -(H_mod)^{-1} g  (same modification as Newton above).
    Step size: weak-Wolfe line search.
    """
    H_raw = problem.compute_H(x)
    H_mod = _modify_hessian(H_raw)             # handle nonconvex regions
    d     = -np.linalg.solve(H_mod, g)         # Newton direction

    alpha, x_new, f_new, g_new = helper_func.weak_wolfe_line_search(
        x, f, g, d, problem, options)
    return x_new, f_new, g_new, d, alpha


# ===========================================================================
# 5. Trust-Region Newton with CG (Steihaug) sub-solver
# ===========================================================================
def TRNewtonCG(x, f, g, Delta, problem, method, options):
    """Trust-region Newton method.  The subproblem min m_k(p) s.t. ‖p‖ ≤ Δ
    is solved approximately with the Steihaug-Toint CG method.

    Nonconvexity: handled implicitly by CG Steihaug – if negative curvature
    is detected, the algorithm moves to the trust-region boundary along the
    direction of negative curvature, ensuring descent even in nonconvex regions.

    Trust-region radius update:
        ρ < ρ_c1  → shrink Δ by γ1
        ρ > ρ_c2 and step hits boundary → expand Δ by γ2 (capped at max_Delta)
    """
    H_raw = problem.compute_H(x)
    # CG Steihaug solves the TR subproblem using the raw (possibly indefinite) Hessian
    p = helper_func.cg_steihaug(g, H_raw, Delta, options)

    # Evaluate trial point
    x_trial = x + p
    f_trial = problem.compute_f(x_trial)

    # Actual vs predicted reduction (ρ)
    actual_red    = f - f_trial
    predicted_red = -(np.dot(g, p) + 0.5 * p @ H_raw @ p)
    rho = actual_red / predicted_red if predicted_red > 1e-12 else 0.0

    # Trust-region radius update parameters
    rho_c1    = getattr(options, "rho_c1",    0.25)
    rho_c2    = getattr(options, "rho_c2",    0.75)
    gamma1    = getattr(options, "gamma1",    0.5)
    gamma2    = getattr(options, "gamma2",    2.0)
    max_Delta = getattr(options, "max_Delta", 10.0)

    if rho < rho_c1:
        Delta = gamma1 * Delta                                   # bad step: shrink
    elif rho > rho_c2 and abs(np.linalg.norm(p) - Delta) < 1e-8:
        Delta = min(gamma2 * Delta, max_Delta)                   # great step: expand

    # Step acceptance (accept if ρ > η0, typically 0)
    eta0 = getattr(options, "eta0", 0.0)
    if rho > eta0:
        x_new = x_trial
        f_new = f_trial
        g_new = problem.compute_g(x_new)
    else:
        x_new, f_new, g_new = x, f, g                           # reject step

    return x_new, f_new, g_new, p, Delta


# ===========================================================================
# 6. Trust-Region SR1 quasi-Newton with CG (Steihaug) sub-solver
# ===========================================================================
def TRSR1CG(x, x_old, f, g, g_old, H, Delta, k, problem, method, options):
    """Trust-region method with SR1 Hessian approximation.

    The SR1 update maintains a symmetric (not necessarily PD) Hessian
    approximation.  The Steihaug CG sub-solver handles indefiniteness.

    SR1 update: H ← H + (y-Hs)(y-Hs)^T / (y-Hs)^T s
    Safeguard: update skipped when |(y-Hs)^T s| < 1e-8 ‖s‖‖y-Hs‖.
    """
    if k == 0:
        H = np.eye(len(x))   # initialise as identity at iteration 0
    else:
        s     = x - x_old
        y     = g - g_old
        Hs    = H @ s
        diff  = y - Hs
        denom = np.dot(diff, s)
        norm_diff = np.linalg.norm(diff)
        # SR1 safeguard: skip update if denominator is too small or diff is near-zero.
        # Both conditions guard against division by near-zero and nan propagation.
        if (norm_diff > 1e-10
                and abs(denom) >= 1e-8 * np.linalg.norm(s) * norm_diff
                and np.isfinite(denom)):
            H = H + np.outer(diff, diff) / denom

    # Solve TR subproblem with Steihaug CG using SR1 Hessian approx
    d = helper_func.cg_steihaug(g, H, Delta, options)

    x_trial = x + d
    f_trial = problem.compute_f(x_trial)

    actual_red    = f - f_trial
    predicted_red = -(np.dot(g, d) + 0.5 * d @ H @ d)
    rho = actual_red / predicted_red if predicted_red > 1e-12 else 0.0

    c1        = getattr(options, "TR_c1",     0.25)
    c2        = getattr(options, "TR_c2",     0.75)
    max_Delta = getattr(options, "max_Delta", 10.0)

    if rho > c1:
        x_new = x_trial
        f_new = f_trial
        g_new = problem.compute_g(x_new)
        if rho > c2:
            Delta = min(2.0 * Delta, max_Delta)   # expand radius
        # else keep radius unchanged
    else:
        x_new, f_new, g_new = x, f, g            # reject step
        Delta = 0.5 * Delta                        # shrink radius

    return x_new, f_new, g_new, H, d, Delta


# ===========================================================================
# 7. BFGS quasi-Newton – Armijo backtracking
# ===========================================================================
def BFGS(x, x_old, f, g, g_old, H, k, problem, method, options):
    """BFGS quasi-Newton with Armijo backtracking line search.

    H stores the *inverse* Hessian approximation (B^{-1}).
    BFGS update (Sherman-Morrison-Woodbury form):
        H ← (I - ρ s y^T) H (I - ρ y s^T) + ρ s s^T
    where ρ = 1/(y^T s).  The curvature condition y^T s > 0 is required; if
    violated the update is skipped (H remains unchanged).

    Nonconvexity: handled indirectly – the BFGS inverse Hessian approximation
    remains positive definite as long as the curvature condition holds.
    The Armijo backtracking then guarantees sufficient decrease.
    """
    if k == 0:
        H_new = np.eye(len(x))   # start with identity inverse Hessian
    else:
        s = x - x_old
        y = g - g_old
        ys = np.dot(y, s)
        if ys > 1e-10:           # curvature condition check
            rho   = 1.0 / ys
            I     = np.eye(len(x))
            A     = I - rho * np.outer(s, y)
            H_new = A @ H @ A.T + rho * np.outer(s, s)
        else:
            H_new = H            # skip update; keep current approximation

    d = -H_new @ g               # quasi-Newton search direction

    x_new, f_new, g_new, alpha = _armijo_backtrack(x, f, g, d, problem, options)
    return x_new, f_new, g_new, H_new, d, alpha


# ===========================================================================
# 8. BFGS quasi-Newton – weak-Wolfe line search
# ===========================================================================
def BFGSW(x, x_old, f, g, g_old, H, k, problem, method, options):
    """BFGS quasi-Newton with weak-Wolfe line search.

    Same BFGS inverse-Hessian update as BFGS above.  Using the Wolfe
    conditions (specifically the curvature condition) guarantees y^T s > 0
    at the next iterate, keeping the BFGS update well-defined.
    """
    if k == 0:
        H_new = np.eye(len(x))
    else:
        s = x - x_old
        y = g - g_old
        ys = np.dot(y, s)
        if ys > 1e-10:
            rho   = 1.0 / ys
            I     = np.eye(len(x))
            A     = I - rho * np.outer(s, y)
            H_new = A @ H @ A.T + rho * np.outer(s, s)
        else:
            H_new = H

    d = -H_new @ g

    alpha, x_new, f_new, g_new = helper_func.weak_wolfe_line_search(
        x, f, g, d, problem, options)
    return x_new, f_new, g_new, H_new, d, alpha


# ===========================================================================
# 9. DFP quasi-Newton – Armijo backtracking
# ===========================================================================
def DFP(x, x_old, f, g, g_old, H, k, problem, method, options):
    """DFP quasi-Newton with Armijo backtracking line search.

    H stores the *inverse* Hessian approximation.
    DFP update (for the inverse Hessian):
        H ← H - (H y y^T H)/(y^T H y) + (s s^T)/(y^T s)
    As with BFGS, the update is skipped when y^T s ≤ 0.
    """
    if k == 0:
        H_new = np.eye(len(x))
    else:
        s  = x - x_old
        y  = g - g_old
        ys = np.dot(y, s)
        if ys > 1e-10:
            Hy  = H @ y
            yHy = np.dot(y, Hy)
            H_new = H - np.outer(Hy, Hy) / yHy + np.outer(s, s) / ys
        else:
            H_new = H

    d = -H_new @ g

    x_new, f_new, g_new, alpha = _armijo_backtrack(x, f, g, d, problem, options)
    return x_new, f_new, g_new, H_new, d, alpha


# ===========================================================================
# 10. DFP quasi-Newton – weak-Wolfe line search
# ===========================================================================
def DFPW(x, x_old, f, g, g_old, H, k, problem, method, options):
    """DFP quasi-Newton with weak-Wolfe line search.

    Same DFP inverse-Hessian update as DFP above.
    Using the Wolfe conditions guarantees y^T s > 0 at the next iterate.
    """
    if k == 0:
        H_new = np.eye(len(x))
    else:
        s  = x - x_old
        y  = g - g_old
        ys = np.dot(y, s)
        if ys > 1e-10:
            Hy  = H @ y
            yHy = np.dot(y, Hy)
            H_new = H - np.outer(Hy, Hy) / yHy + np.outer(s, s) / ys
        else:
            H_new = H

    d = -H_new @ g

    alpha, x_new, f_new, g_new = helper_func.weak_wolfe_line_search(
        x, f, g, d, problem, options)
    return x_new, f_new, g_new, H_new, d, alpha
