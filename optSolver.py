"""IOE 511/MATH 562, University of Michigan
Code template written by: Albert S. Berahas & Jiahao Shi
Implementations by: Kayla Huang, Wuhao Cao, & Thomas Xu ZHANG

Main solver entry point.  Call signature:
    x, f, info = optSolver_Fire_Horse(problem, method, options)

Returned `info` dict contains:
    iterations      – number of outer iterations completed
    f_evals         – total function evaluations (including line-search calls)
    g_evals         – total gradient evaluations
    H_evals         – total Hessian evaluations
    cpu_time        – wall-clock seconds
    converged       – True if ‖g‖∞ ≤ term_tol at termination
    term_reason     – string explaining why the run stopped
    f_history       – list of f values at each iterate (length = iterations+1)
    g_norm_history  – list of ‖g‖∞ values at each iterate
"""

import time
import numpy as np
import algorithms
import project_problems


# ---------------------------------------------------------------------------
# Helper: wrap a problem so that every call to compute_f/g/H is counted
# ---------------------------------------------------------------------------
class _CountedProblem:
    """Thin wrapper that counts oracle calls and forwards all attribute lookups."""

    def __init__(self, problem):
        self._prob   = problem
        self.f_count = 0
        self.g_count = 0
        self.H_count = 0
        # forward non-callable attributes directly
        self.name = problem.name
        self.x0   = problem.x0

    def compute_f(self, x):
        self.f_count += 1
        return self._prob.compute_f(x)

    def compute_g(self, x):
        self.g_count += 1
        return self._prob.compute_g(x)

    def compute_H(self, x):
        self.H_count += 1
        return self._prob.compute_H(x)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def optSolver_Fire_Horse(problem, method, options):
    """Run the selected optimization algorithm and return (x, f, info).

    Parameters
    ----------
    problem : SimpleNamespace
        Must provide .x0, .name, .compute_f, .compute_g, and (for Newton /
        trust-region methods) .compute_H.
    method  : SimpleNamespace
        Must provide .name (one of the 10 recognised algorithm names).
    options : SimpleNamespace
        Optional hyperparameters.  Missing keys fall back to defaults.

    Returns
    -------
    x    : ndarray  – final iterate
    f    : float    – function value at final iterate
    info : dict     – convergence diagnostics (see module docstring)
    """
    # --- Validate mandatory inputs ---
    if not hasattr(problem, 'x0') or problem.x0 is None:
        raise ValueError("problem.x0 (starting point) must be provided.")
    if not hasattr(problem, 'name') or problem.name is None:
        raise ValueError("problem.name must be provided.")
    if not hasattr(method, 'name') or method.name is None:
        raise ValueError("method.name must be provided.")

    # --- Wrap problem with call counters ---
    cp = _CountedProblem(problem)

    # --- Initialise iterates ---
    x      = cp.x0.copy().astype(float)
    f      = cp.compute_f(x)
    g      = cp.compute_g(x)
    norm_g = np.linalg.norm(g, ord=np.inf)

    n = len(x)
    k = 0  # iteration counter

    # --- Algorithm-specific state ---
    # Inverse Hessian approximation (quasi-Newton & TR-SR1)
    H = (np.eye(n)
         if method.name in ["BFGS", "BFGSW", "DFP", "DFPW", "TRSR1CG"]
         else None)

    x_old, g_old = None, None

    # Trust-region radius
    Delta = (getattr(options, "initial_radius", 1.0)
             if method.name in ["TRNewtonCG", "TRSR1CG"]
             else None)

    # --- Termination settings ---
    term_tol  = getattr(options, "term_tol", 1e-6)
    max_iter  = getattr(options, "max_iterations", 1000)

    # --- History tracking for convergence plots ---
    f_history      = [f]
    g_norm_history = [norm_g]

    # --- Start timer ---
    t_start = time.time()

    # -----------------------------------------------------------------------
    # Main optimisation loop
    # -----------------------------------------------------------------------
    term_reason = "max_iterations reached"

    while norm_g > term_tol and k < max_iter:
        try:
            if method.name == "GradientDescent":
                x_new, f_new, g_new, d, alpha = algorithms.GradientDescent(
                    x, f, g, cp, method, options)

            elif method.name == "GradientDescentW":
                x_new, f_new, g_new, d, alpha = algorithms.GradientDescentW(
                    x, f, g, cp, method, options)

            elif method.name == "Newton":
                x_new, f_new, g_new, d, alpha = algorithms.Newton(
                    x, f, g, cp, method, options)

            elif method.name == "NewtonW":
                x_new, f_new, g_new, d, alpha = algorithms.NewtonW(
                    x, f, g, cp, method, options)

            elif method.name == "TRNewtonCG":
                x_new, f_new, g_new, d, Delta = algorithms.TRNewtonCG(
                    x, f, g, Delta, cp, method, options)

            elif method.name == "TRSR1CG":
                x_new, f_new, g_new, H, d, Delta = algorithms.TRSR1CG(
                    x, x_old, f, g, g_old, H, Delta, k, cp, method, options)

            elif method.name == "BFGS":
                x_new, f_new, g_new, H, d, alpha = algorithms.BFGS(
                    x, x_old, f, g, g_old, H, k, cp, method, options)

            elif method.name == "BFGSW":
                x_new, f_new, g_new, H, d, alpha = algorithms.BFGSW(
                    x, x_old, f, g, g_old, H, k, cp, method, options)

            elif method.name == "DFP":
                x_new, f_new, g_new, H, d, alpha = algorithms.DFP(
                    x, x_old, f, g, g_old, H, k, cp, method, options)

            elif method.name == "DFPW":
                x_new, f_new, g_new, H, d, alpha = algorithms.DFPW(
                    x, x_old, f, g, g_old, H, k, cp, method, options)

            else:
                raise ValueError(f"Method '{method.name}' is not recognised.")

        except Exception as exc:
            # Catch numerical failures (singular matrices, overflow, etc.)
            term_reason = f"numerical failure: {exc}"
            break

        # --- Check for NaN / Inf (indicates numerical blow-up) ---
        if not np.isfinite(f_new) or not np.all(np.isfinite(g_new)):
            term_reason = "non-finite value encountered"
            break

        # --- Update state ---
        x_old, g_old = x, g
        x, f, g      = x_new, f_new, g_new
        norm_g       = np.linalg.norm(g, ord=np.inf)
        k           += 1

        # Record history
        f_history.append(f)
        g_norm_history.append(norm_g)

    # -----------------------------------------------------------------------
    # Finalise
    # -----------------------------------------------------------------------
    cpu_time  = time.time() - t_start
    converged = norm_g <= term_tol

    if converged:
        term_reason = "gradient tolerance satisfied"

    info = dict(
        iterations     = k,
        f_evals        = cp.f_count,
        g_evals        = cp.g_count,
        H_evals        = cp.H_count,
        cpu_time       = cpu_time,
        converged      = converged,
        term_reason    = term_reason,
        f_history      = f_history,
        g_norm_history = g_norm_history,
        final_grad_norm= norm_g,
    )

    return x, f, info
