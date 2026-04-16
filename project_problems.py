# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi
# Bugs fixed and Q-matrix generation added by: Kayla Huang, Wuhao Cao, & Thomas Xu ZHANG
#
# Defines all 12 test problems: function value, gradient, and Hessian.
# Problems (1)-(4): Quadratic;  (5)-(6): Quartic;  (7)-(8): Rosenbrock;
# (9): DataFit;  (10)-(11): Exponential;  (12): Genhumps.
#
# NOTE: Q matrices for quadratic problems are generated in-code (no .mat files needed).
#       A module-level cache avoids redundant computation across repeated calls.

import numpy as np

# ---------------------------------------------------------------------------
# Internal helper: generate a symmetric PD matrix with given condition number
# ---------------------------------------------------------------------------
_Q_cache = {}

def _get_Q(n, kappa, seed):
    """Return a cached n×n SPD matrix with condition number `kappa`.

    Uses a random orthogonal basis (QR of a Gaussian matrix) with eigenvalues
    log-spaced from 1 to kappa.  `seed` makes the result reproducible.
    """
    key = (n, kappa, seed)
    if key not in _Q_cache:
        rng = np.random.RandomState(seed)
        A = rng.randn(n, n)
        U, _ = np.linalg.qr(A)                        # random orthogonal matrix
        eigvals = np.logspace(0, np.log10(kappa), n)  # log-spaced eigenvalues
        Q = U @ np.diag(eigvals) @ U.T
        _Q_cache[key] = (Q + Q.T) / 2                 # enforce exact symmetry
    return _Q_cache[key]

def _get_q(n, seed):
    """Return a cached n-dimensional random vector `q`."""
    key = ('q', n, seed)
    if key not in _Q_cache:
        _Q_cache[key] = np.random.RandomState(seed).randn(n)
    return _Q_cache[key]

# Pre-compute starting points (MATLAB: rng(0); x0=20*rand(n,1)-10)
_x0_10   = 20 * np.random.RandomState(0).rand(10)   - 10  # shape (10,)
_x0_1000 = 20 * np.random.RandomState(0).rand(1000) - 10  # shape (1000,)


# ===========================================================================
# Problem 1  –  P1_quad_10_10
# Convex quadratic: f(x) = 0.5 x^T Q x + q^T x
# n=10, condition number κ=10
# ===========================================================================

_Q1 = _get_Q(10, 10, seed=11)
_q1 = _get_q(10, seed=12)

def quad_10_10_func(x):
    """Function value of the n=10, κ=10 quadratic."""
    return float(0.5 * x @ _Q1 @ x + _q1 @ x)

def quad_10_10_grad(x):
    """Gradient of the n=10, κ=10 quadratic: Q x + q."""
    return _Q1 @ x + _q1

def quad_10_10_Hess(x):
    """Hessian of the n=10, κ=10 quadratic: Q (constant)."""
    return _Q1


# ===========================================================================
# Problem 2  –  P2_quad_10_1000
# Convex quadratic: f(x) = 0.5 x^T Q x + q^T x
# n=10, condition number κ=1000
# ===========================================================================

_Q2 = _get_Q(10, 1000, seed=21)
_q2 = _get_q(10, seed=22)

def quad_10_1000_func(x):
    """Function value of the n=10, κ=1000 quadratic."""
    return float(0.5 * x @ _Q2 @ x + _q2 @ x)

def quad_10_1000_grad(x):
    """Gradient of the n=10, κ=1000 quadratic: Q x + q."""
    return _Q2 @ x + _q2

def quad_10_1000_Hess(x):
    """Hessian of the n=10, κ=1000 quadratic: Q (constant)."""
    return _Q2


# ===========================================================================
# Problem 3  –  P3_quad_1000_10
# Convex quadratic: f(x) = 0.5 x^T Q x + q^T x
# n=1000, condition number κ=10
# ===========================================================================

_Q3 = _get_Q(1000, 10, seed=31)
_q3 = _get_q(1000, seed=32)

def quad_1000_10_func(x):
    """Function value of the n=1000, κ=10 quadratic."""
    return float(0.5 * x @ _Q3 @ x + _q3 @ x)

def quad_1000_10_grad(x):
    """Gradient of the n=1000, κ=10 quadratic: Q x + q."""
    return _Q3 @ x + _q3

def quad_1000_10_Hess(x):
    """Hessian of the n=1000, κ=10 quadratic: Q (constant)."""
    return _Q3


# ===========================================================================
# Problem 4  –  P4_quad_1000_1000
# Convex quadratic: f(x) = 0.5 x^T Q x + q^T x
# n=1000, condition number κ=1000
# ===========================================================================

_Q4 = _get_Q(1000, 1000, seed=41)
_q4 = _get_q(1000, seed=42)

def quad_1000_1000_func(x):
    """Function value of the n=1000, κ=1000 quadratic."""
    return float(0.5 * x @ _Q4 @ x + _q4 @ x)

def quad_1000_1000_grad(x):
    """Gradient of the n=1000, κ=1000 quadratic: Q x + q."""
    return _Q4 @ x + _q4

def quad_1000_1000_Hess(x):
    """Hessian of the n=1000, κ=1000 quadratic: Q (constant)."""
    return _Q4


# ===========================================================================
# Problems 5 & 6  –  quartic functions
# f(x) = 0.5 x^T x + (σ/4)(x^T Q x)^2,  n=4
# P5: σ=1e-4 (mild quartic term);  P6: σ=1e4 (dominant quartic term)
# ===========================================================================

_Q_quartic = np.array([[5,   1,   0,   0.5],
                        [1,   4,   0.5, 0  ],
                        [0,   0.5, 3,   0  ],
                        [0.5, 0,   0,   2  ]])

def quartic_1_func(x):
    """f(x) = 0.5 x^T x + (1e-4/4)(x^T Q x)^2."""
    sigma = 1e-4
    return 0.5 * (x @ x) + sigma / 4 * (x @ _Q_quartic @ x) ** 2

def quartic_1_grad(x):
    """Gradient of quartic_1: x + σ (x^T Q x) Q x."""
    sigma = 1e-4
    Qx = _Q_quartic @ x
    return x + sigma * (x @ Qx) * Qx

def quartic_1_Hess(x):
    """Hessian of quartic_1: I + σ(x^T Q x)Q + 2σ (Qx)(Qx)^T."""
    sigma = 1e-4
    Qx = _Q_quartic @ x
    xQx = x @ Qx
    return np.eye(4) + sigma * xQx * _Q_quartic + 2 * sigma * np.outer(Qx, Qx)

def quartic_2_func(x):
    """f(x) = 0.5 x^T x + (1e4/4)(x^T Q x)^2."""
    sigma = 1e4
    return 0.5 * (x @ x) + sigma / 4 * (x @ _Q_quartic @ x) ** 2

def quartic_2_grad(x):
    """Gradient of quartic_2: x + σ (x^T Q x) Q x."""
    sigma = 1e4
    Qx = _Q_quartic @ x
    return x + sigma * (x @ Qx) * Qx

def quartic_2_Hess(x):
    """Hessian of quartic_2: I + σ(x^T Q x)Q + 2σ (Qx)(Qx)^T."""
    sigma = 1e4
    Qx = _Q_quartic @ x
    xQx = x @ Qx
    return np.eye(4) + sigma * xQx * _Q_quartic + 2 * sigma * np.outer(Qx, Qx)


# ===========================================================================
# Problems 7 & 8  –  Rosenbrock
# P7: n=2,   f(x) = (1-x1)^2 + 100(x2-x1^2)^2
# P8: n=100, f(x) = Σ_{i=1}^{99} [(1-xi)^2 + 100(x_{i+1}-xi^2)^2]
# ===========================================================================

def rosenbrock_2_func(x):
    """Rosenbrock function, n=2."""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

def rosenbrock_2_grad(x):
    """Gradient of Rosenbrock, n=2."""
    return np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
        200 * (x[1] - x[0] ** 2)
    ])

def rosenbrock_2_Hess(x):
    """Hessian of Rosenbrock, n=2."""
    return np.array([
        [2 - 400 * (x[1] - x[0] ** 2) + 800 * x[0] ** 2, -400 * x[0]],
        [-400 * x[0],                                        200        ]
    ])


def rosenbrock_100_func(x):
    """Extended Rosenbrock function, n=100."""
    return sum((1 - x[i]) ** 2 + 100 * (x[i + 1] - x[i] ** 2) ** 2
               for i in range(len(x) - 1))

def rosenbrock_100_grad(x):
    """Gradient of extended Rosenbrock, n=100."""
    n = len(x)
    g = np.zeros(n)
    # First component
    g[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    # Interior components
    for i in range(1, n - 1):
        g[i] = (200 * (x[i] - x[i - 1] ** 2)
                - 2 * (1 - x[i])
                - 400 * x[i] * (x[i + 1] - x[i] ** 2))
    # Last component
    g[-1] = 200 * (x[-1] - x[-2] ** 2)
    return g

def rosenbrock_100_Hess(x):
    """Hessian of extended Rosenbrock, n=100."""
    n = len(x)
    H = np.zeros((n, n))
    H[0, 0] = 2 + 1200 * x[0] ** 2 - 400 * x[1]
    H[0, 1] = -400 * x[0]
    for i in range(1, n - 1):
        H[i, i - 1] = -400 * x[i - 1]
        H[i, i]     = 202 + 1200 * x[i] ** 2 - 400 * x[i + 1]
        H[i, i + 1] = -400 * x[i]
    H[-1, -2] = -400 * x[-2]
    H[-1, -1] = 200
    return H


# ===========================================================================
# Problem 9  –  DataFit_2
# f(x) = Σ_{i=1}^{3} (y_i - x1*(1 - x2^i))^2,   n=2
# y = [1.5, 2.25, 2.625]^T,  x0 = [1, 1]^T
# ===========================================================================

_y_data = np.array([1.5, 2.25, 2.625])

def data_fit_2_func(x):
    """Data-fitting function, n=2.  Returns scalar."""
    residuals = _y_data - x[0] * (1 - x[1] ** np.arange(1, 4))
    return float(np.sum(residuals ** 2))

def data_fit_2_grad(x):
    """Gradient of data_fit_2 using chain rule."""
    g = np.zeros(2)
    for i in range(1, 4):
        ri = _y_data[i - 1] - x[0] * (1 - x[1] ** i)
        g[0] += 2 * ri * (-(1 - x[1] ** i))                         # ∂ri/∂x1 = -(1-x2^i)
        g[1] += 2 * ri * (x[0] * i * x[1] ** (i - 1))               # ∂ri/∂x2 = x1*i*x2^{i-1}
    return g

def data_fit_2_Hess(x):
    """Full Hessian of data_fit_2 (second-order terms included)."""
    H = np.zeros((2, 2))
    for i in range(1, 4):
        ri        = _y_data[i - 1] - x[0] * (1 - x[1] ** i)
        dri_dx1   = -(1 - x[1] ** i)
        dri_dx2   = x[0] * i * x[1] ** (i - 1)
        # Second cross-derivative of ri: ∂²ri/(∂x1∂x2) = i*x2^{i-1}
        d2ri_x1x2 = i * x[1] ** (i - 1)
        # Second derivative of ri w.r.t. x2: x1*i*(i-1)*x2^{i-2}
        d2ri_x2x2 = x[0] * i * (i - 1) * x[1] ** max(i - 2, 0) if i >= 2 else 0.0

        H[0, 0] += 2 * dri_dx1 ** 2                                  # J^T J term
        H[0, 1] += 2 * (dri_dx1 * dri_dx2 + ri * d2ri_x1x2)         # full cross term
        H[1, 1] += 2 * (dri_dx2 ** 2 + ri * d2ri_x2x2)              # full diagonal term
    H[1, 0] = H[0, 1]
    return H


# ===========================================================================
# Problems 10 & 11  –  Exponential functions
# f(x) = (e^{x1}-1)/(e^{x1}+1) + 0.1*e^{-x1} + Σ_{i=2}^{n}(xi-1)^4
# P10: n=10;  P11: n=100
# Starting point: x0 = [1, 0, …, 0]^T
# ===========================================================================

def _exp_func(x):
    """Shared exponential function body."""
    ex = np.exp(x[0])
    return float((ex - 1) / (ex + 1) + 0.1 * np.exp(-x[0]) + np.sum((x[1:] - 1) ** 4))

def _exp_grad(x):
    """Shared exponential gradient body.

    df/dx1 = 2*e^{x1}/(e^{x1}+1)^2 - 0.1*e^{-x1}
    df/dxi = 4*(xi-1)^3  for i >= 2
    """
    ex = np.exp(x[0])
    n = len(x)
    g = np.zeros(n)
    g[0] = 2 * ex / (ex + 1) ** 2 - 0.1 * np.exp(-x[0])  # fixed formula
    g[1:] = 4 * (x[1:] - 1) ** 3
    return g

def _exp_hess(x):
    """Shared exponential Hessian body.

    d²f/dx1² = 2*e^{x1}*(1-e^{x1})/(e^{x1}+1)^3 + 0.1*e^{-x1}
    d²f/dxi² = 12*(xi-1)^2  for i >= 2
    """
    ex = np.exp(x[0])
    n = len(x)
    H = np.zeros((n, n))
    H[0, 0] = 2 * ex * (1 - ex) / (ex + 1) ** 3 + 0.1 * np.exp(-x[0])
    diag_idx = np.arange(1, n)
    H[diag_idx, diag_idx] = 12 * (x[1:] - 1) ** 2
    return H

# --- P10: n=10 ---
def exp_10_func(x):
    """Exponential function, n=10.  Returns scalar."""
    return _exp_func(x)

def exp_10_grad(x):
    """Gradient of exponential function, n=10."""
    return _exp_grad(x)

def exp_10_Hess(x):
    """Hessian of exponential function, n=10."""
    return _exp_hess(x)

# --- P11: n=100 ---
def exp_100_func(x):
    """Exponential function, n=100.  Returns scalar."""
    return _exp_func(x)

def exp_100_grad(x):
    """Gradient of exponential function, n=100."""
    return _exp_grad(x)

def exp_100_Hess(x):
    """Hessian of exponential function, n=100."""
    return _exp_hess(x)


# ===========================================================================
# Problem 12  –  Genhumps_5
# f(x) = Σ_{i=1}^{4} [sin²(2xi)*sin²(2x_{i+1}) + 0.05(xi² + x_{i+1}²)]
# n=5;  Starting point: x0 = [-506.2, 506.2, …, 506.2]^T
# ===========================================================================

def genhumps_5_func(x):
    """Genhumps function, n=5."""
    f = 0.0
    for i in range(4):
        f += (np.sin(2 * x[i]) ** 2 * np.sin(2 * x[i + 1]) ** 2
              + 0.05 * (x[i] ** 2 + x[i + 1] ** 2))
    return f

def genhumps_5_grad(x):
    """Gradient of Genhumps, n=5."""
    g = np.zeros(5)
    g[0] = (4 * np.sin(2*x[0]) * np.cos(2*x[0]) * np.sin(2*x[1])**2
            + 0.1 * x[0])
    g[1] = (4 * np.sin(2*x[1]) * np.cos(2*x[1]) * (np.sin(2*x[0])**2 + np.sin(2*x[2])**2)
            + 0.2 * x[1])
    g[2] = (4 * np.sin(2*x[2]) * np.cos(2*x[2]) * (np.sin(2*x[1])**2 + np.sin(2*x[3])**2)
            + 0.2 * x[2])
    g[3] = (4 * np.sin(2*x[3]) * np.cos(2*x[3]) * (np.sin(2*x[2])**2 + np.sin(2*x[4])**2)
            + 0.2 * x[3])
    g[4] = (4 * np.sin(2*x[4]) * np.cos(2*x[4]) * np.sin(2*x[3])**2
            + 0.1 * x[4])
    return np.array(g)

def genhumps_5_Hess(x):
    """Hessian of Genhumps, n=5."""
    H = np.zeros((5, 5))
    H[0, 0] = 8 * np.sin(2*x[1])**2 * (np.cos(2*x[0])**2 - np.sin(2*x[0])**2) + 0.1
    H[0, 1] = 16 * np.sin(2*x[0]) * np.cos(2*x[0]) * np.sin(2*x[1]) * np.cos(2*x[1])
    H[1, 1] = 8 * (np.sin(2*x[0])**2 + np.sin(2*x[2])**2) * (np.cos(2*x[1])**2 - np.sin(2*x[1])**2) + 0.2
    H[1, 2] = 16 * np.sin(2*x[1]) * np.cos(2*x[1]) * np.sin(2*x[2]) * np.cos(2*x[2])
    H[2, 2] = 8 * (np.sin(2*x[1])**2 + np.sin(2*x[3])**2) * (np.cos(2*x[2])**2 - np.sin(2*x[2])**2) + 0.2
    H[2, 3] = 16 * np.sin(2*x[2]) * np.cos(2*x[2]) * np.sin(2*x[3]) * np.cos(2*x[3])
    H[3, 3] = 8 * (np.sin(2*x[2])**2 + np.sin(2*x[4])**2) * (np.cos(2*x[3])**2 - np.sin(2*x[3])**2) + 0.2
    H[3, 4] = 16 * np.sin(2*x[3]) * np.cos(2*x[3]) * np.sin(2*x[4]) * np.cos(2*x[4])
    H[4, 4] = 8 * np.sin(2*x[3])**2 * (np.cos(2*x[4])**2 - np.sin(2*x[4])**2) + 0.1
    # Fill symmetric lower triangle
    H[1, 0] = H[0, 1]
    H[2, 1] = H[1, 2]
    H[3, 2] = H[2, 3]
    H[4, 3] = H[3, 4]
    return H


# ===========================================================================
# Convenience: build a SimpleNamespace "problem" object for each test problem
# ===========================================================================

from types import SimpleNamespace

def get_all_problems():
    """Return a list of SimpleNamespace problem objects for all 12 test problems."""
    problems = [
        # P1
        SimpleNamespace(
            name="P1_quad_10_10",
            x0=_x0_10.copy(),
            compute_f=quad_10_10_func,
            compute_g=quad_10_10_grad,
            compute_H=quad_10_10_Hess,
        ),
        # P2
        SimpleNamespace(
            name="P2_quad_10_1000",
            x0=_x0_10.copy(),
            compute_f=quad_10_1000_func,
            compute_g=quad_10_1000_grad,
            compute_H=quad_10_1000_Hess,
        ),
        # P3
        SimpleNamespace(
            name="P3_quad_1000_10",
            x0=_x0_1000.copy(),
            compute_f=quad_1000_10_func,
            compute_g=quad_1000_10_grad,
            compute_H=quad_1000_10_Hess,
        ),
        # P4
        SimpleNamespace(
            name="P4_quad_1000_1000",
            x0=_x0_1000.copy(),
            compute_f=quad_1000_1000_func,
            compute_g=quad_1000_1000_grad,
            compute_H=quad_1000_1000_Hess,
        ),
        # P5
        SimpleNamespace(
            name="P5_quartic_1",
            x0=np.array([np.cos(70.0), np.sin(70.0), np.cos(70.0), np.sin(70.0)]),
            compute_f=quartic_1_func,
            compute_g=quartic_1_grad,
            compute_H=quartic_1_Hess,
        ),
        # P6
        SimpleNamespace(
            name="P6_quartic_2",
            x0=np.array([np.cos(70.0), np.sin(70.0), np.cos(70.0), np.sin(70.0)]),
            compute_f=quartic_2_func,
            compute_g=quartic_2_grad,
            compute_H=quartic_2_Hess,
        ),
        # P7
        SimpleNamespace(
            name="P7_rosenbrock_2",
            x0=np.array([-1.2, 1.0]),
            compute_f=rosenbrock_2_func,
            compute_g=rosenbrock_2_grad,
            compute_H=rosenbrock_2_Hess,
        ),
        # P8
        SimpleNamespace(
            name="P8_rosenbrock_100",
            x0=np.concatenate([[-1.2], np.ones(99)]),
            compute_f=rosenbrock_100_func,
            compute_g=rosenbrock_100_grad,
            compute_H=rosenbrock_100_Hess,
        ),
        # P9
        SimpleNamespace(
            name="P9_data_fit_2",
            x0=np.array([1.0, 1.0]),
            compute_f=data_fit_2_func,
            compute_g=data_fit_2_grad,
            compute_H=data_fit_2_Hess,
        ),
        # P10
        SimpleNamespace(
            name="P10_exp_10",
            x0=np.array([1.0] + [0.0] * 9),
            compute_f=exp_10_func,
            compute_g=exp_10_grad,
            compute_H=exp_10_Hess,
        ),
        # P11
        SimpleNamespace(
            name="P11_exp_100",
            x0=np.array([1.0] + [0.0] * 99),
            compute_f=exp_100_func,
            compute_g=exp_100_grad,
            compute_H=exp_100_Hess,
        ),
        # P12
        SimpleNamespace(
            name="P12_genhumps_5",
            x0=np.array([-506.2, 506.2, 506.2, 506.2, 506.2]),
            compute_f=genhumps_5_func,
            compute_g=genhumps_5_grad,
            compute_H=genhumps_5_Hess,
        ),
    ]
    return problems
