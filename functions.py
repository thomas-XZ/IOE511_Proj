"""IOE 511/MATH 562, University of Michigan
Code written by: Kayla Huang, Thomas Xu Zhang, and Wuhao Cao
Final Project - Problems

Define all the functions and calculate their gradients and Hessians, those functions include:
    (1) Rosenbrock function
    (2) Quadractic function
"""
import numpy as np

# Problem 1: A randomly generated convex quadratic function. 
# Dimension n = 10; Condition number κ = 10. Starting Point: rng(0); x 0=20*rand(10,1)-10
def P1_quad_10_10(x, Q, q):
    '''Function that computes the function value for the Quadratic function
    Input:
        x: n-dimensional vector
        Q: n x n matrix
        q: n-dimensional vector
    Output:
        f(x)'''
    
    return 0.5 * x.T @ Q @ x + q.T @ x

def P1_quad_10_10_grad(x, Q, q):
    '''Function that computes the gradient of the Quadratic function
    Input:
        x: n-dimensional vector
        Q: n x n matrix
        q: n-dimensional vector
    Output:
        g = nabla f(x)'''
    
    return Q @ x + q

def P1_quad_10_10_Hess(x, Q):
    '''Function that computes the Hessian of the Quadratic function
    Input:
        x: n-dimensional vector
        Q: n x n matrix
    Output:
        H = nabla^2 f(x)'''
    
    return Q   

# Problem 2: A randomly generated convex quadratic function. 
# Dimension n = 10; Condition number κ = 1000. Starting Point: rng(0); x 0=20*rand(10,1)-10
def P2_quad_10_1000(x, Q, q):
    '''Function that computes the function value for the Quadratic function
    Input:
        x: n-dimensional vector
        Q: n x n matrix
        q: n-dimensional vector
    Output:
        f(x)'''
    
    return 0.5 * x.T @ Q @ x + q.T @ x

def P2_quad_10_1000_grad(x, Q, q):
    '''Function that computes the gradient of the Quadratic function
    Input:
        x: n-dimensional vector
        Q: n x n matrix
        q: n-dimensional vector
    Output:
        g = nabla f(x)'''
    
    return Q @ x + q

def P2_quad_10_1000_Hess(x, Q):
    '''Function that computes the Hessian of the Quadratic function
    Input:
        x: n-dimensional vector
        Q: n x n matrix
    Output:
        H = nabla^2 f(x)'''
    
    return Q   

# Problem 3: A randomly generated convex quadratic function. 
# Dimension n = 1000; Condition number κ = 10. Starting Point: rng(0); x 0=20*rand(1000,1)-10
def P3_quad_1000_10(x, Q, q):
    '''Function that computes the function value for the Quadratic function
    Input:
        x: n-dimensional vector
        Q: n x n matrix
        q: n-dimensional vector
    Output:
        f(x)'''
    return 0.5 * x.T @ Q @ x + q.T @ x

def P3_quad_1000_10_grad(x, Q, q):
    '''Function that computes the gradient of the Quadratic function
    Input:
        x: n-dimensional vector
        Q: n x n matrix
        q: n-dimensional vector
    Output:
        g = nabla f(x)'''
    
    return Q @ x + q

def P3_quad_1000_10_Hess(x, Q):
    '''Function that computes the Hessian of the Quadratic function
    Input:
        x: n-dimensional vector
        Q: n x n matrix
    Output:
        H = nabla^2 f(x)'''
    
    return Q   

# Problem 4: A randomly generated convex quadratic function.
# Dimension n = 1000; Condition number κ = 1000. Starting Point: rng(0); x 0=20*rand(1000,1)-10
def P4_quad_1000_1000(x, Q, q):
    '''Function that computes the function value for the Quadratic function
    Input:
        x: n-dimensional vector
        Q: n x n matrix
        q: n-dimensional vector
    Output:
        f(x)'''
    return 0.5 * x.T @ Q @ x + q.T @ x

def P4_quad_1000_1000_grad(x, Q, q):
    '''Function that computes the gradient of the Quadratic function
    Input:
        x: n-dimensional vector
        Q: n x n matrix
        q: n-dimensional vector
    Output:
        g = nabla f(x)'''
    
    return Q @ x + q

def P4_quad_1000_1000_Hess(x, Q):
    '''Function that computes the Hessian of the Quadratic function
    Input:
        x: n-dimensional vector
        Q: n x n matrix
    Output:
        H = nabla^2 f(x)'''
    
    return Q

def P5_quartic_1(x):
    '''Function that computes the function value for the Quartic function
    Input:
        x: n-dimensional vector
        n = 4
    Output:
        f(x)'''
    # Define Q
    Q = np.array([
        [5,   1,   0,   0.5],
        [1,   4,   0.5, 0  ],
        [0,   0.5, 3,   0  ],
        [0.5, 0,   0,   2  ]
    ])
    # Define sigma
    sigma = 10e-4
    













'''def rosen_func(x):
    """Function that computes the function value for the Rosenbrock function
    Input:
        x
    Output:
        f(x)
    """

    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    """Function that computes the gradient of the Rosenbrock function
    Input:
        x
    Output:
        g = nabla f(x)
    """
    return np.array(
        [
            2 * (x[0] - 1) - 400 * x[0] * (x[1] - x[0] ** 2),
            200 * (x[1] - x[0] ** 2),
        ]
    )


def rosen_Hess(x):
    """Function that computes the Hessian of the Rosenbrock function
    Input:
        x
    Output:
        H = nabla^2 f(x)
    """

    return np.array(
        [
            [2 + 1200 * x[0] ** 2 - 400 * x[1], -400 * x[0]],
            [-400 * x[0], 200],
        ]
    )

def quad_func(x, A, b, c):
    """Function that computes the function value for the Quadractic function
    Input:
        x
    Output:
        f(x)
    """

    return float(0.5 * x.T @ A @ x + b.T @ x + c)

def quad_grad(x, A, b):
    return A @ x + b

def quad_Hess(x, A):
    return A

def HW4_func2(x):
    w, z = x[0], x[1]
    y = np.array([1.5, 2.25, 2.625])
    f_val = 0
    for i in range(1, 4):
        f_val += (y[i-1] - w * (1 - z**i))**2
    return f_val

def HW4_func2_grad(x):
    w, z = x[0], x[1]
    y = np.array([1.5, 2.25, 2.625])
    g = np.zeros(2)
    for i in range(1, 4):
        g[0] += 2 * (y[i-1] - w * (1 - z**i)) * -(1 - z**i)
        g[1] += 2 * (y[i-1] - w * (1 - z**i)) * w * i * z**(i-1)
    return g

def HW4_func2_Hess(x):
    w, z = x[0], x[1]
    y = np.array([1.5, 2.25, 2.625])
    H = np.zeros((2, 2))
    for i in range(1, 4):
        H[0, 0] += 2 * (1 - z**i)**2
        H[0, 1] += 2 * (1 - z**i) * w * i * z**(i-1)
        H[1, 0] += 2 * (1 - z**i) * w * i * z**(i-1)
        H[1, 1] += 2 * w**2 * i**2 * z**(2*i-2)
    return H

def HW4_func3():
    z1 = x
    term1 = (np.exp(z1) - 1) / (np.exp(z1) + 1) + 0.1 * np.exp(-z1)
    term2 = np.sum((x[1:] - 1)**4) if len(x) > 1 else 0
    return term1 + term2

def HW4_func3_grad(x):
    z1 = x[0]
    term1_grad = (2 * np.exp(z1) / (np.exp(z1) + 1)**2 - 0.1 * np.exp(-z1))
    term2_grad = 4 * np.sum((x[1:] - 1)**3) if len(x) > 1 else 0
    return np.array([term1_grad + term2_grad])

def HW4_func3_Hess(x):
    z1 = x[0]
    term1_hess = (2 * np.exp(z1) * (np.exp(z1) - 1) / (np.exp(z1) + 1)**3 + 0.1 * np.exp(-z1))
    term2_hess = 12 * np.sum((x[1:] - 1)**2) if len(x) > 1 else 0
    return np.array([[term1_hess + term2_hess]])'''


