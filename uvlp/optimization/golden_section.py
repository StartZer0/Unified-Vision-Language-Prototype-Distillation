"""
Golden Section Search for derivative-free 1D maximization.

This module implements the golden section search algorithm for finding
the optimal alpha value in the UVLP framework.
"""

from __future__ import annotations

from typing import Callable, Tuple

import math


# Golden ratio constant
PHI = (1 + math.sqrt(5)) / 2


def golden_section_maximize(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-3
) -> Tuple[float, float]:
    """Derivative-free 1D maximization via golden section search.
    
    Finds the value x in [a, b] that maximizes f(x).
    
    Args:
        f: Objective function to maximize.
        a: Lower bound of search interval.
        b: Upper bound of search interval.
        tol: Tolerance for convergence.
        
    Returns:
        Tuple of (optimal x, f(x) at optimal x).
    """
    c = b - (b - a) / PHI
    d = a + (b - a) / PHI
    
    fc = f(c)
    fd = f(d)
    
    while abs(b - a) > tol:
        if fc < fd:  # Maximizing, so move toward higher value
            a = c
            c = d
            fc = fd
            d = a + (b - a) / PHI
            fd = f(d)
        else:
            b = d
            d = c
            fd = fc
            c = b - (b - a) / PHI
            fc = f(c)
    
    x_opt = (a + b) / 2
    return x_opt, f(x_opt)


def golden_section_minimize(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-3
) -> Tuple[float, float]:
    """Derivative-free 1D minimization via golden section search.
    
    Finds the value x in [a, b] that minimizes f(x).
    
    Args:
        f: Objective function to minimize.
        a: Lower bound of search interval.
        b: Upper bound of search interval.
        tol: Tolerance for convergence.
        
    Returns:
        Tuple of (optimal x, f(x) at optimal x).
    """
    def neg_f(x):
        return -f(x)
    
    x_opt, neg_val = golden_section_maximize(neg_f, a, b, tol)
    return x_opt, -neg_val

