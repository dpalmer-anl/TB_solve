# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

def density_matrix_minimization_cy(
    double[:] h_data, 
    int[:] rows, 
    int[:] cols, 
    double n_electrons, 
    double mu, 
    double lmbda, 
    double lr=0.01, 
    int max_iter=1000, 
    double tol=1e-6
):
    """
    Minimizes the density matrix functional using sparse inputs.
    Returns the density matrix elements (rho_data) at the same indices.
    """
    cdef int n_elements = h_data.shape[0]
    cdef int i, it
    cdef double trace_rho, trace_rho2, rho_dot_h, grad_ij
    cdef double diff, delta_ij, step
    
    # Initialize rho_data (e.g., small random values or proportional to -H)
    cdef double[:] rho_data = np.zeros(n_elements, dtype=np.float64)
    for i in range(n_elements):
        rho_data[i] = -0.01 * h_data[i]

    for it in range(max_iter):
        trace_rho = 0.0
        trace_rho2 = 0.0
        
        # 1. Calculate Tr(rho) and Tr(rho^2) - O(N_nonzero)
        for i in range(n_elements):
            if rows[i] == cols[i]:
                trace_rho += rho_data[i]
            trace_rho2 += rho_data[i] * rho_data[i]
        
        # 2. Update elements using the gradient
        # Grad = H - mu*I + lambda*I - 2*lambda*rho
        diff = 0.0
        
        for i in range(n_elements):
            delta_ij = 1.0 if rows[i] == cols[i] else 0.0
            
            # Gradient of the functional
            grad_ij = h_data[i] - mu * delta_ij + lmbda * (delta_ij - 2.0 * rho_data[i])
            
            # Step
            step = lr * grad_ij
            rho_data[i] -= step
            diff += step * step
            
        if sqrt(diff / n_elements) < tol:
            break
            
    return np.asarray(rho_data)