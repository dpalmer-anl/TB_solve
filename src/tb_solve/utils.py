import torch
import math
from tb_solve.Solver import device, gpu_avail, Solve_Hamiltonian, fermi_operator_expansion

from scipy import linalg
from scipy.optimize import linear_sum_assignment

def disentangle_bands(e, psi):
    """disentangle bands based on adjacent eigenvector overlap. assumes eigvals, eigvecs are sorted by kpoint index.
    Use for disentangling bands from sparse diagonalization"""
    nbands = e.shape[0]
    sorted_levels = [e[0]]
    sorted_psi = [psi[0]]
    for i in range(nbands-1):
        e2, psi2 = e[i+1], psi[i+1]
        perm, line_breaks = best_match(psi[i], psi2)
        e2 = e2[perm]
        intermediate = (e + e2) / 2
        intermediate[line_breaks] = None
        psi = psi2[:, perm]
        e = e2
        sorted_psi.append(psi)
        #sorted_levels.append(intermediate)
        sorted_levels.append(e)
    return sorted_levels, sorted_psi

def best_match(psi1, psi2, threshold=None):
    """Find the best match of two sets of eigenvectors.

    
    Parameters:
    -----------
    psi1, psi2 : numpy 2D complex arrays
        Arrays of initial and final eigenvectors.
    threshold : float, optional
        Minimal overlap when the eigenvectors are considered belonging to the same band.
        The default value is :math:`1/(2N)^{1/4}`, where :math:`N` is the length of each eigenvector.
    
    Returns:
    --------
    sorting : numpy 1D integer array
        Permutation to apply to ``psi2`` to make the optimal match.
    diconnects : numpy 1D bool array
        The levels with overlap below the ``threshold`` that should be considered disconnected.
    """
    if threshold is None:
        threshold = (2 * psi1.shape[0])**-0.25
    Q = np.abs(psi1.T.conj() @ psi2)  # Overlap matrix
    orig, perm = linear_sum_assignment(-Q)
    return perm, Q[orig, perm] < threshold
    
def Solver_helper(Hamiltonian, method="diagonalization", nbands=None, max_iterations=100, n_moments=100, **kwargs):
    """Generate a time estimate for solving the Hamiltonian.
    
    Estimates the computation time based on the problem size (N), the selected method,
    and the available hardware (CPU vs GPU). Uses empirical coefficients for FLOPs
    and bandwidth.

    Args:
        Hamiltonian (torch.Tensor): The Hamiltonian matrix (used for size N and sparsity check).
        method (str, optional): The solver method to estimate. Defaults to "diagonalization".
        nbands (int, optional): Number of bands (for sparse diagonalization). Defaults to None.
        max_iterations (int, optional): Max iterations (for purification). Defaults to 100.
        n_moments (int, optional): Number of moments (for Fermi operator expansion). Defaults to 100.
        **kwargs: Additional arguments.

    Returns:
        float: Estimated time in seconds.
    """
    N = Hamiltonian.shape[0]
    
    # Time constants estimated for modern hardware (seconds per operation count)
    # CPU: ~1e-9 sec/flop
    # GPU: ~1e-11 sec/flop (approx 100x speedup for dense matmul)
    
    # Constants A, B, C for O(N), O(N^2), O(N^3)
    # These are very rough estimates.
    
    # O(N^3) dense diagonalization
    coeff_diag_cpu = 5e-11 # e.g. 50 GFLOPS effective
    coeff_diag_gpu = 5e-13 # e.g. 5 TFLOPS effective
    
    # O(N^2) per iteration (dense matmul)
    coeff_mm_cpu = 1e-11
    coeff_mm_gpu = 1e-13
    
    # O(N) sparse ops (depends heavily on sparsity)
    coeff_sparse_cpu = 1e-8
    
    if method == "diagonalization":
        if device.type == "cpu":
            # O(N^3)
            time_estimate = coeff_diag_cpu * (N**3)
        else: # gpu
            time_estimate = coeff_diag_gpu * (N**3)
            
    elif method == "sparse_diagonalization":
        if not nbands:
            nbands = 20 # default assumption
        # Arpack / Lanczos: k * O(N) or O(N^2) depending on sparsity.
        # Assuming sparse MVM is O(N) * sparsity_factor. 
        # Let's approximate as O(N * nbands * iterations)
        # Iterations typically ~ O(nbands) or constant?
        
        if device.type == "cpu":
            # Rough scaling
            time_estimate = coeff_sparse_cpu * N * nbands * 100 
        else:
            print("Sparse diagonalization is a linear scaling method, but is only implemented for CPU's")
            time_estimate = coeff_sparse_cpu * N * nbands * 100

    elif method == "density_matrix_purification":
        # Matrix multiplications: O(N^3) actually for dense P
        # Purification involves dense matrix multiplies P^2, P^3 even if H is sparse (fill-in).
        # So it is O(N^3) per iteration unless P remains sparse (which it usually doesn't for metals/small gap).
        # If we assume dense:
        ops_per_iter = 4 * (N**3) # 2 matmuls (P^2, P^3) + overhead
        
        if device.type == "cpu":
            time_estimate = coeff_mm_cpu * ops_per_iter * max_iterations
        else:
            # GPU acceleration for dense matmul is good
             time_estimate = coeff_mm_gpu * ops_per_iter * max_iterations

    elif method == "fermi_operator_expansion":
        # KPM/Chebyshev: involves sparse MVM: H * T_k.
        # Cost: n_moments * O(N_nz)
        # If H is dense: n_moments * N^2
        # If H is sparse: n_moments * N * avg_degree
        
        is_sparse = Hamiltonian.is_sparse
        if is_sparse:
            # Estimate avg degree
            # n_nz = Hamiltonian._nnz() if hasattr(Hamiltonian, '_nnz') else N*10
            ops_per_moment = N * 100 # Approx sparse MVM
            coeff = coeff_sparse_cpu
        else:
            ops_per_moment = N**2
            coeff = coeff_mm_cpu
            
        if device.type == "cpu":
            time_estimate = coeff * ops_per_moment * n_moments
        else:
            # GPU can accelerate sparse or dense MVM
            # Approx 10x-100x speedup
            if is_sparse:
                 # Sparse support on GPU varies, assume moderate speedup
                 time_estimate = (coeff * 0.1) * ops_per_moment * n_moments
            else:
                 time_estimate = coeff_mm_gpu * ops_per_moment * n_moments
                 
    else:
        raise ValueError("Invalid method")
        
    print(f"Time estimate for {method} is {time_estimate:.2e} seconds, on device: {device} with Number of orbitals={N}")
    return time_estimate

def Get_optimal_solver(Hamiltonian, method="all", **kwargs):
    """
    Selects the optimal solver method based on the estimated computation time.
    
    Args:
        Hamiltonian (torch.Tensor): The Hamiltonian matrix.
        method (str, optional): "all" or specific method to check. Defaults to "all".
        **kwargs: Additional arguments for Solver_helper.
        
    Returns:
        tuple: (optimal_method_name, estimated_time)
    """
    methods = ["diagonalization", "sparse_diagonalization", "density_matrix_purification", "fermi_operator_expansion"]
    
    best_method = None
    min_time = float('inf')
    
    for m in methods:
        try:
            # Skip incompatible methods if possible (e.g., sparse diag on GPU if not supported)
            # Solver_helper prints a message but returns a time, so we can use it.
            # However, sparse_diagonalization doesn't give density matrix, so if user needs DM, this is tricky.
            # Assuming user wants optimal solver for *some* task.
            
            # Special handling: sparse_diagonalization is only for eigenvalues
            # If Hamiltonian is small, diagonalization is usually best.
            
            est_time = Solver_helper(Hamiltonian, method=m, **kwargs)
            if est_time < min_time:
                min_time = est_time
                best_method = m
        except Exception:
            continue
            
    return best_method, min_time

def Converge_Solver_settings(Hamiltonian, method="fermi_operator_expansion", variable="n_moments", 
                             start_value=50, end_value=500, step=50, tolerance=1e-3, **kwargs):
    """
    Finds the optimal parameter setting for a solver by checking convergence.
    
    Currently tailored for 'n_moments' in Fermi Operator Expansion.
    
    Args:
        Hamiltonian (torch.Tensor): The Hamiltonian matrix.
        method (str): Solver method.
        variable (str): Parameter to optimize (e.g., "n_moments").
        start_value (int/float): Starting value for the parameter.
        end_value (int/float): Maximum value.
        step (int/float): Step size.
        tolerance (float): Convergence tolerance (difference norm).
        **kwargs: Other fixed arguments for the solver (e.g., kbT).
        
    Returns:
        int/float: The converged value of the parameter.
    """
    
    prev_result = None
    converged_value = end_value
    
    print(f"Converging {variable} for {method}...")
    
    current_value = start_value
    while current_value <= end_value:
        # Construct kwargs with the current variable value
        current_kwargs = kwargs.copy()
        current_kwargs[variable] = current_value
        
        # Run solver
        result = Solve_Hamiltonian(Hamiltonian, method=method, **current_kwargs)
        
        if prev_result is not None:
            # Check convergence (Frobenius norm of difference)
            diff = torch.norm(result - prev_result).item()
            print(f"  {variable}={current_value}: diff={diff:.2e}")
            
            if diff < tolerance:
                print(f"  Converged at {variable}={current_value}")
                converged_value = current_value
                break
        else:
            print(f"  {variable}={current_value}: (initial)")
            
        prev_result = result
        current_value += step
        
    return converged_value
