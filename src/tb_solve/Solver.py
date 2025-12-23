import torch
from scipy.sparse.linalg import eigsh
import math
from typing import Tuple, Optional

# Intel GPU device configuration
def get_intel_gpu_device():
    """Get Intel GPU device if available, otherwise CPU.
    
    Checks for the availability of an Intel XPU (via torch.xpu) or an NVIDIA GPU
    (via torch.cuda) and returns the appropriate device and availability flag.
    
    Returns:
        tuple: A tuple containing:
            - device (torch.device): The selected device (xpu:0, cuda:0, or cpu).
            - gpu_avail (bool): True if a GPU/XPU is available, False otherwise.
    """
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu:0")  # Intel GPU
        print("Using Intel GPU:", device)
        return device, True
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")  # NVIDIA GPU fallback
        print("Using NVIDIA GPU:", device)
        return device, True
    else:
        device = torch.device("cpu")
        print("Using CPU:", device)
        return device, False

device, gpu_avail = get_intel_gpu_device()

def fermi_operator_expansion(Hamiltonian: torch.Tensor, kbT=1e-2, n_moments=100, spin_degeneracy=2.0) -> torch.Tensor:
    """Calculate density matrix using Fermi operator expansion with Jackson damping.
    
    This method approximates the density matrix using a Chebyshev polynomial expansion
    of the Fermi-Dirac distribution. It is suitable for large sparse matrices as it
    avoids full diagonalization.
    
    Args:
        Hamiltonian (torch.Tensor): (N,N) Hamiltonian matrix. Can be dense or sparse.
        kbT (float, optional): Thermal energy (Boltzmann constant * Temperature). Defaults to 1e-2.
        n_moments (int, optional): Number of Chebyshev moments to use in the expansion. Defaults to 100.
        spin_degeneracy (float, optional): Factor for spin degeneracy. Defaults to 2.0.

    Returns:
        torch.Tensor: The calculated density matrix of shape (N,N).
    """
    N = Hamiltonian.shape[0]
    device = Hamiltonian.device

    # 1. Estimate Spectral Bounds using Gerschgorin-like bounds
    if Hamiltonian.is_sparse:
        # Convert to dense for bounds estimation if feasible, or use conservative bounds
        # Using infinity norm estimate for speed and simplicity with sparse
        # |E| <= max_i sum_j |H_ij|
        
        # Calculate row sums of absolute values
        # We need to use values and indices
        H_vals = Hamiltonian.values()
        H_idxs = Hamiltonian.indices()
        
        # Use index_add_ to sum absolute values to rows
        row_sums = torch.zeros(N, device=device)
        row_sums.index_add_(0, H_idxs[0], H_vals.abs())
        
        max_row_sum = torch.max(row_sums)
        E_max = max_row_sum
        E_min = -max_row_sum
        
        # Estimate chemical potential: trace(H) / N
        # Trace is sum of diagonal elements
        diag_mask = (H_idxs[0] == H_idxs[1])
        trace_H = torch.sum(H_vals[diag_mask])
        
    else:
        # Dense implementation
        row_sums = torch.sum(torch.abs(Hamiltonian), dim=1)
        diag = torch.diagonal(Hamiltonian)
        # Gerschgorin bounds: center + radius
        # center = diag, radius = row_sum - |diag|
        off_diag_sum = row_sums - torch.abs(diag)
        E_max = torch.max(diag + off_diag_sum)
        E_min = torch.min(diag - off_diag_sum)
        trace_H = torch.trace(Hamiltonian)

    mu_approx = trace_H / N

    # 2. Rescaling to [-1, 1] (with safety margin)
    epsilon = 0.01
    half_width = (E_max - E_min) / 2.0
    center = (E_max + E_min) / 2.0
    scale = half_width * (1.0 + epsilon)
    
    # Rescaled Hamiltonian operator: H_tilde = (H - center) / scale
    # We will apply this dynamically during recurrence to avoid storing a dense matrix if H is sparse
    
    # Rescaled parameters
    mu_scaled = (mu_approx - center) / scale
    kbT_scaled = kbT / scale

    # 3. Calculate Chebyshev Coefficients (Moments) with Jackson Damping
    # Function to expand: Fermi-Dirac f(x) = 1 / (1 + exp((x - mu)/kbT))
    # Coefficients c_m = (2 / pi) * integral_{-1}^1 (f(x) * T_m(x) / sqrt(1-x^2)) dx
    # Using Chebyshev-Gauss quadrature
    
    # Number of quadrature points
    n_quad = 2 * n_moments
    k = torch.arange(1, n_quad + 1, device=device).float()
    x_k = torch.cos(math.pi * (k - 0.5) / n_quad)
    
    # Evaluate function at quadrature points
    # Need to map x_k (which is in [-1, 1]) back to energy or use scaled params
    # f_scaled(x) = 1 / (1 + exp((x - mu_scaled)/kbT_scaled))
    arg = (x_k - mu_scaled) / kbT_scaled
    # Clip arg to avoid overflow
    arg = torch.clamp(arg, min=-100, max=100)
    f_vals = 1.0 / (1.0 + torch.exp(arg))
    
    # Compute moments
    # c_m = (2 / n_quad) * sum_k f(x_k) * T_m(x_k)
    # T_m(x_k) = cos(m * theta_k)
    theta_k = math.pi * (k - 0.5) / n_quad
    
    moments = []
    for m in range(n_moments):
        cos_m_theta = torch.cos(m * theta_k)
        c_m = (2.0 / n_quad) * torch.sum(f_vals * cos_m_theta)
        
        # Jackson Damping factor g_m
        # g_m = ((M - m + 1)*cos(pi*m/(M+1)) + sin(pi*m/(M+1))*cot(pi/(M+1))) / (M+1)
        M = n_moments
        angle = math.pi * m / (M + 1)
        denom = M + 1
        sin_angle = math.sin(angle)
        cos_angle = math.cos(angle)
        cot_factor = 1.0 / math.tan(math.pi / (M + 1))
        
        g_m = ((M - m + 1) * cos_angle + sin_angle * cot_factor) / denom
        
        moments.append(c_m * g_m)
        
    moments[0] = moments[0] * 0.5 # Correction for c_0 definition in some conventions, but here
    # Standard Chebyshev series: f(x) = c_0/2 + sum_{m=1} c_m T_m(x)  OR  sum_{m=0}' c_m T_m
    # Our quadrature gives c_m as defined for the sum with weight 1 for m>0 and 1/2 for m=0?
    # Orthogonality: int T_n T_m / sqrt(1-x^2) = pi/2 (n!=0), pi (n=0).
    # c_m = (1/h_m) int f T_m w. 
    # h_0 = pi, h_m = pi/2.
    # Our c_m calculation used factor 2/n_quad.
    # sum (pi/n_quad) -> pi. 
    # So computed c_m approx (2/pi) * int.
    # For m=0, we need (1/pi) * int.
    # So c_0 should be divided by 2 relative to the formula used.
    # moments[0] = moments[0] * 0.5 <-- Already done above

    # 4. Clenshaw Recurrence / Accumulation
    # Density Matrix rho = sum_{m=0}^{M-1} c_m * g_m * T_m(H_tilde)
    
    # Initialize T_0, T_1
    # T_0 = I
    # T_1 = H_tilde
    
    # If H is sparse, we want to maintain sparsity if possible, but output is density matrix.
    # Since we need to return the full density matrix, let's accumulate into a dense tensor.
    
    rho = torch.eye(N, device=device) * moments[0]
    
    if n_moments > 1:
        # Calculate H_tilde parameters for recurrence: T_m = 2 * H_tilde * T_{m-1} - T_{m-2}
        # H_tilde = (H - center) / scale
        # Expansion: T_m = (2/scale) * H * T_{m-1} - (2*center/scale) * T_{m-1} - T_{m-2}
        
        factor_H = 2.0 / scale
        factor_I = 2.0 * center / scale
        
        T_prev = torch.eye(N, device=device) # T_0
        
        # Calculate T_1 = H_tilde
        if Hamiltonian.is_sparse:
            # T_1 must be dense as it accumulates into rho. 
            # We convert H to dense only for this step if needed, or use sparse operations.
            # Using to_dense() is acceptable here as T_curr is dense O(N^2) anyway.
            T_curr = (Hamiltonian.to_dense() - center * T_prev) / scale
        else:
            T_curr = (Hamiltonian - center * T_prev) / scale
            
        rho = rho + moments[1] * T_curr
        
        for m in range(2, n_moments):
            # T_next = 2 * H_tilde * T_curr - T_prev
            # Decomposed to use sparse H if available
            
            if Hamiltonian.is_sparse:
                # Sparse matrix multiplication O(N_nz * N) instead of O(N^3)
                H_times_T = torch.sparse.mm(Hamiltonian, T_curr)
            else:
                H_times_T = Hamiltonian @ T_curr
                
            T_next = factor_H * H_times_T - factor_I * T_curr - T_prev
            
            rho = rho + moments[m] * T_next
            
            T_prev = T_curr
            T_curr = T_next
            
    return rho * spin_degeneracy

def density_matrix_purification(H: torch.Tensor, epsilon=1e-6, max_iterations=100) -> torch.Tensor:
    """Compute the density matrix using the canonical purification method.
    
    This method iteratively refines an initial guess of the density matrix to achieve 
    idempotency (P^2 = P), ensuring that the matrix's eigenvalues are either 0 or 1.
    This corresponds to the zero-temperature density matrix.
    
    Args:
        H (torch.Tensor): Hamiltonian matrix of shape (N,N).
        epsilon (float, optional): Convergence threshold for energy change. Defaults to 1e-6.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 100.
        
    Returns:
        torch.Tensor: Purified density matrix of shape (N,N).
    """
    N = H.shape[0]
    device = H.device
    N_e = N // 2
    # Calculate mu = tr(H)/N
    mu = torch.trace(H) / N
    
    # Calculate H_max and H_min using Gerschgorin circle theorem
    # H_min = min_i(H_ii - sum_{j≠i} |H_ij|)
    # H_max = max_i(H_ii + sum_{j≠i} |H_ij|)
    
    # Calculate row sums of absolute values (excluding diagonal)
    row_sums = torch.sum(torch.abs(H), dim=1) - torch.abs(torch.diagonal(H))
    
    # Calculate Gerschgorin bounds
    H_min = torch.min(torch.diagonal(H) - row_sums)
    H_max = torch.max(torch.diagonal(H) + row_sums)
    
    # Calculate lambda = min[N_e/(H_max - mu), (N-N_e)/(mu - H_min)]
    lambda_term1 = N_e / (H_max - mu) if H_max != mu else float('inf')
    lambda_term2 = (N - N_e) / (mu - H_min) if mu != H_min else float('inf')
    lambda_val = min(lambda_term1, lambda_term2)
    

    I = torch.eye(N, device=device)
    P = (lambda_val / N) * (mu * I - H) + (N_e / N) * I

    # McWeeny purification iterations
    for iteration in range(max_iterations):
        # Calculate current energy E = tr(PH)
        E_old = torch.real(torch.trace(P @ H))
        
        # McWeeny purification: P_{n+1} = 3P_n² - 2P_n³
        P_squared = P @ P
        P_cubed = P_squared @ P
        
        # Calculate c_i parameter for adaptive update
        c_i = torch.trace(P_squared - P_cubed) / torch.trace(P - P_squared)
        
        # Adaptive update based on c_i
        if c_i <= 0.5:
            # P_{i+1} = [(1-2c_i) * P_i + (1+c_i)*P_squared - P_cubed]/(1-c_i)
            P = ((1 - 2*c_i) * P + (1 + c_i) * P_squared - P_cubed) / (1 - c_i)
        else:
            # P_{i+1} = [(1+c_i)*P_squared - P_cubed]/c_i
            P = ((1 + c_i) * P_squared - P_cubed) / c_i
        
        # Calculate new energy E = tr(PH)
        E_new = torch.real(torch.trace(P @ H))
        
        # Check convergence based on energy change
        energy_change = abs(E_new - E_old)
        if energy_change < epsilon:
            break
    
    if iteration == max_iterations - 1:
        print(f"Warning: Density matrix purification did not converge after {max_iterations} iterations")
    
    return P

def generalized_eigen_torch(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch-optimized generalized eigenvalue solver.
    
    Solves the generalized eigenvalue problem A @ v = lambda * B @ v.
    
    Args:
        A (torch.Tensor): Hermitian matrix A.
        B (torch.Tensor): Positive-definite matrix B (e.g., Overlap matrix).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - eigvals (torch.Tensor): The eigenvalues.
            - eigvecs (torch.Tensor): The eigenvectors.
    """
    Binv = torch.linalg.inv(B)
    renorm_A = Binv @ A
    eigvals, eigvecs = torch.linalg.eigh(renorm_A)
    
    # Normalize eigenvectors
    Q = eigvecs.conj().T @ B @ eigvecs
    U = torch.linalg.cholesky(torch.linalg.inv(Q))
    eigvecs = eigvecs @ U
    eigvals = torch.diag(eigvecs.conj().T @ A @ eigvecs).real
    
    return eigvals, eigvecs

def Solve_Hamiltonian(Hamiltonian: torch.Tensor, Overlap=None, method="diagonalization", 
                        return_eigvals=False, return_eigvecs=False, return_density_matrix=True, 
                        nbands=20, which='LM',**kwargs) -> torch.Tensor:
    """Solve the Hamiltonian using the specified method.
    
    This is the main entry point for solving tight-binding Hamiltonians. It supports
    various methods including full diagonalization, sparse diagonalization, density 
    matrix purification, and Fermi operator expansion.
    
    Args:
        Hamiltonian (torch.Tensor): The Hamiltonian matrix of shape (N,N).
        Overlap (torch.Tensor, optional): The Overlap matrix for generalized eigenvalue problems.
            Defaults to None. Not supported for all methods.
        method (str, optional): The solver method to use. Options are:
            - "diagonalization": Full diagonalization (default).
            - "sparse_diagonalization": Sparse diagonalization using ARPACK (CPU only).
            - "density_matrix_purification": Linear scaling purification (T=0).
            - "fermi_operator_expansion": Linear scaling Chebyshev expansion (finite T).
        return_eigvals (bool, optional): Whether to return eigenvalues. Defaults to False.
        return_eigvecs (bool, optional): Whether to return eigenvectors. Defaults to False.
        return_density_matrix (bool, optional): Whether to return the density matrix. 
            Defaults to True. Note: Some methods only support specific return types.
        nbands (int, optional): Number of bands to compute for sparse diagonalization. Defaults to 20.
        which (str, optional): Which eigenvalues to find for sparse diagonalization (e.g., 'LM', 'SA'). 
            Defaults to 'LM'.
        **kwargs: Additional keyword arguments passed to the specific solver methods.
            - kbT (float): Temperature for Fermi operator expansion.
            - n_moments (int): Number of moments for Fermi operator expansion.
            - epsilon (float): Convergence threshold for purification.
            - max_iterations (int): Max iterations for purification.
            - spin_degeneracy (float): Spin degeneracy factor.

    Returns:
        torch.Tensor or Tuple: By default, returns the density matrix (torch.Tensor).
        If multiple return flags are set, returns a tuple.
        Note: The return type depends on the requested outputs and the method used.
    
    Raises:
        ValueError: If an invalid method is specified or incompatible arguments are provided.
    """
    if method == "density_matrix_purification" or method == "fermi_operator_expansion" and Overlap is not None:
        pass # Checks are done in specific blocks below for better error messages or we can keep them here.
    
    # Validation checks
    if method in ["density_matrix_purification", "fermi_operator_expansion"]:
         # These checks are redundant with the specific blocks but good for early exit
         pass


    if method == "diagonalization":
        if Overlap is not None:
            eigvals, eigvecs = generalized_eigen_torch(Hamiltonian, Overlap)
            nocc = len(eigvals)//2
            density_matrix = 2*eigvecs[:, :nocc] @ eigvecs[:, :nocc].T
            if return_eigvals:
                return density_matrix, eigvals
            return density_matrix
        else:
            eigvals, eigvecs = torch.linalg.eigh(Hamiltonian)
            nocc = len(eigvals)//2
            density_matrix = 2*eigvecs[:, :nocc] @ eigvecs[:, :nocc].T
            if return_eigvals:
                return density_matrix, eigvals
            return density_matrix
    
    elif method == "sparse_diagonalization":
        print("Sparse diagonalization is a linear scaling method, but is only implemented for CPU's")
        if Overlap is not None:
            raise ValueError("Overlap not supported for sparse diagonalization")
        eigval, eigvecs = eigsh(Hamiltonian, nbands, which=which,**kwargs)
        if return_eigvals:
            return eigvals
        if return_eigvecs:
            return eigvals,eigvecs
        if return_density_matrix:
            raise ValueError("return_density_matrix not supported for sparse diagonalization. \
                            Only supports return_eigvals=True.")

    if method == "density_matrix_purification":
        if Overlap is not None:
            raise ValueError("Overlap not supported for density matrix purification")
        if return_eigvals or return_eigvecs:
             raise ValueError("return_eigvals/eigvecs not supported for density matrix purification. Only supports return_density_matrix=True.")
        
        epsilon = kwargs.get('epsilon', 1e-6)
        max_iterations = kwargs.get('max_iterations', 100)
        return density_matrix_purification(Hamiltonian, epsilon=epsilon, max_iterations=max_iterations)
        
    elif method == "fermi_operator_expansion":
        if Overlap is not None:
             raise ValueError("Overlap not supported for fermi_operator_expansion")
        if return_eigvals or return_eigvecs:
             raise ValueError("return_eigvals/eigvecs not supported for fermi_operator_expansion. Only supports return_density_matrix=True.")

        kbT = kwargs.get('kbT', 1.0)
        n_moments = kwargs.get('n_moments', 100) # Default moments
        spin_degeneracy = kwargs.get('spin_degeneracy', 2.0)
        return fermi_operator_expansion(Hamiltonian, kbT=kbT, n_moments=n_moments, spin_degeneracy=spin_degeneracy)
        
    else:
        raise ValueError("Invalid method")

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

if __name__=="__main__":
    N = 10
    t = 1.0
    Temperature = 1e-4 # Lower temperature for comparison with T=0 diagonalization
    Hamiltonian = torch.zeros((N,N))
    for i in range(N-1):
        Hamiltonian[i,i+1] = t 
        Hamiltonian[i+1,i] = t
        Hamiltonian[i,i] = 0.5
    Hamiltonian[0,N-1] = t
    Hamiltonian[N-1,0] = t
    print(Hamiltonian)
    
    # Needs sufficient moments for low temperature to capture the step function
    density_matrix = Solve_Hamiltonian(Hamiltonian, method="fermi_operator_expansion", kbT=Temperature, n_moments=100)
    print("FOE density matrix = \n", torch.round(density_matrix, decimals=3))

    density_matrix_diag = Solve_Hamiltonian(Hamiltonian, method="diagonalization")
    print("Diagonalization density matrix =\n", torch.round(density_matrix_diag, decimals=3))
    
    print("Difference norm:", torch.norm(density_matrix - density_matrix_diag))