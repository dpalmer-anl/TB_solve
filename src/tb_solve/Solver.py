import torch
from scipy.sparse.linalg import eigsh
import math
from typing import Tuple, Optional, Sequence, Iterable, List
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from .cython_scripts.pexsi_wrapper import run_pexsi
from mpi4py import MPI

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

def prepare_distributed_csc(global_H, comm, diag_shift: float = 1e-8):
    """
    Partitions a global Hamiltonian matrix into a distributed CSC format
    suitable for PEXSI.
    """
    n_ranks = comm.Get_size()
    mpirank = comm.Get_rank()
    nrows = global_H.shape[0]
    
    # 1. Convert to Scipy CSC for easy slicing
    if not isinstance(global_H, csc_matrix):
        global_H = csc_matrix(global_H)
    
    # Apply a small diagonal shift if needed to avoid zero diagonal
    if diag_shift is not None and diag_shift > 0:
        diag_vals = global_H.diagonal()
        if np.any(diag_vals == 0):
            global_H = global_H + csc_matrix(np.eye(nrows) * diag_shift)

    # 2. Determine column distribution (linear split)
    cols_per_rank = nrows // n_ranks
    start_col = mpirank * cols_per_rank
    # The last rank takes any remaining columns
    end_col = (mpirank + 1) * cols_per_rank if mpirank < n_ranks - 1 else nrows
    
    numColLocal = end_col - start_col
    
    # 3. Slice the local portion of the matrix
    # local_H contains columns from start_col to end_col
    local_H = global_H[:, start_col:end_col]
    
    # 4. Extract CSC arrays
    # colptrLocal: start of each column in the rowind/nzval arrays
    colptrLocal = local_H.indptr.astype(np.int32)
    # rowindLocal: row indices of non-zero elements
    rowindLocal = local_H.indices.astype(np.int32)
    # HnzvalLocal: non-zero values
    if np.iscomplexobj(local_H.data):
        HnzvalLocal = np.ascontiguousarray(local_H.data, dtype=np.complex128)
    else:
        HnzvalLocal = np.ascontiguousarray(local_H.data, dtype=np.float64)
    
    # PEXSI expects 1-based indexing for CSC pointers/indices.
    colptrLocal = (colptrLocal + 1).astype(np.int32, copy=False)
    rowindLocal = (rowindLocal + 1).astype(np.int32, copy=False)
    nnz = global_H.nnz

    return nrows, nnz, colptrLocal, rowindLocal, HnzvalLocal

def get_density_matrix_pexsi(Hamiltonian):
    num_electrons = Hamiltonian.shape[0] 
    comm = MPI.COMM_WORLD

    # Define PEXSI process grid (e.g., for a 1D column distribution)
    # Note: nprow * npcol must be <= total MPI ranks [cite: 366]
    nprow = 1
    npcol = comm.Get_size()

    # Prepare local matrix data
    nrows, nnz, colptr, rowind, h_vals = prepare_distributed_csc(Hamiltonian, comm)

    # Call the Cython wrapper
    # Returns the local non-zero values of the Density Matrix [cite: 351]
    dm_local_nzval, rowind, colind, mu = run_pexsi(
        comm, nprow, npcol, nrows, nnz, colptr, rowind, h_vals, num_electrons
    )

    return dm_local_nzval, rowind, colind, mu

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

def Solve_Hamiltonian(Hamiltonian, Overlap=None, method="diagonalization", 
                        return_eigvals=False, return_eigvecs=False, return_density_matrix=True, 
                        nbands=20, which='LM',fermi_level=0,**kwargs) -> torch.Tensor:
    """Solve the Hamiltonian using the specified method.
    
    This is the main entry point for solving tight-binding Hamiltonians. It supports
    various methods including full diagonalization, sparse diagonalization, density 
    matrix minimization, and Fermi operator expansion.
    
    Args:
        Hamiltonian (torch.Tensor): The Hamiltonian matrix of shape (N,N).
        Overlap (torch.Tensor, optional): The Overlap matrix for generalized eigenvalue problems.
            Defaults to None. Not supported for all methods.
        method (str, optional): The solver method to use. Options are:
            - "diagonalization": Full diagonalization (default).
            - "sparse_diagonalization": Sparse diagonalization using ARPACK (CPU only).
            - "pexsi": Parallelized PEXSI solver (CPU only).
        return_eigvals (bool, optional): Whether to return eigenvalues. Defaults to False.
        return_eigvecs (bool, optional): Whether to return eigenvectors. Defaults to False.
        return_density_matrix (bool, optional): Whether to return the density matrix. 
            Defaults to True. Note: Some methods only support specific return types.
        nbands (int, optional): Number of bands to compute for sparse diagonalization. Defaults to 20.
        which (str, optional): Which eigenvalues to find for sparse diagonalization (e.g., 'LM', 'SA'). 
            Defaults to 'LM' (Largest Magnitude).
        **kwargs: Additional keyword arguments passed to the specific solver methods.
            - kbT (float): Temperature for Fermi operator expansion.
            - spin_degeneracy (float): Spin degeneracy factor.

    Returns:
        torch.Tensor or Tuple: By default, returns the density matrix (torch.Tensor).
        If multiple return flags are set, returns a tuple.
        Note: The return type depends on the requested outputs and the method used.
    
    Raises:
        ValueError: If an invalid method is specified or incompatible arguments are provided.
    """
    # Validation checks
    if method in ["pexsi"]:
         # These checks are redundant with the specific blocks but good for early exit
         pass

    if not isinstance(Hamiltonian, torch.Tensor) and method != "sparse_diagonalization":
        Hamiltonian = torch.tensor(Hamiltonian)
        if Overlap is not None:
            Overlap = torch.tensor(Overlap)
    elif method == "sparse_diagonalization":
        if not isinstance(Hamiltonian, np.ndarray):
            Hamiltonian = np.array(Hamiltonian)
            if Overlap is not None:
                Overlap = np.array(Overlap)
        Hamiltonian = np.squeeze(Hamiltonian)
        if Overlap is not None:
            Overlap = np.squeeze(Overlap)

    if method == "diagonalization":
        if Overlap is not None:
            eigvals, eigvecs = generalized_eigen_torch(Hamiltonian, Overlap)
            if not return_density_matrix:
                if return_eigvals and return_eigvecs:
                    return eigvals, eigvecs
                if return_eigvals:
                    return eigvals
                if return_eigvecs:
                    return eigvecs

            nocc = len(eigvals)//2
            density_matrix = 2*eigvecs[:, :nocc] @ eigvecs[:, :nocc].T
            if return_eigvals and return_eigvecs:
                return density_matrix, eigvals, eigvecs
            if return_eigvals:
                return density_matrix, eigvals
            if return_eigvecs:
                return density_matrix, eigvecs
            return density_matrix
        else:
            eigvals, eigvecs = torch.linalg.eigh(Hamiltonian)
            if not return_density_matrix:
                if return_eigvals and return_eigvecs:
                    return eigvals, eigvecs
                if return_eigvals:
                    return eigvals
                if return_eigvecs:
                    return eigvecs

            nocc = len(eigvals)//2
            density_matrix = 2*eigvecs[:, :nocc] @ eigvecs[:, :nocc].T
            if return_eigvals and return_eigvecs:
                return density_matrix, eigvals, eigvecs
            if return_eigvals:
                return density_matrix, eigvals
            if return_eigvecs:
                return density_matrix, eigvecs
            return density_matrix
    
    elif method == "sparse_diagonalization":
        print("Sparse diagonalization is a linear scaling method, but is only implemented for CPU's")
        if Overlap is not None:
            eigvals, eigvecs = eigsh(Hamiltonian, k=nbands,sigma=fermi_level,M=Overlap, which=which,**kwargs)
        else:
            eigvals, eigvecs = eigsh(Hamiltonian, k=nbands,sigma=fermi_level, which=which,**kwargs)
        return eigvals,eigvecs

    elif method == "pexsi":
        if Overlap is not None:
            raise ValueError("Overlap not supported for PEXSI")
        if return_eigvals or return_eigvecs:
            raise ValueError("return_eigvals/eigvecs not supported for PEXSI. Only supports return_density_matrix=True.")
        dm_vals, rowind, colind, _mu = get_density_matrix_pexsi(Hamiltonian)
        dm_sparse = coo_matrix((dm_vals, (rowind, colind)), shape=(Hamiltonian.shape[0], Hamiltonian.shape[1]))
        return torch.from_numpy(dm_sparse.toarray())

    else:
        raise ValueError("Invalid method")

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

    density_matrix_pexsi = Solve_Hamiltonian(Hamiltonian, method="pexsi")
    print("PEXSI density matrix = \n", torch.round(density_matrix_pexsi, decimals=3))

    density_matrix_diag = Solve_Hamiltonian(Hamiltonian, method="diagonalization")
    print("Diagonalization density matrix =\n", torch.round(density_matrix_diag, decimals=3))
    
    print("Difference norm:", torch.norm(density_matrix_pexsi - density_matrix_diag))

    