import torch
import scipy.linalg
from scipy.sparse.linalg import eigsh
import math
from typing import Tuple, Optional, Sequence, Iterable, List, Union
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, issparse
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

def get_density_matrix_pexsi(Hamiltonian, temperature: float, numPoles: int):
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
        comm,
        nprow,
        npcol,
        nrows,
        nnz,
        colptr,
        rowind,
        h_vals,
        num_electrons,
        temperature,
        numPoles,
    )

    return dm_local_nzval, rowind, colind, mu

def generalized_eigen_torch(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch-optimized generalized eigenvalue solver for torch.Tensor inputs.

    Solves the generalized eigenvalue problem A @ v = lambda * B @ v using
    GPU-accelerated PyTorch operations. Both inputs must be ``torch.Tensor``.
    For numpy or sparse inputs use ``scipy.linalg.eigh`` directly, or pass
    the matrices through :func:`Solve_Hamiltonian` which dispatches
    automatically based on input type.

    Args:
        A (torch.Tensor): Hermitian matrix A of shape ``(N, N)``.
        B (torch.Tensor): Positive-definite overlap matrix B of shape ``(N, N)``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - eigvals (torch.Tensor): Real eigenvalues of shape ``(N,)``,
              sorted in ascending order.
            - eigvecs (torch.Tensor): B-orthonormal eigenvectors of shape
              ``(N, N)``, where column ``i`` corresponds to ``eigvals[i]``.
    """
    Binv = torch.linalg.inv(B)
    renorm_A = Binv @ A
    eigvals, eigvecs = torch.linalg.eigh(renorm_A)
    
    # Normalize eigenvectors with respect to B
    Q = eigvecs.conj().T @ B @ eigvecs
    U = torch.linalg.cholesky(torch.linalg.inv(Q))
    eigvecs = eigvecs @ U
    eigvals = torch.diag(eigvecs.conj().T @ A @ eigvecs).real
    
    return eigvals, eigvecs

def _detect_input_type(H) -> str:
    """Return a string tag for the matrix type of *H*.

    Returns:
        str: One of ``'torch'``, ``'csc'``, or ``'numpy'``.
    """
    if isinstance(H, torch.Tensor):
        return 'torch'
    elif issparse(H):
        return 'csc'
    else:
        return 'numpy'


def Solve_Hamiltonian(
    Hamiltonian: Union[torch.Tensor, np.ndarray, csc_matrix],
    Overlap: Union[torch.Tensor, np.ndarray, csc_matrix, None] = None,
    method: str = "diagonalization",
    return_eigvals: bool = False,
    return_eigvecs: bool = False,
    return_density_matrix: bool = True,
    nbands: int = 20,
    which: str = "LM",
    fermi_level: float = 0,
    numPoles: int = 50,
    temperature: float = 0.01,
    **kwargs,
):
    """Solve the Hamiltonian using the specified method.

    This is the main entry point for solving tight-binding Hamiltonians. It
    supports full diagonalization, sparse diagonalization, and the PEXSI
    Fermi-operator expansion.

    **Input / output type matching**

    The function accepts ``torch.Tensor``, ``numpy.ndarray``, or any SciPy
    sparse matrix (internally converted to ``csc_matrix``) for both
    ``Hamiltonian`` and ``Overlap``.  The *same* type is used for all
    returned arrays:

    +-------------------+------------------------------------------+-------------------------------------------+
    | Input type        | ``diagonalization`` solver               | Returned arrays                           |
    +===================+==========================================+===========================================+
    | ``torch.Tensor``  | :func:`generalized_eigen_torch` (with S) | ``torch.Tensor``                          |
    |                   | or ``torch.linalg.eigh`` (without S)     |                                           |
    +-------------------+------------------------------------------+-------------------------------------------+
    | ``numpy.ndarray`` | ``scipy.linalg.eigh``                    | ``numpy.ndarray``                         |
    +-------------------+------------------------------------------+-------------------------------------------+
    | ``csc_matrix``    | ``scipy.linalg.eigh`` on dense view      | ``numpy.ndarray`` (eigvals/eigvecs),      |
    |                   |                                          | ``csc_matrix`` (density matrix)           |
    +-------------------+------------------------------------------+-------------------------------------------+

    For ``sparse_diagonalization``, SciPy's ``eigsh`` always returns
    ``numpy.ndarray`` regardless of input type (this is a limitation of
    ARPACK).  For ``pexsi``, the density matrix is returned as
    ``torch.Tensor``, ``numpy.ndarray``, or ``csc_matrix`` matching the input.

    Args:
        Hamiltonian (torch.Tensor | numpy.ndarray | csc_matrix):
            The Hamiltonian matrix of shape ``(N, N)``.
        Overlap (torch.Tensor | numpy.ndarray | csc_matrix | None, optional):
            Overlap matrix for generalized eigenvalue problems.  Must be the
            same type as ``Hamiltonian``.  Defaults to ``None``.  Not
            supported by the ``pexsi`` method.
        method (str, optional): Solver to use. Options:

            - ``"diagonalization"`` – Full dense diagonalization (default).
              Dispatches to PyTorch or SciPy depending on input type.
            - ``"sparse_diagonalization"`` – Iterative sparse solver via
              ARPACK (CPU only). Output is always ``numpy.ndarray``.
            - ``"pexsi"`` – Parallelized Fermi-operator expansion (CPU / MPI).

        return_eigvals (bool, optional): Include eigenvalues in the return
            value. Defaults to ``False``.
        return_eigvecs (bool, optional): Include eigenvectors in the return
            value. Defaults to ``False``.
        return_density_matrix (bool, optional): Include the density matrix in
            the return value. Defaults to ``True``.
        nbands (int, optional): Number of bands for ``sparse_diagonalization``.
            Defaults to ``20``.
        which (str, optional): Which eigenvalues to target in
            ``sparse_diagonalization`` (e.g. ``'LM'``, ``'SA'``).
            Defaults to ``'LM'``.
        fermi_level (float, optional): Shift (sigma) for ``sparse_diagonalization``
            to find eigenvalues near the Fermi level. Defaults to ``0``.
        numPoles (int, optional): Number of poles for the ``pexsi`` solver.
            Defaults to ``50``.
        temperature (float, optional): Electronic temperature (eV) for the
            ``pexsi`` solver. Defaults to ``0.01``.
        **kwargs: Extra keyword arguments forwarded to the underlying solver
            (e.g. ``tol``, ``maxiter`` for ``eigsh``).

    Returns:
        The return value depends on the active flags:

        - Only ``return_density_matrix=True`` (default):
          returns *density_matrix*.
        - ``return_density_matrix=True`` and ``return_eigvals=True``:
          returns *(density_matrix, eigvals)*.
        - ``return_density_matrix=True`` and ``return_eigvecs=True``:
          returns *(density_matrix, eigvecs)*.
        - ``return_density_matrix=True``, ``return_eigvals=True``, and
          ``return_eigvecs=True``: returns *(density_matrix, eigvals, eigvecs)*.
        - ``return_density_matrix=False``, ``return_eigvals=True``, and
          ``return_eigvecs=True``: returns *(eigvals, eigvecs)*.
        - ``return_density_matrix=False`` and ``return_eigvals=True``:
          returns *eigvals*.
        - ``return_density_matrix=False`` and ``return_eigvecs=True``:
          returns *eigvecs*.

        All arrays match the type of ``Hamiltonian`` (see the table above).

    Raises:
        ValueError: If an invalid ``method`` is specified, or if incompatible
            argument combinations are used (e.g. ``Overlap`` with ``pexsi``).
    """
    # --- detect and normalise input type ---
    input_type = _detect_input_type(Hamiltonian)

    if input_type == 'csc':
        # Normalise any sparse format to csc_matrix
        Hamiltonian = csc_matrix(Hamiltonian)
        if Overlap is not None:
            Overlap = csc_matrix(Overlap)
    elif input_type == 'numpy':
        Hamiltonian = np.asarray(Hamiltonian)
        if Overlap is not None:
            Overlap = np.asarray(Overlap)

    # -----------------------------------------------------------------------
    # Full diagonalization
    # -----------------------------------------------------------------------
    if method == "diagonalization":
        if input_type == 'torch':
            # --- PyTorch path (GPU-capable) ---
            if Overlap is not None:
                eigvals, eigvecs = generalized_eigen_torch(Hamiltonian, Overlap)
            else:
                eigvals, eigvecs = torch.linalg.eigh(Hamiltonian)

            if not return_density_matrix:
                if return_eigvals and return_eigvecs:
                    return eigvals, eigvecs
                if return_eigvals:
                    return eigvals
                if return_eigvecs:
                    return eigvecs

            nocc = len(eigvals) // 2
            density_matrix = 2 * eigvecs[:, :nocc] @ eigvecs[:, :nocc].conj().T
            if return_eigvals and return_eigvecs:
                return density_matrix, eigvals, eigvecs
            if return_eigvals:
                return density_matrix, eigvals
            if return_eigvecs:
                return density_matrix, eigvecs
            return density_matrix

        else:
            # --- SciPy path for numpy.ndarray and csc_matrix ---
            H_dense = Hamiltonian.toarray() if input_type == 'csc' else Hamiltonian
            S_dense = None
            if Overlap is not None:
                S_dense = Overlap.toarray() if input_type == 'csc' else Overlap

            eigvals, eigvecs = scipy.linalg.eigh(H_dense, S_dense)

            if not return_density_matrix:
                if return_eigvals and return_eigvecs:
                    return eigvals, eigvecs
                if return_eigvals:
                    return eigvals
                if return_eigvecs:
                    return eigvecs

            nocc = len(eigvals) // 2
            density_matrix_np = 2 * eigvecs[:, :nocc] @ eigvecs[:, :nocc].conj().T

            # Return density matrix in the original sparse format when applicable
            if input_type == 'csc':
                density_matrix = csc_matrix(density_matrix_np)
            else:
                density_matrix = density_matrix_np

            if return_eigvals and return_eigvecs:
                return density_matrix, eigvals, eigvecs
            if return_eigvals:
                return density_matrix, eigvals
            if return_eigvecs:
                return density_matrix, eigvecs
            return density_matrix

    # -----------------------------------------------------------------------
    # Sparse (iterative) diagonalization — ARPACK / eigsh
    # -----------------------------------------------------------------------
    elif method == "sparse_diagonalization":
        print("Sparse diagonalization is a linear scaling method, but is only implemented for CPUs")

        # eigsh accepts sparse matrices natively; convert torch tensors to numpy
        if input_type == 'torch':
            H_sp = Hamiltonian.cpu().numpy()
            S_sp = Overlap.cpu().numpy() if Overlap is not None else None
        else:
            H_sp = np.squeeze(np.asarray(Hamiltonian.toarray() if issparse(Hamiltonian) else Hamiltonian))
            S_sp = None
            if Overlap is not None:
                S_sp = np.squeeze(np.asarray(Overlap.toarray() if issparse(Overlap) else Overlap))

        if S_sp is not None:
            eigvals, eigvecs = eigsh(H_sp, k=nbands, sigma=fermi_level, M=S_sp, which=which, **kwargs)
        else:
            eigvals, eigvecs = eigsh(H_sp, k=nbands, sigma=fermi_level, which=which, **kwargs)
        return eigvals, eigvecs

    # -----------------------------------------------------------------------
    # PEXSI (Fermi-operator expansion, MPI parallel)
    # -----------------------------------------------------------------------
    elif method == "pexsi":
        if Overlap is not None:
            raise ValueError("Overlap not supported for PEXSI")
        if return_eigvals or return_eigvecs:
            raise ValueError(
                "return_eigvals/eigvecs not supported for PEXSI. "
                "Only return_density_matrix=True is supported."
            )
        dm_vals, rowind, colind, _mu = get_density_matrix_pexsi(
            Hamiltonian, temperature=temperature, numPoles=numPoles
        )
        dm_coo = coo_matrix(
            (dm_vals, (rowind, colind)),
            shape=(Hamiltonian.shape[0], Hamiltonian.shape[1]),
        )
        if input_type == 'csc':
            return dm_coo.tocsc()
        elif input_type == 'numpy':
            return dm_coo.toarray()
        else:
            return torch.from_numpy(dm_coo.toarray())

    else:
        raise ValueError(f"Invalid method '{method}'. Choose from 'diagonalization', 'sparse_diagonalization', or 'pexsi'.")

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

    