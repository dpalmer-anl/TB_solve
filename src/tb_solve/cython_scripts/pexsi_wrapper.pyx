#I some how need to call PEXSI from python and make sure mpi tasks are set up correctly
#conda install conda-forge::parmetis
#conda install conda-forge::superlu_dis
#conda install conda-forge::pexsi

# pexsi_wrapper.pyx
import numpy as np
cimport numpy as cnp
from mpi4py import MPI
from mpi4py cimport MPI
from mpi4py cimport libmpi as mpi
from libc.stdint cimport intptr_t

# Import the PEXSI C-interface headers
cdef extern from "c_pexsi_interface.h":
    ctypedef intptr_t PPEXSIPlan

    ctypedef struct PPEXSIOptions:
        double spin
        double temperature
        double gap
        double deltaE
        int numPole
        int isInertiaCount
        int maxPEXSIIter
        double muMin0
        double muMax0
        double mu0
        double muInertiaTolerance
        double muInertiaExpansion
        double muPEXSISafeGuard
        double numElectronPEXSITolerance
        int matrixType
        int isSymbolicFactorize
        int isConstructCommPattern
        int solver
        int symmetricStorage
        int ordering
        int rowOrdering
        int npSymbFact
        int symmetric
        int transpose
        int method
        int nPoints
        int verbosity
        int iFLAG

    void PPEXSISetDefaultOptions(PPEXSIOptions* options)

    PPEXSIPlan PPEXSIPlanInitialize(
        mpi.MPI_Comm comm, int nprow, int npcol,
        int outputFileIndex, int* info)

    void PPEXSILoadRealHSMatrix(
        PPEXSIPlan plan, PPEXSIOptions options, int nrows,
        int nnz, int nnzLocal, int numColLocal,
        int* colptrLocal, int* rowindLocal, double* HnzvalLocal,
        int isSIdentity, double* SnzvalLocal, int* info)

    void PPEXSILoadComplexHSMatrix(
        PPEXSIPlan plan, PPEXSIOptions options, int nrows,
        int nnz, int nnzLocal, int numColLocal,
        int* colptrLocal, int* rowindLocal, double* HnzvalLocal,
        int isSIdentity, double* SnzvalLocal, int* info)

    void PPEXSIDFTDriver2(
        PPEXSIPlan plan, PPEXSIOptions* options, double numElectronExact,
        double* muPEXSI, double* numElectronPEXSI,
        int* numTotalInertiaIter, int* info)

    void PPEXSIRetrieveRealDM(PPEXSIPlan plan, double* DMnzvalLocal,
                              double* totalEnergyH, int* info)

    void PPEXSIRetrieveComplexDM(PPEXSIPlan plan, double* DMnzvalLocal,
                                 double* totalEnergyH, int* info)

    void PPEXSIPlanFinalize(PPEXSIPlan plan, int* info)

def run_pexsi(comm, int nprow, int npcol, int nrows, int nnz,
                  int[:] colptrLocal, int[:] rowindLocal,
                  HnzvalLocal, double num_electrons):
    
    cdef mpi.MPI_Comm c_comm = (<MPI.Comm>comm).ob_mpi
    cdef int mpirank = comm.Get_rank()
    cdef int info = 0
    cdef int outputFileIndex = -1
    if (mpirank % (nprow * npcol) == 0):
        outputFileIndex = mpirank // (nprow * npcol)
    # 1. Initialize Plan
    cdef PPEXSIPlan plan = PPEXSIPlanInitialize(c_comm, nprow, npcol, outputFileIndex, &info)

    # 2. Set Options
    cdef PPEXSIOptions options
    PPEXSISetDefaultOptions(&options)
    options.temperature = 0.019
    options.method = 2
    options.numPole = 20
    # For single-process runs, ensure point parallelization is 1.
    options.nPoints = 1
    # Avoid zero-diagonal failures in SuperLU_DIST symbolic factorization.
    options.rowOrdering = 1
    options.symmetric = 1
    options.symmetricStorage = 1

    # 3. Load Matrix (Assuming S is Identity)
    cdef int nnzLocal = rowindLocal.shape[0]
    cdef int numColLocal = colptrLocal.shape[0] - 1
    cdef bint is_complex = np.iscomplexobj(HnzvalLocal)
    cdef double[:] HnzvalLocal_real
    cdef object H_realview
    if is_complex:
        H_complex = np.ascontiguousarray(HnzvalLocal, dtype=np.complex128)
        if np.max(np.abs(np.imag(H_complex))) > 1e-12:
            raise ValueError(
                "PEXSI build only supports real matrices (matrixType=0). "
                "Provide a real Hamiltonian or install a complex-enabled PEXSI."
            )
        H_real = np.ascontiguousarray(np.real(H_complex), dtype=np.float64)
        HnzvalLocal_real = H_real
    else:
        H_real = np.ascontiguousarray(HnzvalLocal, dtype=np.float64)
        HnzvalLocal_real = H_real

    options.matrixType = 0
    PPEXSILoadRealHSMatrix(
        plan, options, nrows, nnz, nnzLocal, numColLocal,
        &colptrLocal[0], &rowindLocal[0], &HnzvalLocal_real[0],
        1, NULL, &info)

    # 4. Run Driver
    cdef double muPEXSI, numElectronPEXSI
    cdef int numTotalInertiaIter
    PPEXSIDFTDriver2(plan, &options, num_electrons, &muPEXSI,
                     &numElectronPEXSI, &numTotalInertiaIter, &info)

    # 5. Retrieve Density Matrix (Local values)
    cdef double totalEnergyH
    cdef double[:] dm_view
    cdef object dm_real
    dm_nzval_local = np.zeros(nnzLocal, dtype=np.float64)
    dm_view = dm_nzval_local
    if mpirank < nprow * npcol:
        PPEXSIRetrieveRealDM(plan, &dm_view[0], &totalEnergyH, &info)

    # 6. Cleanup
    PPEXSIPlanFinalize(plan, &info)

    colptr_np = np.asarray(colptrLocal, dtype=np.int64)
    rowind_np = np.asarray(rowindLocal, dtype=np.int64)
    nnz_local = rowind_np.shape[0]
    colind_np = np.empty(nnz_local, dtype=np.int64)
    for j in range(numColLocal):
        start = colptr_np[j] - 1
        end = colptr_np[j + 1] - 1
        if end > start:
            colind_np[start:end] = j

    # Convert from 1-based to 0-based indices for Python users.
    rowind_np = rowind_np - 1
    colind_np = colind_np.astype(np.int64, copy=False)

    return dm_nzval_local, rowind_np, colind_np, muPEXSI