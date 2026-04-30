Usage
=====

Installation
------------
Install dependencies in a conda environment:

.. code-block:: bash

   pip install numpy scipy mpi4py
   conda install conda-forge::parmetis
   conda install conda-forge::superlu_dis
   conda install conda-forge::pexsi

To install TB_solve, use pip:

.. code-block:: bash

   pip install tb_solve

Or install from source:

.. code-block:: bash

   git clone https://github.com/dpalmer-anl/TB_solve.git
   cd TB_solve
   pip install .

Basic Usage
-----------

``Solve_Hamiltonian`` accepts ``torch.Tensor``, ``numpy.ndarray``, or any
SciPy sparse matrix (e.g. ``csc_matrix``).  The returned arrays always match
the input type.

**torch.Tensor input** — uses GPU-accelerated PyTorch routines:

.. code-block:: python

   import torch
   from tb_solve import Solve_Hamiltonian

   N = 10
   H = torch.randn(N, N, dtype=torch.complex128)
   H = (H + H.conj().T) / 2          # make Hermitian

   rho = Solve_Hamiltonian(H, method="diagonalization")
   print(type(rho))   # <class 'torch.Tensor'>
   print(rho)

**numpy.ndarray input** — dispatches to ``scipy.linalg.eigh``:

.. code-block:: python

   import numpy as np
   from tb_solve import Solve_Hamiltonian

   N = 10
   H = np.random.randn(N, N) + 1j * np.random.randn(N, N)
   H = (H + H.conj().T) / 2          # make Hermitian

   rho = Solve_Hamiltonian(H, method="diagonalization")
   print(type(rho))   # <class 'numpy.ndarray'>

   # Request eigenvalues and eigenvectors as well
   rho, eigvals, eigvecs = Solve_Hamiltonian(
       H, method="diagonalization",
       return_eigvals=True, return_eigvecs=True,
   )
   print(eigvals)     # numpy.ndarray

**scipy.sparse.csc_matrix input** — diagonalises the dense view internally via
``scipy.linalg.eigh``; the density matrix is returned as a ``csc_matrix``:

.. code-block:: python

   import numpy as np
   from scipy.sparse import csc_matrix
   from tb_solve import Solve_Hamiltonian

   N = 10
   H_dense = np.random.randn(N, N) + 1j * np.random.randn(N, N)
   H_dense = (H_dense + H_dense.conj().T) / 2
   H = csc_matrix(H_dense)

   rho = Solve_Hamiltonian(H, method="diagonalization")
   print(type(rho))   # <class 'scipy.sparse.csc_matrix'>

**Generalized eigenvalue problem** (with overlap matrix) — pass the same type
for both ``Hamiltonian`` and ``Overlap``.  PyTorch input uses
:func:`~tb_solve.Solver.generalized_eigen_torch`; numpy / sparse input uses
``scipy.linalg.eigh(H, S)``:

.. code-block:: python

   import numpy as np
   from tb_solve import Solve_Hamiltonian

   N = 10
   H = np.random.randn(N, N) + 1j * np.random.randn(N, N)
   H = (H + H.conj().T) / 2
   S = np.eye(N) + 0.01 * (np.random.randn(N, N) + np.random.randn(N, N).T)

   rho = Solve_Hamiltonian(H, Overlap=S, method="diagonalization")

API Reference
-------------

See the :doc:`/modules` section for detailed API documentation.

