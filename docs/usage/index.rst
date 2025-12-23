Usage
=====

Installation
------------

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

Here is a simple example of how to use the library:

.. code-block:: python

   import torch
   from tb_solve import Solve_Hamiltonian

   # Create a random Hamiltonian (hermitian)
   N = 10
   H = torch.randn(N, N, dtype=torch.complex128)
   H = (H + H.T.conj()) / 2

   # Solve for density matrix
   rho = Solve_Hamiltonian(H, method="diagonalization")
   print(rho)

API Reference
-------------

See the :doc:`/modules` section for detailed API documentation.

