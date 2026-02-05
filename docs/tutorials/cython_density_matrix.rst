Density matrix minimization (Cython)
====================================

This tutorial shows how to:

- install TB_solve with the Cython-accelerated density matrix minimizer
- build a simple graphene Hamiltonian with `PythTB <https://pythtb.org/>`_
- call ``density_matrix_minimization_cy`` with an easy, script-ready interface

Prerequisites
-------------

- Python 3.8+
- `PyTorch <https://pytorch.org/>`_ (CPU is fine for this demo)
- `Cython <https://cython.org/>`_ (to compile the extension)
- `pythtb <https://github.com/danielkhuszp/pytb>`_ for generating the Hamiltonian

Installation (from source)
--------------------------

.. code-block:: bash

   # 1) Clone the repo
   git clone https://github.com/dpalmer/TB_solve.git
   cd TB_solve

   # 2) Install the package
   pip install ./

   # 3) Install Cython for accelerated performance (optional but recommended)
   pip install cython

The Cython extension will compile automatically on first import if Cython is
installed. Without Cython, the function falls back to the pure Python
implementation.

Quick start: graphene Hamiltonian + Cython minimizer
----------------------------------------------------

The snippet below mirrors the graphene example used elsewhere in the docs,
but calls the new Cython routine. It keeps the interface simple: create
the Hamiltonian with PythTB, convert to Torch, and call
``density_matrix_minimization_cy``.

.. code-block:: python

   

Notes and tips
--------------

- The Cython routine runs on CPU; GPU inputs are copied to CPU internally
  and returned on the original device/dtype.
- For repeated use in scripts, pre-compile with ``cythonize`` once. For
  experimentation, ``import pyximport; pyximport.install(language_level=3)``
  before the import also works.
- The function is a drop-in replacement for
  ``tb_solve.Solver.density_matrix_minimization``—same inputs and outputs,
  but with reduced Python overhead.
