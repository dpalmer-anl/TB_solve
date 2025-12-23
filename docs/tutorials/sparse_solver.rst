SparseSolver Tutorial
=========================

This tutorial demonstrates how to use the sparse solver to solve the Hamiltonian of a graphene supercell. 
This method is a wrapper for scipy.sparse.linalg.eigsh, which uses ARPACK's SSEUPD and DSEUPD functions 
which use the Implicitly Restarted Lanczos Method to find the eigenvalues and eigenvectors. The default for this function
is to find the eigenvalues and eigenvectors near the Fermi level. **Note** this function only works on CPU's,
 but because it is sparse, it still can be used for very large systems.

.. literalinclude:: ../../examples/tutorial_sparse_solve.py
   :language: python
   :linenos:
   :caption: tutorial_sparse_solve.py