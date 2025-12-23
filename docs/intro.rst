Introduction
============

TB_solve is a Python library designed to efficiently solve tight-binding models. The core problem we address is the generalized eigenvalue problem:

.. math::

   H |\psi\rangle = \epsilon S |\psi\rangle

where:
* :math:`H` is the Hamiltonian matrix representing the system's energy.
* :math:`S` is the Overlap matrix (which is the identity matrix :math:`I` for orthogonal bases).
* :math:`|\psi\rangle` are the eigenvectors (wave functions).
* :math:`\epsilon` are the eigenvalues (energies).

For many physical applications, we are interested in two main quantities:

1. **Band Structure**: The eigenvalues :math:`\epsilon(\mathbf{k})` as a function of the wave vector :math:`\mathbf{k}`.
2. **Density Matrix**: defined as:

.. math::

   \rho = \sum_{i=1}^{N_{occ}} |\psi_{i}\rangle\langle\psi_{i}|

.. math::

   \rho_{\mu\nu} = \sum_{i=1}^{N_{occ}} c^*_{\mu,i} c_{\nu,i} 

where :math:`c_{\mu,i}` are the coefficients of the :math:`i`-th occupied eigenvector :math:`\psi_{i}`.

Solver Methods
--------------

TB_solve provides multiple solver methods tailored for different system sizes, sparsity patterns, and physical requirements. 
TB_solve is written in PyTorch and supports both CPU and GPU computation for most methods (including intel gpu's!). 
tb_solve.Solver() will automatically detect available devices and prioritize using GPU if available. 
The following reference contains all details for each solver method listed here, (amongst others): https://doi.org/10.1103/RevModPhys.71.1085

1. **Diagonalization** (``method="diagonalization"``)
   
   * **Description**: Performs a full dense diagonalization of the Hamiltonian.
   * **Best for**: Small to medium-sized systems (:math:`N < 20,000`) where all eigenvalues/eigenvectors are needed. Supports Generalized Eigenvalue Problems (:math:`S \neq I`).
   * **Limitations**: Scaling is cubic :math:`O(N^3)`, making it computationally prohibitive for very large systems. 

2. **Sparse Diagonalization** (``method="sparse_diagonalization"``)
   
   * **Description**: Uses :math:`O(N)` sparse iterative methods (Lanczos/Arnoldi via ARPACK) to find a subset of eigenvalues/eigenvectors, default is to find bands near the fermi level.
   * **Best for**: Large sparse systems where only a few bands (``nbands``) are required. Useful for finding band structure near the fermi level.
   * **Limitations**: CPU-only implementation currently. Does not efficiently compute the full density matrix. 

3. **Fermi Operator Expansion** (``method="fermi_operator_expansion"``)
   
   * **Description**: A linear-scaling :math:`O(N)` method (for sparse matrices) that approximates the density matrix using a Chebyshev polynomial expansion of the Fermi-Dirac distribution.
   * **Feature**: Includes **Jackson kernel damping**, which suppresses Gibbs oscillations, making it robust and accurate even for metallic systems at low temperatures.
   * **Best for**: Very large systems (:math:`N > 10^5`), calculating local observables, and finite-temperature calculations.
   * **Limitations**: Does not provide individual eigenvalues. Does not currently work with Overlap matrices (:math:`S \neq I`). Requires user to determine convergence for number of moments (``n_moments``).(~200 is usually enough)

4. **Density Matrix Purification** (``method="density_matrix_purification"``)
   
   * **Description**: An iterative method that converges the density matrix to the canonical ensemble at zero temperature (:math:`T=0`) by enforcing idempotency (:math:`P^2 = P`).
   * **Best for**: Very large systems (:math:`N > 10^5`), or as an alternative linear-scaling approach for ground state properties.
   * **Limitations**: Does not provide individual eigenvalues. Does not currently work with Overlap matrices (:math:`S \neq I`). 

