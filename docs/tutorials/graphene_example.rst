Graphene Density Matrix Tutorial
================================

This tutorial demonstrates how to calculate the density matrix of graphene using three different solvers provided by ``tb_solve``:

1. **Diagonalization** (Standard method)
2. **Fermi Operator Expansion** (Linear scaling, finite temperature)
3. **Density Matrix Purification** (Linear scaling, zero temperature)

`PythTB <https://pythtb.org/>`_ is used to generate the Hamiltonian, but these solvers are general to any tight-binding model.

First, import the necessary library and create the graphene Hamiltonian.

.. code-block:: python

   import numpy as np
   import torch
   from pythtb import TBModel, Lattice
   from tb_solve import Solve_Hamiltonian
   from tb_solve.utils import disentangle_bands

   #Creates a graphene Hamiltonian using PythTB.
   #Based on: https://pythtb.readthedocs.io/en/latest/tutorials/graphene.html
   supercell_size = [5, 5]
   # define lattice vectors
   lat_vecs = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
   # define coordinates of orbitals
   orb_vecs = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]
   
   # Create lattice object (periodic in all directions)
   lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])
   
   # make two dimensional tight-binding graphene model
   my_model = TBModel(lat)

   # set model parameters
   delta = 0.0
   t = -2.7

   # set on-site energies
   my_model.set_onsite([-delta, delta])
   # set hoppings (one for each connected pair of orbitals)
   # (amplitude, i, j, [lattice vector to cell containing j])
   my_model.set_hop(t, 0, 1, [0, 0])
   my_model.set_hop(t, 1, 0, [1, 0])
   my_model.set_hop(t, 1, 0, [0, 1])

   my_model = my_model.make_supercell([[supercell_size[0], 1], [1, supercell_size[1]]])

Next, we can solve the Hamiltonian for the density matrix. We will use the diagonalization, Fermi Operator Expansion, and Density Matrix Purification solvers to compare the results.

.. code-block:: python
    
    # Generate Hamiltonian at a specific k-point (or mesh)
    # Using a mesh as requested, but we'll select one k-point to solve
    # so we don't print 100 density matrices.
    print("Generating Hamiltonian from k-mesh...")
    k_points = my_model.k_uniform_mesh([10, 10])
    ham_k = my_model.hamiltonian(k_pts=k_points)
    
    # Select one Hamiltonian from the mesh (e.g., the first one at Gamma)
    # ham_k has shape (N_k, N_orb, N_orb)
    H = ham_k[0]
    
    # Convert to PyTorch tensor
    # Ensure it's complex if necessary, usually PythTB returns complex128

    print(f"\nSolving Hamiltonian (Shape: {H.shape})")
    print("-" * 40)

    # 1. Diagonalization
    print("\nMethod: Diagonalization")
    dm_diag = Solve_Hamiltonian(H, method="diagonalization")
    print("Density Matrix:\n", dm_diag)

    # 2. Fermi Operator Expansion
    # Note: This is a finite temperature method. To approximate zero temperature, set kbT to a small value.
    # For Graphene (gapless/semimetal), results depend on T and chemical potential.
    # Solve_Hamiltonian estimates mu = trace(H)/N which is 0 for this model.
    print("\nMethod: Fermi Operator Expansion")
    # Using a small temperature and sufficient moments
    dm_foe = Solve_Hamiltonian(
        H, 
        method="fermi_operator_expansion", 
        kbT=0.01, 
        n_moments=200
    )
    print("Density Matrix:\n", dm_foe)

    # 3. Density Matrix Purification
    # Note: This is a T=0 method (canonical purification).
    print("\nMethod: Density Matrix Purification")
    dm_purification = Solve_Hamiltonian(
        H, 
        method="density_matrix_purification",
        max_iterations=50
    )
    print("Density Matrix:\n", dm_purification)

