Graphene Density Matrix Tutorial
================================

This tutorial demonstrates how to calculate the density matrix of graphene using three different solvers provided by ``tb_solve``:

1. **Diagonalization** (Standard method)
2. **PEXSI** (O(N^(d+1)/2) scaling where d is the number of dimensions, zero and finite temperature, metals and insulators)

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

Next, we can solve the Hamiltonian for the density matrix. We will use the diagonalization, and PEXSI solvers to compare the results.

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

    # 2. PEXSI
    # Note: This works for zero temperature and finite temperature. insulators and metals
    # PEXSI will only return the elements of the density matrix where H is non-zero.
    # This is useful, since these are the only elements we need to compute the total band energy and the forces on the atoms.
    print("\nMethod: PEXSI")
    dm_pexsi = Solve_Hamiltonian(
        H, 
        method="PEXSI",
    )
    print("Density Matrix:\n", dm_pexsi)


