Solver Utilities Tutorial
=========================

This tutorial demonstrates the helper functions in ``tb_solve.utils`` designed to assist in choosing the right solver and parameters.

Key utilities covered:

* ``Solver_helper``: Estimates computation time for various solvers.
* ``Get_optimal_solver``: Automatically selects the fastest method.
* ``Converge_Solver_settings``: Finds stable parameters (e.g., number of moments).

First, import the necessary library and create the graphene Hamiltonian.

.. code-block:: python

   import torch
   import numpy as np
   from pythtb import TBModel, Lattice
   from tb_solve.utils import Solver_helper, Get_optimal_solver, Converge_Solver_settings
   from tb_solve import Solve_Hamiltonian

   # Helper function to create a Hamiltonian (same as in the graphene tutorial)
   supercell_size=[5, 5]
   lat_vecs = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
   orb_vecs = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]
   lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])
   my_model = TBModel(lat)
   my_model.set_onsite([-0.0, 0.0])
   my_model.set_hop(-2.7, 0, 1, [0, 0])
   my_model.set_hop(-2.7, 1, 0, [1, 0])
   my_model.set_hop(-2.7, 1, 0, [0, 1])
   sc_model = my_model.make_supercell([[supercell_size[0], 1], [1, supercell_size[1]]])

Next, we can explore the different utilities.

.. code-block:: python
    
    # 1. Create a large Hamiltonian
    print("\n1. Creating a large Graphene Supercell (10x10)...")
    # Get Hamiltonian at Gamma point
    H= sc_model.hamiltonian()
    
    # Ensure we use CPU for this tutorial to be safe, or detect device
    if torch.cuda.is_available():
        H = H.cuda()
        print("Using GPU.")
    else:
        print("Using CPU.")
        
    print(f"Hamiltonian shape: {H.shape}")

We can estimate the computation time for different solvers.

.. code-block:: python

    t_diag = Solver_helper(H, method="diagonalization")
    print(f"Diagonalization time: {t_diag:.2e} s")
    t_foe = Solver_helper(H, method="fermi_operator_expansion", n_moments=100)
    print(f"Fermi Operator Expansion time: {t_foe:.2e} s")
    t_sparse = Solver_helper(H, method="sparse_diagonalization", nbands=20)
    print(f"Sparse Diagonalization time: {t_sparse:.2e} s")
   
We can automatically select the optimal solver, for a given hamiltonian and settings.

.. code-block:: python

    best_method, min_time = Get_optimal_solver(H, n_moments=100, nbands=20, max_iterations=50)
    print(f"Optimal solver: {best_method} (Est. time: {min_time:.2e} s)")

We can also converge the solver settings for different methods.

.. code-block:: python
    
    optimal_moments = Converge_Solver_settings(
        H, 
        method="fermi_operator_expansion", 
        variable="n_moments",
        start_value=50, 
        end_value=300, 
        step=50, 
        tolerance=1e-4,
        kbT=0.01 # Fixed argument
    )
    
    print(f"Converged n_moments: {optimal_moments}")
    
