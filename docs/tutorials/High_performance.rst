High Performance Solvers Tutorial
================================

This tutorial demonstrates how to use the high performance solvers provided by ``tb_solve``:

1. **PEXSI** (CPU only, MPI parallelization over real space pole inversions)(O(N^(d+1)/2) scaling where d is the number of dimensions, zero and finite temperature, metals and insulators) can be parallelized up to 10,000 cores reliably. Note that this solver is only implemented for cpu currently and Gamma point calculations. While this seems like a limitation for metals, it is not if the system is large enough. A large system in real space will converge to the same energy as a small system with a dense k-point mesh.

2. **Direct Diagonalization** (GPU or CPU, MPI parallelization over K-points)(O(N^3) scaling, zero and finite temperature, metals and insulators)

3. **Sparse Diagonalization** (CPU, MPI parallelization over K-points)(O(mN) scaling, where m is the number of bands. zero and finite temperature, metals and insulators)

.. note::

   ``Solve_Hamiltonian`` automatically selects the underlying linear-algebra
   library based on the type of the input matrix:

   * **torch.Tensor** → ``torch.linalg.eigh`` / :func:`~tb_solve.Solver.generalized_eigen_torch` (GPU-capable).
   * **numpy.ndarray** → ``scipy.linalg.eigh`` (CPU only).
   * **csc_matrix** (or any SciPy sparse) → ``scipy.linalg.eigh`` on the dense
     view; the returned density matrix is a ``csc_matrix``.

   The output type always matches the input type.  See the
   :doc:`/usage/index` page for concrete examples.

`PythTB <https://pythtb.org/>`_ is used to generate the Hamiltonian, but these solvers are general to any tight-binding model.

First, import the necessary library and create a Hamiltonian. This hamiltnonian will be used for all solver examples.

.. code-block:: python

   import numpy as np
   import torch
   from pythtb import TBModel, Lattice
   from tb_solve import Solve_Hamiltonian
   from tb_solve.utils import disentangle_bands
   from scipy.sparse import csr_matrix
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

Parallelization using PEXSI solver. PEXSI is parallelized over real space pole inversions. If you can construct the Hamiltonian in a sparse manner, you can pass the csr matrix to the solver. 
Otherwise, you can pass the dense matrix to the solver and it will convert it to a sparse matrix internally.

.. code-block:: python
    
    # Generate Hamiltonian at a specific k-point (or mesh)
    # Using a mesh as requested, but we'll select one k-point to solve
    ham_k = my_model.hamiltonian(k_pts=[0,0])
    
    # Select one Hamiltonian from the mesh (e.g., the first one at Gamma)
    # ham_k has shape (N_k, N_orb, N_orb)
    H = ham_k[0]
    H_csr = csr_matrix(H)
    
    # Convert to PyTorch tensor
    # Ensure it's complex if necessary, usually PythTB returns complex128

    print(f"\nSolving Hamiltonian (Shape: {H.shape})")
    print("-" * 40)

    # Note: This works for zero temperature and finite temperature. insulators and metals
    print("\nMethod: PEXSI")
    dm_pexsi = Solve_Hamiltonian(
        H_csr, 
        method="PEXSI",
    )
    print("Density Matrix:\n", dm_pexsi)

Run the script with the following command and replace $(nproc) with the number of cores you want to use:

.. code-block:: bash

   mpirun -n $(nproc) python pexsi_hpc_example.py 

Example parallelization over K-points using the Direct Diagonalization or sparse diagonalization solvers.
If there is a GPU in the environment, the Direct Diagonalization solver will automatically use it when
the Hamiltonian is passed as a ``torch.Tensor``.  Pass a ``numpy.ndarray`` or ``csc_matrix`` to stay
on the CPU and use ``scipy.linalg.eigh`` instead.

.. code-block:: python
    
    from mpi4py import MPI

    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. Define the mesh only on the root process (rank 0)
    if rank == 0:
        k_mesh = my_model.k_uniform_mesh([10, 10])
        # Split the k_mesh into sub-arrays for each worker
        chunks = np.array_split(k_mesh, size)
    else:
        k_mesh = None
        chunks = None

    # 2. Scatter the work: Rank 0 sends a chunk to every other process
    local_k_mesh = comm.scatter(chunks, root=0)

    # 3. Each process computes its local list
    local_dm_list = []
    for k_pt in local_k_mesh:
        # PythTB returns a numpy array; pass it directly to stay on the CPU
        # (scipy.linalg.eigh path), or wrap in torch.tensor() to use the GPU.
        H = my_model.hamiltonian(k_pts=k_pt)   # numpy.ndarray
        dm_direct_k = Solve_Hamiltonian(
            H, 
            method="diagonalization",
        )
        local_dm_list.append(dm_direct_k)

    # 4. Gather all local results back to the root process
    all_results = comm.gather(local_dm_list, root=0)

    # 5. Root process flattens the list of lists into the final result
    if rank == 0:
        dm_direct_k_list = [item for sublist in all_results for item in sublist]
    print(f"Calculation complete. Total density matrices: {len(dm_direct_k_list)}")

Run the script with the following command and replace $(nproc) with the number of cores you want to use:

.. code-block:: bash

   mpirun -n $(nproc) python kpt_parallelization_direct_diagonalization_hpc_example.py 