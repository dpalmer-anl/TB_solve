SparseSolver Tutorial
=========================

This tutorial demonstrates how to use the sparse solver to solve the Hamiltonian of a graphene supercell. 
This method is a wrapper for scipy.sparse.linalg.eigsh, which uses ARPACK's SSEUPD and DSEUPD functions 
which use the Implicitly Restarted Lanczos Method to find the eigenvalues and eigenvectors. The default for this function
is to find the eigenvalues and eigenvectors near the Fermi level. 

**Note** this function only works on CPU's, but because it is sparse, it still can be used for very large systems.

First, import the necessary library and create the graphene Hamiltonian.

.. code-block:: python

    import numpy as np
    import torch
    from pythtb import TBModel, Lattice
    import matplotlib.pyplot as plt
    from tb_solve import Solve_Hamiltonian
    from tb_solve.utils import disentangle_bands

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
    t = -1.0

    # set on-site energies
    my_model.set_onsite([-delta, delta])
    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])
    my_model.set_hop(t, 0, 1, [0, 0])
    my_model.set_hop(t, 1, 0, [1, 0])
    my_model.set_hop(t, 1, 0, [0, 1])

    tb_model = my_model.make_supercell([[supercell_size[0], 1], [1, supercell_size[1]]])

Next, we need to create the k-path for the band structure.

.. code-block:: python

    k_nodes = [[0, 0], [2 / 3, 1 / 3], [1 / 2, 1/2], [0, 0]]
    k_node_labels = (r"$\Gamma $", r"$K$", r"$M$", r"$\Gamma $")
    nk = 121
    k_path, k_dist, k_node_dist = tb_model.k_path(k_nodes, nk)

Now, we can solve the Hamiltonian for the k-path. Here use the sparse and dense solvers to compare the results. 
Note that the sparse solver needs an estimate of the fermi level to find bands near the fermi level. 
The average trace of the Hamiltonian is a good estimate of the fermi level.

.. code-block:: python

    nbands = 8
    efermi = np.trace(ham_k) / ham_k.shape[0] # this is zero for our hamiltonian

    eigvals_sparse = []
    eigvecs_sparse = []
    eigvals_dense = []

    for k in range(len(k_path)):
        ham_k = tb_model.hamiltonian(k_pts=k_path[k])

        # Sparse (iterative) solve for a subset of bands
        e, psi = Solve_Hamiltonian(
            ham_k,
            method="sparse_diagonalization",
            nbands=nbands,
            return_eigvals=True,
            return_eigvecs=True,
            return_density_matrix=False,
            fermi_level=efermi
        )
        # Ensure numpy arrays
        e = np.array(e)
        psi = np.array(psi)
        eigvals_sparse.append(e)
        eigvecs_sparse.append(psi)

        # Dense solve for reference (no density matrix needed)
        e_dense = Solve_Hamiltonian(
            ham_k,
            method="diagonalization",
            return_eigvals=True,
            return_density_matrix=False
        )
        e_dense = np.array(e_dense)
        eigvals_dense.append(e_dense)

    eigvals_sparse = np.squeeze(np.stack(eigvals_sparse))                  # (nk, nbands)
    eigvecs_sparse = np.squeeze(np.stack(eigvecs_sparse))                  # (nk, norb, nbands)
    eigvals_dense = np.squeeze(np.stack(eigvals_dense))                  # (nk, nbands)

If we want to disentangle the bands, we can use the disentangle_bands function. 
The sparse solver will return incomplete bands if the bands cross over, so we need to disentangle them.

.. code-block:: python

    e_disentangled_sparse, psi_disentangled_sparse = disentangle_bands(eigvals_sparse, eigvecs_sparse)

Finally, we can plot the band structure.

.. code-block:: python

    # plot the bands
    fig, ax = plt.subplots()

    ax.set_xlim(k_node_dist[0], k_node_dist[-1])
    ax.set_xticks(k_node_dist)
    ax.set_xticklabels(k_node_labels)

    for n in range(len(k_node_dist)):
        ax.axvline(x=k_node_dist[n], linewidth=0.5, color="k")

    ax.set_title("Graphene band structure from sparse diagonalization")
    ax.set_xlabel("Path in k-space")
    ax.set_ylabel("Band energy")

    # plot bands
    nbands_sparse = e_disentangled_sparse.shape[1]
    nbands_dense = eigvals_dense.shape[1]

    # take nbands around fermi level from dense bands 
    start =  nbands_dense//2 - nbands_sparse//2 
    stop =  start + nbands_sparse

    eigvals_dense_sel = eigvals_dense[:, start:stop]
    nbands_plot = eigvals_dense_sel.shape[1]

    for b in range(nbands_sparse):
        if b == 0:
            ax.scatter(k_dist, e_disentangled_sparse[:, b], c="b", label="Sparse")
            ax.plot(k_dist, eigvals_dense_sel[:, b], c="r", label="Dense",linestyle="--")
        else:
            ax.scatter(k_dist, e_disentangled_sparse[:, b], c="b")
            ax.plot(k_dist, eigvals_dense_sel[:, b], c="r",linestyle="--")
    ax.legend()

    plt.show()

The results are shown in the figure below.

.. figure:: ../images/sparse_solver_graphene.png
    :alt: Sparse solver results
    :align: center
    :width: 75%
