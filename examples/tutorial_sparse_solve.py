import numpy as np
import torch
from pythtb import TBModel, Lattice
import matplotlib.pyplot as plt
from tb_solve import Solve_Hamiltonian
from tb_solve.utils import disentangle_bands

def create_graphene_hamiltonian(supercell_size=[5, 5]):
    """
    Creates a graphene Hamiltonian using PythTB.
    Based on: https://pythtb.readthedocs.io/en/latest/tutorials/graphene.html
    """
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

    sc_model = my_model.make_supercell([[supercell_size[0], 1], [1, supercell_size[1]]]) #, to_home=True)

    return sc_model

if __name__=="__main__":
    nbands = 8
    efermi = 0.0 
    tb_model = create_graphene_hamiltonian()
    
    k_nodes = [[0, 0], [2 / 3, 1 / 3], [1 / 2, 1/2], [0, 0]]
    k_node_labels = (r"$\Gamma $", r"$K$", r"$M$", r"$\Gamma $")
    nk = 121
    k_path, k_dist, k_node_dist = tb_model.k_path(k_nodes, nk)
    eigvals_sparse = []
    eigvecs_sparse = []
    eigvals_dense = []

    for k in range(len(k_path)):
        ham_k = tb_model.hamiltonian(k_pts=k_path[k])
        H_torch = torch.from_numpy(ham_k).to(torch.complex128)

        # Sparse (iterative) solve for a subset of bands
        e, psi = Solve_Hamiltonian(
            H_torch,
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
            H_torch,
            method="diagonalization",
            return_eigvals=True,
            return_density_matrix=False
        )
        e_dense = np.array(e_dense)
        eigvals_dense.append(e_dense)

    eigvals_sparse = np.squeeze(np.stack(eigvals_sparse))                  # (nk, nbands)
    eigvecs_sparse = np.squeeze(np.stack(eigvecs_sparse))                  # (nk, norb, nbands)
    eigvals_dense = np.squeeze(np.stack(eigvals_dense))                  # (nk, nbands)

    #useful to disentangle bands and get rid of incomplete bands when using sparse solver (but not necessary)
    e_disentangled_sparse, psi_disentangled_sparse = disentangle_bands(eigvals_sparse, eigvecs_sparse)

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

    