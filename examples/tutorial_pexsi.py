import numpy as np
import torch
from scipy.sparse import coo_matrix
from pythtb import TBModel, Lattice
from tb_solve import get_density_matrix_pexsi, Solve_Hamiltonian

def graphene_model(supercell_size=(4, 4)):
    lat_vecs = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb_vecs = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])
    model = TBModel(lat)

    delta = 0.0
    t = -2.7
    model.set_onsite([-delta, delta])
    model.set_hop(t, 0, 1, [0, 0])
    model.set_hop(t, 1, 0, [1, 0])
    model.set_hop(t, 1, 0, [0, 1])

    return model.make_supercell([[supercell_size[0], 1], [-1, supercell_size[1]]], to_home=True)

def run():
    model = graphene_model()
    k_points = model.k_uniform_mesh([6, 6])
    H_numpy = model.hamiltonian(k_pts=k_points)[0]  # pick Gamma

    #H_torch = torch.from_numpy(H_numpy)

    # PEXSI returns local CSC values and indices
    rho_vals, rho_row, rho_col, mu = get_density_matrix_pexsi(H_numpy, temperature=0.01, numPoles=50)
    rho_sparse = coo_matrix((rho_vals, (rho_row, rho_col)), shape=H_numpy.shape)
    rho_dense = rho_sparse.toarray()

    rho_diag = Solve_Hamiltonian(torch.from_numpy(H_numpy), method="diagonalization")
    #PEXSI will get the density matrix at the values where the Hamiltonian is non-zero, so it will be sparse
    # This is not all the elements of the density matrix, since in principle the density matrix is long ranged
    # but to calculate the total energy, we only need the elements where the Hamiltonian is non-zero, so it is fine
    print("rho pexsi: ", np.round(rho_dense, decimals=3))
    print("rho diag: ", np.round(rho_diag.numpy().real, decimals=3))

    print("PEXSI chemical potential:", mu)
    print("Density matrix (PEXSI) trace:", np.real(np.trace(rho_dense)))
    print("Density matrix (diag) trace:", np.real(np.trace(rho_diag.numpy())))
    print("Difference norm:", np.linalg.norm(rho_dense - rho_diag.numpy()))

if __name__ == "__main__":
    run()