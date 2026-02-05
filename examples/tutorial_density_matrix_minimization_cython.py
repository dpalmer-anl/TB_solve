import numpy as np
import torch
from pythtb import TBModel, Lattice
from tb_solve import density_matrix_minimization_cython

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

    # Same signature/output as tb_solve.Solver.density_matrix_minimization
    rho = density_matrix_minimization_cython(
        H_numpy,
        epsilon=1e-6,
        max_iterations=60,
        spin_degeneracy=2.0,
    )

    print("Density matrix:", rho)
    print("Trace (should be ~N/2 * spin):", np.real(np.trace(rho)))

if __name__ == "__main__":
    run()