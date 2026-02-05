import numpy as np
import torch
from pythtb import TBModel, Lattice
from tb_solve import Solve_Hamiltonian
import time
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

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
    t = -2.7

    # set on-site energies
    my_model.set_onsite([-delta, delta])
    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])
    my_model.set_hop(t, 0, 1, [0, 0])
    my_model.set_hop(t, 1, 0, [1, 0])
    my_model.set_hop(t, 1, 0, [0, 1])

    sc_model = my_model.make_supercell([[supercell_size[0], 1], [-1, supercell_size[1]]], to_home=True)

    return np.squeeze(sc_model.hamiltonian([0,0]))



if __name__ == "__main__":
    system_sizes = [2,5, 10, 15, 20, 25,  30,35,40,50,60,70,80]
    ham_dim = []
    dmm_time = []
    direct_diag_time = []
    foe_time = []
    tri_diag_time = []

    for i, size in enumerate(system_sizes):
        H = create_graphene_hamiltonian(supercell_size=[size, size])

        print(np.shape(H))
        ham_dim.append(np.shape(H)[0])
        H_torch = torch.from_numpy(H)
        
        # start_time = time.time()
        # direct_diag_density_matrix = Solve_Hamiltonian(H_torch, method="diagonalization")
        # direct_diag_time.append(time.time() - start_time)
        direct_diag_time.append(0)
        foe_time.append(0)
        dmm_time.append(0)
        natoms = np.shape(H)[0]
        tri_diag_matrix = np.diag(-2.7*np.ones(natoms), k=1) + np.diag(-2.7*np.ones(natoms), k=-1)
        start_time = time.time()
        _,_ = eigh_tridiagonal(np.zeros(natoms), -2.7*np.ones(natoms-1))
        tri_diag_time.append(time.time() - start_time)
        # start_time = time.time()
        # dmm_density_matrix = Solve_Hamiltonian(H_torch, method="density_matrix_minimization")
        # dmm_time.append(time.time() - start_time)
        # start_time = time.time()
        # foe_density_matrix = Solve_Hamiltonian(H_torch, method="fermi_operator_expansion")
        # foe_time.append(time.time() - start_time)

    ham_dim_continuous = np.linspace(min(ham_dim), max(ham_dim),1000)
    dmm_fit = np.polyfit(ham_dim, dmm_time, 2)
    dmm_extrapolated_time = np.polyval(dmm_fit, ham_dim_continuous)
    direct_diag_fit = np.polyfit(ham_dim, direct_diag_time, 3)
    direct_diag_extrapolated_time = np.polyval(direct_diag_fit, ham_dim_continuous)
    foe_fit = np.polyfit(ham_dim, foe_time, 2)
    foe_extrapolated_time = np.polyval(foe_fit, ham_dim_continuous)
    tri_diag_fit = np.polyfit(ham_dim, tri_diag_time, 2)
    tri_diag_extrapolated_time = np.polyval(tri_diag_fit, ham_dim_continuous)
    # plt.scatter(ham_dim, np.array(direct_diag_time)/3600, label="Direct Diagonalization",color="blue")
    # plt.plot(ham_dim_continuous, np.array(direct_diag_extrapolated_time)/3600, color="blue")
    plt.scatter(ham_dim, np.array(tri_diag_time), label="Tridiagonal Diagonalization",color="green")
    #plt.plot(ham_dim_continuous, np.array(tri_diag_extrapolated_time)/3600, color="green")
    # plt.scatter(ham_dim, dmm_time, label="Density Matrix Minimization",color="red")
    # plt.plot(ham_dim_continuous, dmm_extrapolated_time, color="red")
    # plt.scatter(ham_dim, foe_time, label="Fermi Operator Expansion",color="green")
    # plt.plot(ham_dim_continuous, foe_extrapolated_time, color="green")
    plt.legend()
    plt.xlabel("Hamiltonian Dimension")
    plt.ylabel("Time (seconds)")
    plt.title("Time taken to solve density matrix")
    plt.savefig("density_matrix_scaling.png")
    plt.show()
    plt.clf()