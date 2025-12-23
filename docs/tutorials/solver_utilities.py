import numpy as np
import torch
from pythtb import TBModel, Lattice
import time
from tb_solve import Solver_helper
from tb_solve.utils import Get_optimal_solver, Converge_Solver_settings

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

    return sc_model

def run_solver_utilities_tutorial():
    print("TB_solve Utilities Tutorial")
    print("===========================\n")

    # 1. Create a larger model to demonstrate scaling
    print("Creating Graphene Supercell (10x10)...")
    my_model = create_graphene_hamiltonian(supercell_size=[10, 10])
    k_points = my_model.k_uniform_mesh([1, 1]) # Single k-point for simplicity
    ham_k = my_model.hamiltonian(k_pts=k_points)
    H_numpy = ham_k[0]
    
    device = torch.device("cpu")
    H_torch = torch.from_numpy(H_numpy).to(device=device)
    # Convert to sparse for relevant methods
    H_sparse = H_torch.to_sparse()
    
    print(f"Hamiltonian shape: {H_torch.shape}")
    print("-" * 40)

    # 2. Solver_helper: Estimate Time
    print("\n2. Time Estimation with Solver_helper")
    print("-------------------------------------")
    print("Estimating computation time for different methods:")
    
    t_diag = Solver_helper(H_torch, method="diagonalization")
    print(f"  Diagonalization: {t_diag:.2e} s")
    
    t_foe = Solver_helper(H_sparse, method="fermi_operator_expansion", n_moments=100)
    print(f"  Fermi Operator Expansion: {t_foe:.2e} s")
    
    t_pur = Solver_helper(H_torch, method="density_matrix_purification", max_iterations=50)
    print(f"  Purification: {t_pur:.2e} s")

    # 3. Get_optimal_solver: Automatic Selection
    print("\n3. Selecting Optimal Solver")
    print("---------------------------")
    optimal_method, est_time = Get_optimal_solver(H_torch, method="all")
    print(f"Recommended method: {optimal_method} (Estimated time: {est_time:.2e} s)")
    
    # 4. Converge_Solver_settings: Optimizing Parameters
    print("\n4. Converging Solver Parameters")
    print("-------------------------------")
    print("Finding optimal parameters for Fermi Operator Expansion...")
    
    # This function runs the solver with increasing accuracy settings (e.g., n_moments)
    # until the result (density matrix trace or other metric) converges.
    # Note: For FOE, it checks convergence w.r.t n_moments.
    
    # We use a smaller Hamiltonian here for speed in the tutorial
    my_model_small = create_graphene_hamiltonian(supercell_size=[4, 4])
    H_small = torch.from_numpy(my_model_small.hamiltonian([0., 0.])[0]).to(device)
    
    # Converge n_moments
    converged_params = Converge_Solver_settings(
        H_small,
        method="fermi_operator_expansion",
        variable="n_moments",
        start_value=50,
        end_value=300,
        step=50,
        tolerance=1e-3,
        kbT=0.01 # Fixed temperature
    )
    
    print(f"\nConverged settings: {converged_params}")

if __name__ == "__main__":
    run_solver_utilities_tutorial()

