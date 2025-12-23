import torch
import numpy as np
from pythtb import TBModel, Lattice
from tb_solve.utils import Solver_helper, Get_optimal_solver, Converge_Solver_settings
from tb_solve import Solve_Hamiltonian

# Helper function to create a Hamiltonian (same as in the graphene tutorial)
def create_graphene_hamiltonian(supercell_size=[5, 5]):
    lat_vecs = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb_vecs = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])
    my_model = TBModel(lat)
    my_model.set_onsite([-0.0, 0.0])
    my_model.set_hop(-2.7, 0, 1, [0, 0])
    my_model.set_hop(-2.7, 1, 0, [1, 0])
    my_model.set_hop(-2.7, 1, 0, [0, 1])
    sc_model = my_model.make_supercell([[supercell_size[0], 1], [-1, supercell_size[1]]], to_home=True)
    return sc_model

def run_utilities_tutorial():
    print("--- Solver Utilities Tutorial ---")
    
    # 1. Create a large Hamiltonian
    print("\n1. Creating a large Graphene Supercell (10x10)...")
    sc_model = create_graphene_hamiltonian(supercell_size=[10, 10])
    # Get Hamiltonian at Gamma point
    H_numpy = sc_model.hamiltonian(k_pts=[[0, 0]])[0]
    H_torch = torch.from_numpy(H_numpy).to(torch.complex128)
    
    # Ensure we use CPU for this tutorial to be safe, or detect device
    if torch.cuda.is_available():
        H_torch = H_torch.cuda()
        print("Using GPU.")
    else:
        print("Using CPU.")
        
    print(f"Hamiltonian shape: {H_torch.shape}")

    # 2. Estimate Computation Time
    print("\n2. Estimating computation time for different methods:")
    t_diag = Solver_helper(H_torch, method="diagonalization")
    t_foe = Solver_helper(H_torch, method="fermi_operator_expansion", n_moments=100)
    # Note: sparse_diagonalization is usually for eigenvalues, not full density matrix
    # but Solver_helper can still estimate it.
    t_sparse = Solver_helper(H_torch, method="sparse_diagonalization", nbands=20)
    
    # 3. Get Optimal Solver automatically
    print("\n3. Selecting optimal solver...")
    best_method, min_time = Get_optimal_solver(H_torch, n_moments=100, nbands=20, max_iterations=50)
    print(f"Optimal solver: {best_method} (Est. time: {min_time:.2e} s)")
    
    # 4. Converge Solver Settings
    # Example: Find sufficient number of moments for Fermi Operator Expansion
    print("\n4. Converging 'n_moments' for Fermi Operator Expansion...")
    # We want to find n_moments such that the density matrix doesn't change much
    optimal_moments = Converge_Solver_settings(
        H_torch, 
        method="fermi_operator_expansion", 
        variable="n_moments",
        start_value=50, 
        end_value=300, 
        step=50, 
        tolerance=1e-4,
        kbT=0.01 # Fixed argument
    )
    
    print(f"Converged n_moments: {optimal_moments}")
    
    # 5. Run with optimized settings
    print("\n5. Running final calculation...")
    dm = Solve_Hamiltonian(H_torch, method="fermi_operator_expansion", n_moments=optimal_moments, kbT=0.01)
    print("Done. Density matrix shape:", dm.shape)

if __name__ == "__main__":
    run_utilities_tutorial()

