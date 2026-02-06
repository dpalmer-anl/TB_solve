import torch
import math
from tb_solve.Solver import device, gpu_avail, Solve_Hamiltonian
import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment


def disentangle_bands(e, psi, o_tol=1):
    """Disentangle bands across k-points using eigenvector overlaps.

    Args:
        e (array-like): Eigenvalues with shape (nk, nbands).
        psi (array-like): Eigenvectors with shape (nk, norb, nbands).

    Returns:
        tuple: (e_sorted, psi_sorted) with the same shapes as inputs.
    """
    e = np.array(e)
    psi = np.array(psi)

    if e.ndim != 2 or psi.ndim != 3:
        raise ValueError("Expected e with shape (nk, nbands) and psi with shape (nk, norb, nbands)")

    nk, nbands = e.shape
    _, norb, _ = psi.shape

    e_sorted = [e[0]]
    psi_sorted = [psi[0]]

    for k in range(nk - 1):
        psi_curr = psi_sorted[-1]
        e_next = e[k + 1]
        psi_next = psi[k + 1]

        perm, line_breaks = best_match(psi_curr, psi_next, o_tol=o_tol)
        e_next = e_next[perm]
        psi_next = psi_next[:, perm]

        psi_sorted.append(psi_next)
        e_sorted.append(e_next)

    return np.array(e_sorted), np.array(psi_sorted)

def best_match(psi1, psi2, o_tol=1):
    """Find the best match of two sets of eigenvectors.

    
    Parameters:
    -----------
    psi1, psi2 : numpy 2D complex arrays
        Arrays of initial and final eigenvectors.
    threshold : float, optional
        Minimal overlap when the eigenvectors are considered belonging to the same band.
        The default value is :math:`1/(2N)^{1/4}`, where :math:`N` is the length of each eigenvector.
    
    Returns:
    --------
    sorting : numpy 1D integer array
        Permutation to apply to ``psi2`` to make the optimal match.
    diconnects : numpy 1D bool array
        The levels with overlap below the ``threshold`` that should be considered disconnected.
    """
    
    threshold = o_tol * (2 * psi1.shape[0])**-0.25
    Q = np.abs(psi1.T.conj() @ psi2)  # Overlap matrix
    orig, perm = linear_sum_assignment(-Q)
    return perm, Q[orig, perm] < threshold
