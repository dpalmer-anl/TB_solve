import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment

def disentangle_bands(e, psi):
    """disentangle bands based on adjacent eigenvector overlap. assumes eigvals, eigvecs are sorted by kpoint index.\
    Use for disentangling bands from sparse diagonalization""".
    nbands = e.shape[0]
    sorted_levels = [e[0]]
    sorted_psi = [psi[0]]
    for i in range(nbands-1):
        e2, psi2 = e[i+1], psi[i+1]
        perm, line_breaks = best_match(psi[i], psi2)
        e2 = e2[perm]
        intermediate = (e + e2) / 2
        intermediate[line_breaks] = None
        psi = psi2[:, perm]
        e = e2
        sorted_psi.append(psi)
        #sorted_levels.append(intermediate)
        sorted_levels.append(e)
    return sorted_levels, sorted_psi

def best_match(psi1, psi2, threshold=None):
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
    if threshold is None:
        threshold = (2 * psi1.shape[0])**-0.25
    Q = np.abs(psi1.T.conj() @ psi2)  # Overlap matrix
    orig, perm = linear_sum_assignment(-Q)
    return perm, Q[orig, perm] < threshold
