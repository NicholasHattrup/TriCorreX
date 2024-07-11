from numba import jit
import numpy as np

@jit(nopython=True)
def corre2(coords, L, R_max, num_bins):
    """
    Estimate the 2-body correlation function using histogramming.

    Parameters:
    coords (ndarray): Array of atomic coordinates.
    L (float): Box length for periodic boundary conditions.
    R_max (float): Maximum distance for binning.
    num_bins (int): Number of bins for the histogram.
    
    Returns:
    counts (ndarray): Histogram of particle counts at varying distance.
    num_atoms (int): Number of atoms considered.
    """
    counts = np.zeros(num_bins)
    del_r = (R_max) / num_bins
    coords = coords % L
    num_atoms = len(coords)
    for i in range(num_atoms-1):
        ctr_coords = coords[i]
        for j in range(i+1, num_atoms):
            # Nearest image 
            r_ij = coords[j] - ctr_coords
            r_ij -= L * np.round(r_ij / L)
            dist_ij = np.sqrt(np.sum((r_ij)**2))
            if R_max < dist_ij:
                continue
            bin_i = int((dist_ij) / del_r)
            counts[bin_i] += 2 # Counting pair ij and ji
    # This is a O(n^2) operation, total loops is n(n-1)/2
    return counts, num_atoms
