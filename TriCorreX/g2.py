from numba import jit
import numpy as np

@jit(nopython=True)
def g2(coords, L, R_max, num_bins, tol=1e-2):
    hist = np.zeros(num_bins)
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
            hist[bin_i] += 2 # Counting pair ij and ji
    # This is a O(n^2) operation, total loops is n(n-1)/2
    return hist, num_atoms
