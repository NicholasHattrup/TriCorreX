import numpy as np 
from numba import jit



@jit(nopython=True)
def compute_s3(r, s, t, g2, g3, vols, density, tol=1e-6):
    """
    Compute the per-particle 3-body excess entropy s3.

    Parameters:
    r (ndarray): Array of distances for pair-wise distance r.
    s (ndarray): Array of distances for pair-wise distance s.
    t (ndarray): Array of distances for pair-wise distance t.
    g2 (callable): Pair-wise distribution function 
    g3 (ndarray): Three-body distribution function values corresponding to the bins of r, s, and t.
    vols (ndarray): Array of volume elements corresponding to the bins of r, s, and t.
    density (float): Number density of particles.
    tol (float): Tolerance value 

    Returns:
    float: The computed per-particle 3-body excess entropy s3.
    """

    if len(r) != len(s) or len(r) != len(t) or len(s) != len(t):
        raise ValueError("r, s, t, and g3 must have the same length.")
    del_r = r[1] - r[0]
    del_s = s[1] - s[0]
    del_t = t[1] - t[0]
    s3 = 0
    
    # Get weights for volume elements 
    vols[0, :, :] *= 1/2
    vols[-1, :, :] *= 1/2
    vols[:, 0, :] *= 1/2
    vols[:, -1, :] *= 1/2
    vols[:, :, 0] *= 1/2
    vols[:, :, -1] *= 1/2

    """
    In 1D trapezoidal integration, the first and last points are only counted half in the sum.

    This extends to higher dimensions like 3D:
        - Corners with one edge have weight 1/2
        - Corners with two edges have weight 1/4
        - Corners with three edges have weight 1/8

    Instead of generating a redudant array of weights, we can just multiply the volume elements by the appropriate factor.
    """
    
    n_r, n_s, n_t = len(r), len(s), len(t)

    for i in range(n_r):
        for j in range(n_s):
            for k in range(n_t):
                r_mid = r[i] + del_r / 2
                s_mid = s[j] + del_s / 2
                t_mid = t[k] + del_t / 2
                g2_r = g2(r_mid)
                g2_s = g2(s_mid)
                g2_t = g2(t_mid)
                integrand = g3[i, j, k] * np.log(g3[i, j, k] / (g2_r * g2_s * g2_t)) - (g3[i, j, k] + g2_r + g2_s + g2_t) + g2_r * g2_s + g2_s * g2_t + g2_t * g2_r - 1
                integrand *= vols[i, j, k]
                s3 += integrand
    
    s3 *= -8 * np.pi ** 2 * density ** 2 / 6 
    return s3
