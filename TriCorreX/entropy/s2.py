import numpy as np 
from numba import jit

def compute_s2(distances, g2, rho, tol=1e-6):
    """
    Compute the per-particle 2-body excess entropy s2.

    Parameters:
    distances (ndarray): Array of distance values.
    g2 (ndarray): Radial distribution function values corresponding to the distances.
    density (float): Number density of particles.
    tol (float): Tolerance value to filter rdf values considered as zero (default is 1e-6).

    Returns:
    float: The computed per-particle 2-body excess entropy s2.
    """
    if len(distances) != len(g2):
        raise ValueError("distances and rdf must have the same length.")

    mask = g2 > tol
    integrand = distances.copy()**2
    integrand[mask] = (g2[mask] * np.log(g2[mask]) - g2[mask] + 1)
    s2 = -2 * np.pi * rho * integrand 
    return np.trapz(s2, distances)


def compute_s2_cumulative(distances, g2, rho, tol=1e-6):
    """
    Compute the cumulative per-particle 2-body excess entropy s2.

    Parameters:
    distances (ndarray): Array of distance values.
    g2 (ndarray): Radial distribution function values corresponding to the distances.
    density (float): Number density of particles.
    tol (float): Tolerance value to filter rdf values considered as zero (default is 1e-6).

    Returns:
    ndarray: The computed cumulative per-particle 2-body excess entropy s2.
    """
    if len(distances) != len(g2):
        raise ValueError("distances and rdf must have the same length.")

    dr = (distances[1] - distances[0])/2
    mask = g2 > tol
    integrand = distances.copy()**2
    integrand[mask] = (g2[mask] * np.log(g2[mask]) - g2[mask] + 1)
    cum_int = np.zeros(len(distances))
    cum_int[0] = integrand[0]
    for i in range(1, len(distances)-1):
        cum_int[i] = cum_int[i-1] + 2*integrand[i]
    cum_int[-1] = cum_int[-2] + integrand[-1]
    cum_int *= dr 
    cum_s2 = -2 * np.pi * rho * cum_int
    return cum_s2

