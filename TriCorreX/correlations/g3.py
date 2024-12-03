import numpy as np
from scipy.spatial import KDTree as Tree
from numba import jit

def get_neighbors(idx, pos, tree, L, R_max):
    """
    Get the indices of neighbors within R_max of a given atom position.

    Parameters:
    idx (int): Index of the reference atom.
    pos (ndarray): Position of the reference atom.
    tree (KDTree): KDTree constructed from atomic positions.
    L (float): Box length for periodic boundary conditions.
    R_max (float): Maximum distance for neighbor search.

    Returns:
    ndarray: Indices of neighboring atoms.
    """
    neighbors = tree.query_ball_point(pos, R_max, return_sorted=True)
    # Recast as array
    neighbors = np.array(neighbors)
    # Remove self and all neighbors with lower indices
    return neighbors[neighbors > idx]

def local3(idx, neighbors, positions, L, R_max, num_bins, counts):
    """
    Compute the local three-body correlation function for a given atom.

    Parameters:
    idx (int): Index of the reference atom.
    neighbors (ndarray): Indices of neighboring atoms.
    positions (ndarray): Atomic positions.
    L (float): Box length for periodic boundary conditions.
    R_max (float): Maximum distance for correlation calculation.
    num_bins (int): Number of bins for the histogram.
    counts (ndarray): Histogram of three-body distances.

    Returns:
    ndarray: Updated histogram of three-body distances.
    """
    # Given an atom in the system, we find its local g3 approximation
    neighbor_pos = positions[neighbors]
    # Recast to nearest image with origin at positions[idx]
    d_ij = neighbor_pos - positions[idx]
    d_ij -= L * np.round(d_ij / L)
    delr = R_max / num_bins
    num_neighbors = len(neighbors)
    for i in range(num_neighbors - 1):
        for j in range(i + 1, num_neighbors):
            d_ik = d_ij[j]
            d_jk = d_ik - d_ij[i]
            if np.sqrt(np.sum(d_jk**2)) > R_max:
                continue
            r = np.sqrt(np.sum(d_ij[i] ** 2))
            s = np.sqrt(np.sum(d_ij[j] ** 2))
            t = np.sqrt(np.sum(d_jk ** 2))
            r_bin = int(r / delr)
            s_bin = int(s / delr)
            t_bin = int(t / delr)
            r_bin, s_bin, t_bin = sorted((r_bin, s_bin, t_bin), reverse=True)
            counts[r_bin, s_bin, t_bin] += 1
    return counts

def tree3(coords, L, R_max, num_bins):
    """
    Calculate the three-body correlation function using KDTree for neighbor searching.

    Parameters:
    coords (ndarray): Array of atomic coordinates.
    L (float): Box length for periodic boundary conditions.
    R_max (float): Maximum distance for correlation calculation.
    num_bins (int): Number of bins for the histogram.
    tol (float): Tolerance for numerical accuracy (default 1e-2).

    Returns:
    tuple: Histogram of three-body distances, Number of atoms considered.
    """
    counts = np.zeros((num_bins, num_bins, num_bins))
    num_atoms = len(coords)
    tree = Tree(coords)
    for idx in range(num_atoms - 2):
        pos = coords[idx]
        neighbors = get_neighbors(idx, pos, tree, L, R_max)
        counts += local3(idx, neighbors, coords, L, R_max, num_bins, counts)
    return counts, num_atoms



@jit(nopython=True)
def corre3(coords, L, R_max, num_bins):
    """
    Calculate the three-body correlation function brute force.

    Parameters:
    coords (ndarray): Array of atomic coordinates.
    L (float): Box length for periodic boundary conditions.
    R_max (float): Maximum distance for correlation calculation.
    num_bins (int): Number of bins for the histogram.
    tol (float): Tolerance for numerical accuracy (default 1e-2).

    Returns:
    tuple: Histogram of three-body distances, Number of atoms considered.
    """
    counts = np.zeros((num_bins, num_bins, num_bins), dtype=np.int64)
    num_atoms = len(coords)
    R_max_sq = R_max**2
    delr = R_max / num_bins

    for i in range(0, num_atoms-2):
        for j in range(i+1, num_atoms-1):
            d_ij = coords[j] - coords[i]
            d_ij -= L * np.round(d_ij / L)
            r_sq = np.sum(d_ij**2)
            if r_sq > R_max_sq:
                continue
            for k in range(j+1, num_atoms):
                d_ik = coords[k] - coords[i]
                d_ik -= L * np.round(d_ik / L)
                s_sq = np.sum(d_ik**2)
                if s_sq > R_max_sq:
                    continue
                d_jk = coords[k] - coords[j]
                d_jk -= L * np.round(d_jk / L)
                t_sq = np.sum(d_jk**2)
                if t_sq > R_max_sq:
                    continue
                r, s, t = np.sqrt(r_sq), np.sqrt(s_sq), np.sqrt(t_sq)
                r_bin = int(r / delr)
                s_bin = int(s / delr)
                t_bin = int(t / delr)
                counts[r_bin, s_bin, t_bin] += 1
    return counts, num_atoms

