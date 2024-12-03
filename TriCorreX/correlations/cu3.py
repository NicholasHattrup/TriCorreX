from numba import jit
import numpy as np
from scipy.spatial import KDTree as Tree
import time



    
@jit(nopython=True)
def transform(r, s, t, R_max, R_min=0):
    # Manually reorder so that r >= s >= t
    if r < s:
        r, s = s, r
    if r < t:
        r, t = t, r
    if s < t:
        s, t = t, s
    
    r_prime = (r - R_min) / (R_max - R_min)
    s_prime = 2 * (s - 0.5 * r) / r
    t_prime = (t - r + s) / (2 * s - r)
    
    return r_prime, s_prime, t_prime

@jit(nopython=True)
def indices(r, s, t, R_max, num_bins):
    r_prime, s_prime, t_prime = transform(r, s, t, R_max)
    r_bin = int(r_prime * num_bins)
    s_bin = int(s_prime * num_bins)
    t_bin = int(t_prime * num_bins)
    return r_bin, s_bin, t_bin

@jit(nopython=True)
def volumes(num_bins, R_max):
    factor = R_max**6*np.pi**2/(72*num_bins**12)
    vols = np.empty((num_bins, num_bins, num_bins))
    for i in range(1, num_bins+1):
        for j in range(1, num_bins+1):
            for k in range(1, num_bins+1):
                vols[i-1, j-1, k-1] = (i**6-(i-1)**6)*(12*j**3*(2*k-num_bins-1)+6*j**2*(k*(4*num_bins-6)+num_bins+3)-12*j*(2*k*(num_bins-1)-num_bins**3+1)+k*(8*num_bins-6)-6*num_bins**3-num_bins-3)
    vols *= factor
    return vols

@jit(nopython=True)
def corre3(coords, L, R_max, num_bins, tol=1e-2):
    hist = np.zeros((num_bins, num_bins, num_bins))
    num_atoms = len(coords)
    coords = coords % L
    R_max_sq = R_max**2
    for i in range(num_atoms-2):
        for j in range(i+1, num_atoms-1):
            d_ij = coords[j] - coords[i]
            r_sq = np.sum(d_ij**2)
            if r_sq > R_max_sq:
                continue
            for k in range(j+1, num_atoms):
                d_ik = coords[k] - coords[i]
                s_sq = np.sum(d_ik**2)
                if s_sq > R_max_sq:
                    continue
                d_jk = d_ik - d_ij
                t_sq = np.sum(d_jk**2)
                if t_sq > R_max_sq:
                    continue
                r = np.sqrt(r_sq)
                s = np.sqrt(s_sq)
                t = np.sqrt(t_sq)
                r_bin, s_bin, t_bin = indices(r, s, t, R_max, num_bins)
                hist[r_bin, s_bin, t_bin] += 1
    return hist, num_atoms

def g3(configurations, L, R_max, num_bins, tol=1e-2):
    # Dimensions of configurations is either(num_atoms, 3) or (1, num_atoms, 3) 
    if configurations.ndim == 2:
        configurations = configurations[np.newaxis, :, :]

    num_configurations = configurations.shape[0]
    hist_tot, num_atoms_tot = corre3(configurations[0], L, R_max, num_bins, tol)
    
    for n in range(1, num_configurations):
        hist, num_atoms = corre3(configurations[n], L, R_max, num_bins, tol)
        hist_tot += hist
        num_atoms_tot += num_atoms

    # Compute normalizing factor 
    bin_vols = volumes(num_bins, R_max)
    system_vol = L**3
    rho = num_atoms_tot/(num_configurations*system_vol) 
    hist_ideal = rho**2*bin_vols
    
    hist_tot /= (num_atoms_tot*hist_ideal)
    return hist_tot, num_atoms_tot





### Code below uses the scaled method WITH KDTree ###
### Want to ensure working w/o KDTree first###

# @jit(nopython=True)
# def local_corre3(hist, center_coords, local_coords, R_max):
#     # Given atom 1 in the inner region and atoms 2 and 3 in the bulk region (which includes inner)
#     # I want to ensure that each triplet is only counted once
    
#     # Pre-calculate all distances between center and neighbors
#     num_neighbors = len(local_coords)
#     distances_to_coord = np.sqrt(np.sum((local_coords - center_coords)**2, axis=1))
#     # Pre-calculate distances between all neighbors where d_ij = dist(neighbors[i], neighbors[j]
#     num_bins = hist.shape[0]

#     for j in range(num_neighbors-1):
#         distances_to_neighbors = np.sqrt(np.sum((local_coords[j+1:] - local_coords[j])**2, axis=1))
#         for k in range(num_neighbors-j-1):
#             dist_jk = distances_to_neighbors[k]
#             if R_max < dist_jk:
#                 continue
#             dist_ij = distances_to_coord[j]
#             dist_ik = distances_to_coord[j+1+k]
#             r_bin, s_bin, t_bin = get_bin(dist_ij, dist_ik, dist_jk, R_max, num_bins)
#             hist[r_bin, s_bin, t_bin] += 1
#     return hist




# def g3_cubic(coords, L, R_max, num_bins, tol=1e-2):
#     hist = np.zeros((num_bins, num_bins, num_bins))
#     coords = coords % L
#     # Filtering atoms
#     inner_bound = R_max + tol
#     outer_bound = L - (R_max + tol)
#     mask = np.all((coords > inner_bound) & (coords < outer_bound), axis=1)
#     inner_atoms = np.where(mask)[0]
#     num_inner = len(inner_atoms)
#     tree = Tree(coords, boxsize=L) # O(nlogn) operation
#     for atom in inner_atoms:
#         center_coords = coords[atom]
#         neighbors = tree.query_ball_point(x=center_coords, r=R_max) # O(n) operation
#         neighbors.remove(atom) # O(n) operation
#         # If a neighbor is less than atom and is in the inner region, remove it 
#         # i.e. if atoms 2 and 3 are in inner, only use combination (2, 3, x)
#         # But if only atoms 3 is inner but 2 is a neighbor, can use combination (3, 2, x)
#         neighbors = [neighbor for neighbor in neighbors if neighbor > atom or (neighbor < atom and not mask[neighbor])]
#         local_coords = coords[neighbors]
#         hist = update_hist_cubic(hist, center_coords, local_coords, R_max)

#     return hist, num_inner
