from numba import jit
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

@jit(nopython=True)
def compute_corre2(coords, L, R_max, num_bins, num_atoms):
    counts = np.zeros(num_bins)
    del_r = R_max / num_bins
    coords = coords % L
    for i in range(num_atoms-1):
        ctr_coords = coords[i]
        for j in range(i+1, num_atoms):
            r_ij = coords[j] - ctr_coords
            r_ij -= L * np.round(r_ij / L)
            dist_ij = np.sqrt(np.sum(r_ij ** 2))
            if dist_ij >= R_max:
                continue
            bin_i = int(dist_ij / del_r)
            counts[bin_i] += 2
    return counts

@jit(nopython=True)
def compute_bin_volumes(R_max, num_bins):
    radial_dists = compute_distances(R_max, num_bins)
    volumes = 4 * np.pi / 3 * (radial_dists[1:]**3 - radial_dists[:-1]**3)
    return volumes

def compute_g2(coords, L, R_max, num_bins, rho):
    counts = np.zeros(num_bins)
    if len(coords.shape) == 2:
        num_atoms, _ = coords.shape
        counts += compute_corre2(coords, L, R_max, num_bins, num_atoms)
    else:
        samples, num_atoms, _ = coords.shape
        with ProcessPoolExecutor(max_workers=9) as executor:
            futures = [executor.submit(compute_corre2, coords[i], L, R_max, num_bins, num_atoms) for i in range(samples)]
            for future in as_completed(futures):
                counts += future.result()
        counts /= samples

    distances = compute_distances(R_max, num_bins)
    bin_vols = compute_bin_volumes(R_max, num_bins)
    bin_counts = rho * bin_vols
    g2 = counts / (num_atoms * bin_counts)
    return g2, distances

@jit(nopython=True)
def compute_distances(R_max, num_bins):
    return np.linspace(0, R_max, num_bins + 1)

