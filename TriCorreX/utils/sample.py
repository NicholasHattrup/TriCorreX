import numpy as np


'''
Critical points and inequality bounds:
S. Stephan, J. Staubach, and H. Hasse, Review and comparison of equations of state for the Lennard-Jones fluid, Fluid Phase Equilibria 523, 112772 (2020)
  
Parameters and expressions for computing the dew and bubble points:
S. Stephan, M. Thol, J. Vrabec, and H. Hasse, Thermophysical Properties of the Lennard-Jones Fluid: Database and Data Assessment, J. Chem. Inf. Model. 59, 4248 (2019)

Parameters and expression for computing the freezing point:
A. Köster, P. Mausbach, and J. Vrabec, Premelting, solid-fluid equilibria, and thermodynamic properties in the high density region based on the Lennard-Jones potential, The Journal of Chemical Physics 147, 144502 (2017)
'''

T_c = 1.321 # Critical temperature
rho_c = 0.316 # Critical density

def sat_rho_vap(T): # p'
    n = np.array([1.341700, 2.075332, -2.123475, 0.328998, 1.386131])
    t = np.array([0.327140, 0.958759, 1.645654, 17.000001, 2.400858])
    return rho_c * (1 + np.sum(n * (1 - T/T_c)**t))

def sat_rho_liquid(T): # p''
    n = np.array([-8.135822, -102.919110, -3.037979, -44.381841, -34.55892948])
    t = np.array([1.651685, 43.469214, 0.462877, 11.500462, 5.394370]) 
    return rho_c * np.exp(np.sum(n * (1 - T/T_c)**t))

def freezing_point(T): # p'''
    l = np.array([0.794326405787077, 0.287446151493139, -0.405667818555559, 0.417645193659883, -0.211440758862587, 0.040324958732013])
    return (T**(1/4)) * np.sum(l * (T**np.arange(0, -6, -1)))
   

def gas(tol=1e-1, seed=None):
    if seed is None:
        seed = np.random.randint(0, 2**32)
    rng = np.random.default_rng(seed=seed)
    # T = rng.uniform(0.66, 1.321) From the paper, but we bring T -> 0.92 to avoid super small densities rho < 0.015
    T = rng.uniform(1.21, 1.321)
    rho_vap = sat_rho_liquid(T)
    rho = rng.uniform(tol, rho_vap) # 0 - p''
    return np.round(rho, 3), np.round(T, 3)

def liquid(seed=None):
    if seed is None:
        seed = np.random.randint(0, 2**32)
    rng = np.random.default_rng(seed=seed)  
    T = rng.uniform(0.723, 1.321) # 0.66-0.723 can only be a high density liquid
    rho_liq = sat_rho_vap(T)
    rho_sol = freezing_point(T)
    rho = rng.uniform(rho_liq, 0.95 * rho_sol) # p' - 0.95 * p'''
    return np.round(rho, 3), np.round(T, 3)

def high_density_liquid(seed=None):
    if seed is None:
        seed = np.random.randint(0, 2**32)
    rng = np.random.default_rng(seed=seed)
    T = rng.uniform(0.66, 1.321)
    rho_sol = freezing_point(T)
    rho = rng.uniform(0.95 * rho_sol, rho_sol) # 0.95 * p''' - p'''
    return np.round(rho, 3), np.round(T, 3)

def critical(seed=None):
    if seed is None:
        seed = np.random.randint(0, 2**32)
    rng = np.random.default_rng(seed=seed)
    T = rng.uniform(1.3, 1.45)
    rho = rng.uniform(.21, .44)
    return np.round(rho, 3), np.round(T, 3)

def super_critical(tol=5e-2, seed=None):
    if seed is None:
        seed = np.random.randint(0, 2**32)
    rng = np.random.default_rng(seed=seed)
    T = rng.uniform(1.321, 6)
    rho_sol = freezing_point(T)
    rho = rng.uniform(tol, 0.95 * rho_sol) # T >= 1.321 and 0 - 0.95 * p'''
    return np.round(rho, 3), np.round(T, 3)

def high_density_super_critical(seed=None):
    if seed is None:
        seed = np.random.randint(0, 2**32)
    rng = np.random.default_rng(seed=seed)
    T = rng.uniform(1.321, 6)
    rho_sol = freezing_point(T)
    rho = rng.uniform(0.95 * rho_sol, rho_sol) # 0.95 * p''' - p'''
    return np.round(rho, 3), np.round(T, 3)

def sample_points(N=1, seed=None):
    dict = {
        'gas': gas,
        'liquid': liquid,
        'high_density_liquid': high_density_liquid,
        'critical': critical,
        'super_critical': super_critical,
        'high_density_super_critical': high_density_super_critical
    }
    if seed is None:
        seed = np.random.randint(0, 2**32)

    rng = np.random.default_rng(seed=seed)
    choices = rng.choice(['gas', 'liquid', 'high_density_liquid', 'critical', 'super_critical', 'high_density_super_critical'], N, replace=True) # Remove gas for now
    seeds = rng.integers(0, 2**32, N)
    return [[dict[choice](seed=seed), choice] for choice, seed in zip(choices, seeds)]
    

