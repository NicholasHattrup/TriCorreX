import numpy as np 

def compute_msd(trajectory):
    """
    Compute the mean squared displacement (MSD) of a trajectory.

    Parameters:
    trajectory (ndarray): Array of particle positions over time.

    Returns:
    ndarray: The computed MSD values.
    """
    msd_per_atom = np.sum((trajectory - trajectory[0])**2, axis=2)
    msd = np.mean(msd_per_atom, axis=1)
    return msd

def compute_diffusion(msd, time, dim=3):
    """
    Compute the diffusion coefficient from the mean squared displacement (MSD).

    Parameters:
    msd (ndarray): Array of MSD values.
    time (ndarray): Array of time values.
    dim (int): Dimension of the system (default is 3).

    Returns:
    float: The computed diffusion coefficient.
    """
    n = len(msd)
    # Linear regressor minimizing the L2 loss 
    D = (n*np.sum(msd*time)-np.sum(msd)*np.sum(time))/(n*np.sum(time**2)-np.sum(time)**2)/(2*dim)
    return D