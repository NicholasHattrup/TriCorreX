
import numpy as np 
from numba import jit

@jit(nopython=True)
def compute_volume(L, Nb):
    """
    L: Maximum distance to account for
    Nb: Number of bins along each dimension
    """
    
    V = np.zeros((Nb, Nb, Nb))

    # Ns: Number of samples for Monte Carlo integration
    Ns = 1000
    D = L / Nb

    for i in range(1, Nb + 1):
        for j in range(1, i + 1):
            for k in range(1, j + 1):
                if i <= k + j - 2 and j <= i + k - 2 and k <= i + j - 2:
                    V[i-1, j-1, k-1] = np.pi**2 * D**6 * (2 * i - 1) * (2 * j - 1) * (2 * k - 1)
                else:
                    Vr = np.pi**2 * D**6 * (2 * i - 1) * (2 * j - 1) * (2 * k - 1)
                    X = (np.array([i-1, j-1, k-1]) + np.random.rand(Ns, 3)) * D

                    ls = (X[:, 0] + X[:, 1] >= X[:, 2]) & (X[:, 2] + X[:, 1] >= X[:, 0]) & (X[:, 2] + X[:, 0] >= X[:, 1])

                    S0 = np.sum(X[:, 0] * X[:, 1] * X[:, 2])
                    S1 = np.sum(X[ls, 0] * X[ls, 1] * X[ls, 2])

                    V[i-1, j-1, k-1] = (S1 / S0) * Vr

    for i in range(1, Nb + 1):
        for j in range(1, i + 1):
            for k in range(1, j + 1):
                V[i-1, k-1, j-1] = V[i-1, j-1, k-1]
                V[j-1, i-1, k-1] = V[i-1, j-1, k-1]
                V[j-1, k-1, i-1] = V[i-1, j-1, k-1]
                V[k-1, i-1, j-1] = V[i-1, j-1, k-1]
                V[k-1, j-1, i-1] = V[i-1, j-1, k-1]

    return V
