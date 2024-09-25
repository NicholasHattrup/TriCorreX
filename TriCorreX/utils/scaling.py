def scale_D(D, rho, T):
    return rho**(1/3) * T**(-1/2) * D

def unscale_D(D, rho, T):
    return rho**(-1/3) * T**(1/2) * D