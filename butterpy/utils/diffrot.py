import numpy as np

def sin2(omega_0, delta_omega, lat):
    return omega_0 - delta_omega * np.sin(lat)**2