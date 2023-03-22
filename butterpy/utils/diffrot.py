import numpy as np

def sin2(omega_0, delta_omega, lat):
    """
    Sine squared differential rotation profile

    Parameters
    ----------
    omega_0 (float): the equatorial rotation rate.

    delta_omega (float): the difference in angular velocity
        between the pole and equator. Note that what Butterpy
        calls the "shear" is delta_omega/omega.

    lat (float or array of floats): the latitudes at which to
        evaluate the angular velocity.

    Returns
    -------
    omega (same type as lat): angular velocity as function of latitude.
    """
    return omega_0 - delta_omega * np.sin(lat)**2

def solar(lat):
    """
    The solar differential rotation profile, from
    https://en.wikipedia.org/wiki/Solar_rotation.

    Parameters
    ----------
    lat (float or array of floats): latitutdes at which to
        evaluate angular velocity.

    Returns
    -------
    omega (same type as lat): angular velocity as function of latitude.
    """
    conv = np.pi/180/3600/24 # deg/day to rad/s
    a = 14.713 * conv
    b = 2.396 * conv  # note intentional sign change from wikipedia
    c = -1.787 * conv #    for consistency with sin2 function signs

    return sin2(a, b, lat) + c*np.sin(lat)**4 