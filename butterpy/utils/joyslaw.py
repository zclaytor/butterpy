import numpy as np


def _symtruncnorm(v):
    """
    Symmetric truncated normal random variable. Returns a value drawn
    from a normal distribution truncated between +/- v.

    Note that scipy.stats.truncnorm exists for this, but as of 
    my current version of scipy (1.9.3), this implementation is almost
    100 times faster on a single draw.
    """
    while True:
        x = np.random.normal()
        if np.abs(x) < v:
            return x
        

def tilt(lat):
    """
    Active region tilt following Joy's Law, as implemented in the original ipython notebook.

    The behavior is that 86% of the time, the tilt angle is proportional to the latitude as:

        ang = lat/2 + 2 (in degrees)

    with non-gaussian scatter of roughly 15 degrees. The other 14% of the time, the tilt
    angle is random but near zero, drawn from a truncated normal distribution 
    between -0.5 and 0.5 degrees.

    NOTE: In the original implementation, the `else` block was converted to radians, but the
    output of this function also gets converted to radians... so I've removed the first
    conversion from this function.

    For tilt angles, see 
        Wang and Sheeley, Sol. Phys. 124, 81 (1989)
        Wang and Sheeley, ApJ. 375, 761 (1991)
        Howard, Sol. Phys. 137, 205 (1992)
    """
    z = np.random.uniform()
    if z > 0.14:
        # 86% of the time, the angle is proportional to the latitude with some scatter.
        x = _symtruncnorm(1.6)
        y = _symtruncnorm(1.8)
        return (0.5*lat + 2.0) + 27*x*y
    else:
        # The other 14%, the angle is near zero, randomly between -0.5 and 0.5 deg.
        return _symtruncnorm(0.5)