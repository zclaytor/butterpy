"""
Utilities for Solar normalization.

Contains
--------
solar_regions (function):
    Runs the `regions` function with Solar inputs.

fit_spot_counts (function):
    Computes spot counts and fits a squared sine function
    to the monthly spot count.

NOTE
----
According to Hathaway (2015), LRSP 12: 4, Figure 2, the peak monthly
    sunspot number varies from cycle to cycle, with peaks as low as 
    50 spots/month (1818, Cycle 6) and as high as 200 (1960, Cycle 19).
    The Solar normalization should return an amplitude somewhere in this range.

NOTE
----
This currently computes the normalization based on *newly* emerged spots,
    which is different from daily sunspot counts, which can include spots
    that were previously counted. To do the normalization right, we need
    to run the regions through the `spots` code and count the spots that
    are visible on a given day.
"""

import numpy as np
import matplotlib.pyplot as plt

from butterpy.core import Surface
from plots import spot_counts


def solar_regions():
    """
    Runs the `regions` code with Solar inputs:

    activityrate = 1,
    minlat = 7,
    maxlat = 35,
    cyclelength = 11,
    cycleoverlap = 1,
    ndays = 100 years
    """
    sun = Surface()
    return sun.emerge_regions(activity_level=1, min_lat=7, max_lat=35, 
        cycle_period=11, cycle_overlap=1, ndays=365*100)


def fit_spot_counts(spots, make_plot=True):
    """
    Computes spot counts and fits the function

        A sin^2(pi*t/P + phi),
    
    where A is the amplitude, t is time, P is the cycle period,
    and phi is a temporal shift. The form of this function is chosen
    to resemble the form of the emergence rate, defined in `regions`.

    Parameters
    ----------
    spots (astropy Table):
        The output of `regions` containing the table of star spots.

    make_plot (bool, optional, default=True):
        Whether to make and display the plot.

    Returns
    -------
    A (float):
        The spot count amplitude with units of spots/month
    """
    time, nspots = spot_counts(spots, bin_size=30, make_plot=False)

    # Fit Asin^2(pi*t/P + phi), based on emergence rate definition.
    from scipy.optimize import curve_fit

    def f(t, A, P, phi):
        return A*np.sin(np.pi*t/P + phi)**2
    
    T = np.array(time)
    N = np.array(nspots)
    popt, _ = curve_fit(f, T, N, p0=(25, 11, 0))
    A = popt[0]

    if make_plot:
        plt.figure()
        plt.plot(T, N, "k", label="Smoothed Spot Count")
        plt.plot(T[::10], f(T[::10], *popt), "bo", ms=8, label=f"Model with A={A:.2f}")
        plt.xlabel("Time (years)")
        plt.ylabel("Monthly Sunspot Number")
        plt.legend()
        plt.show()
    return A

if __name__ == "__main__":
    np.random.seed(88)
    sun = solar_regions()
    A = fit_spot_counts(sun, make_plot=True)
    print(f"amplitude: {A} spots/month")