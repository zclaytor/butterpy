import numpy as np
import matplotlib.pyplot as plt


def spot_counts(spots, bin_size=30, make_plot=True):
    """
    Compute and plot the monthly spot number.

    Following Hathaway (2015), LRSP 12: 4, we compute the number of spots
    emerging in a particular month, then smooth the time series with a
    sliding 13-month window, centered on each month. This corresponds to 
    the "International Sunspot Number" (see section 3.1).

    This function generalizes the spot count window size, but keeps 
    the month window by default.

    Parameters
    ----------
    spots (astropy Table):
        The output of `regions` containing the table of star spots.

    bin_size (float, 30):
        The window length in days for the spot count.

    make_plot (bool, optional, default=True):
        Whether to make and display the plot.

    Returns
    -------
    T (array):
        The times corresponding to each spot count.

    N (array):
        The monthly spot counts.
    """
    days = spots["nday"]

    # first bin spot counts by month
    d = bin_size/2
    time = []
    nspots = []
    while d < days.max():
        time.append(d)
        nspots.append(sum((d-bin_size/2 < days) & (days <= d+bin_size/2)))
        d += bin_size
    
    # smooth monthly spot counts
    time = np.array(time)
    nspots = np.array(nspots)
    bin_size *= 13
    d = bin_size/2 
    T = []
    N = []
    while d < days.max():
        T.append(d/365)
        N.append(np.mean(nspots[((d-bin_size/2) < time) & (time <= (d+bin_size/2))]))
        d += bin_size/13

    if make_plot:
        plt.figure()
        plt.plot(time/365, nspots, "ko", ms=3, label="Monthly Spot Count")
        plt.plot(T, N, "r", label="13-Month Smoothing")
        plt.xlabel("Time (years)")
        plt.ylabel("Monthly Sunspot Number")
        plt.legend()
        plt.show()

    return np.array(T), np.array(N)