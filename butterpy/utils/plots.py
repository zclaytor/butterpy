import numpy as np
import matplotlib.pyplot as plt


def butterfly(r):
    """Plot the butterfly diagram of a regions table.
    """
    thcen = 90*(1 - (r["thpos"] + r["thneg"])/np.pi)
    ndays = r["nday"]

    plt.figure()
    plt.scatter(ndays/365, thcen, c="k", alpha=0.5)
    plt.xlabel("Time (years)")
    plt.ylabel("Latitude (deg)")
    plt.title(f"{len(r)} spots")
    plt.show()


def monthly_spot_number(spots, make_plot=True):
    """
    Compute and plot the monthly spot number.

    Following Hathaway (2015), LRSP 12: 4, we compute the number of spots
    emerging in a particular month, then smooth the time series with a
    sliding 13-month window, centered on each month. This corresponds to 
    the "International Sunspot Number" (see section 3.1).

    Parameters
    ----------
    spots (astropy Table):
        The output of `regions` containing the table of star spots.

    make_plot (bool, optional, default=True):
        Whether to make and display the plot.

    Returns
    -------
    T (list):
        The list of times corresponding to each spot count.

    N (list):
        The list of spot counts.
    """
    days = spots["nday"]
    bin_days = 30 

    # first bin spot counts by month
    d = bin_days/2
    time = []
    nspots = []
    while d < days.max():
        time.append(d)
        nspots.append(sum((d-bin_days/2 < days) & (days <= d+bin_days/2)))
        d += bin_days
    
    # smooth monthly spot counts
    time = np.array(time)
    nspots = np.array(nspots)
    bin_days = 30*13
    d = bin_days/2 
    T = []
    N = []
    while d < days.max():
        T.append(d/365)
        N.append(np.mean(nspots[((d-bin_days/2) < time) & (time <= (d+bin_days/2))]))
        d += bin_days/13

    if make_plot:
        plt.figure()
        plt.plot(time/365, nspots, "ko", ms=3, label="Monthly Spot Count")
        plt.plot(T, N, "r", label="13-Month Smoothing")
        plt.xlabel("Time (years)")
        plt.ylabel("Monthly Sunspot Number")
        plt.legend()
        plt.show()

    return T, N