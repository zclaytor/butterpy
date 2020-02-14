import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .regions import regions
from .spots import Spots, get_animation
from .constants import FLUX_SCALE

def _sun_test():
    ani = _star_test()
    return ani

def _star_test(activity_rate=1, cycle_length=11, cycle_overlap=2,
        decay_timescale=5, period=24.5, max_ave_lat=35, min_ave_lat=7):
    print('Generating spot evolution for the star...')
    star = regions(activity_rate=activity_rate, cycle_length=cycle_length,
        cycle_overlap=cycle_overlap, decay_time=(period*decay_timescale),
        max_ave_lat=max_ave_lat, min_ave_lat=min_ave_lat,
        alpha_med=(activity_rate*FLUX_SCALE))
    spots = Spots(star, alpha_med=(activity_rate*FLUX_SCALE), period=period,
        decay_timescale=decay_timescale)
    
    print('Generating light curve...')
    time = np.arange(0, 3650)
    flux = 1 + spots.calc(time)

    lc = pd.DataFrame(np.vstack([time, flux]).T, columns=['time', 'flux'])
    ani = spots.ortho_animation(time, lc)
    plt.show()

    return ani

def _test_animation():
    time = np.linspace(1040, 1070, 361)
    ani = get_animation(
        "/home/zach/PhD/tess_sim/lightcurves/0100.fits",
        time,
        projection='ortho',
        window_size=100,
        interval=20,
    )
    plt.show()

    #ani.save("/home/zach/Desktop/lightcurve.gif", writer="imagemagick", dpi=100, fps=10) 
    return ani