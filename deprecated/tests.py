import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .regions import regions
from .spots import Spots, get_animation
from .constants import FLUX_SCALE

def sun_test():
    ani = star_test()
    return ani

def star_test(
    activity_rate=1, 
    cycle_length=11, 
    cycle_overlap=2, 
    inclination=1,
    spot_min=7, 
    spot_max=35, 
    period=24.5, 
    shear=0.2, 
    decay_time=5, 
    butterfly=True, 
    t1=0, 
    t2=3650, 
    tstep=1
):
    print('Generating spot evolution for the star...')
    star = regions(
        butterfly=butterfly, 
        activity_rate=activity_rate, 
        cycle_length=cycle_length, 
        cycle_overlap=cycle_overlap, 
        decay_time=(period*decay_time),
        max_ave_lat=spot_max, 
        min_ave_lat=spot_min,
        alpha_med=(activity_rate*FLUX_SCALE)
    )
    spots = Spots(
        star, 
        alpha_med=(activity_rate*FLUX_SCALE), 
        period=period,
        incl=inclination, 
        decay_timescale=decay_time, 
        diffrot_shear=shear
    )
    
    print('Generating light curve...')
    time = np.arange(t1, t2, tstep)
    flux = 1 + spots.calc(time)

    lc = pd.DataFrame(np.vstack([time, flux]).T, columns=['time', 'flux'])
    ani = spots.ortho_animation(time, lc)
    plt.show()

    return ani

def test_animation(path, t1=1000, t2=1365, tstep=1, verbose=False):
    time = np.arange(t1, t2, tstep)
    ani = get_animation(
        path,
        time,
        projection='ortho',
        window_size=100,
        interval=20,
        verbose=verbose,
    )
    plt.show()

    #ani.save("/home/zach/Desktop/lightcurve.gif", writer="imagemagick", dpi=100, fps=10) 
    return ani