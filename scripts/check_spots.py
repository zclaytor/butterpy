import numpy as np
import matplotlib.pyplot as plt

from butterpy.core import regions, spots
from butterpy import regions as regions_dep, Spots as spots_dep


if __name__ == "__main__":
    np.random.seed(88)
    r = regions(activityrate=2, minlat=10, maxlat=50, cyclelength=3, cycleoverlap=1, ndays=3600, butterfly=True)
    print(len(r), len(r)*36000*8/1_000_000_000)
    s = spots(r, period=27, delta_omega=0.2)
    
    #np.random.seed(88)
    #r2 = regions_dep(activity_rate=10, min_ave_lat=10, max_ave_lat=50, cycle_length=3, cycle_overlap=1, tsim=3600, butterfly=True)
    #s2 = spots_dep(r2, alpha_med=1e-4)

    time = np.arange(0, 3600, 0.1)
    
    flux = s.calc(time)
    #flux = 1 + df.sum(axis=0)
    #flux2 = 1 + s2.calc(time)

    plt.plot(
        time, flux, 
        #time, flux2
    )
    plt.xlabel("time")
    plt.ylabel("flux")
    plt.show()