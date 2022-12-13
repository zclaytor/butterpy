import numpy as np
import matplotlib.pyplot as plt
from butterpy.core_original import regions

if __name__ == "__main__":
    np.random.seed(88)
    r = regions(activityrate=1, minlat=30, maxlat=35, 
        cyclelength=3, cycleoverlap=1, ndays=3600)
    thcen = 90*(1 - (r["thpos"] + r["thneg"])/np.pi) #type:ignore
    ndays = r["nday"]

    plt.figure()
    plt.scatter(ndays/365, thcen, c="k", alpha=0.5) #type:ignore
    plt.xlabel("Time (years)")
    plt.ylabel("Latitude (deg)")
    plt.title(f"{len(r)} spots")
    plt.show()