"""
Run the same input to regions (old and new) several times.
Compare maximum/minimum spot latitudes and the number of spots emerged.
"""
import numpy as np

from butterpy import regions as regions1
from butterpy.core import regions as regions2


if __name__ == "__main__":
    np.random.seed(88)
    print("     l1A l1B |  l2A  l2B |   NA   NB")
    l1As = []
    l1Bs = []
    l2As = []
    l2Bs = []
    NAs = []
    NBs = []
    for i in range(100):
        r1 = regions1(activity_rate=1, min_ave_lat=10, max_ave_lat=50, cycle_length=3, cycle_overlap=1, tsim=3600, butterfly=True)
        r2 = regions2(activityrate=1, minlat=10, maxlat=50, cyclelength=3, cycleoverlap=1, ndays=3600, butterfly=True)
        r2 = r2.to_pandas().eval("lat = 90 - (thpos + thneg)*90/3.1416")

        latsA = r1.lat.abs()
        latsB = r2.lat.abs()
        
        l1A = latsA.min()
        l1As.append(l1A)

        l1B = latsB.min()
        l1Bs.append(l1B)

        l2A = latsA.max()
        l2As.append(l2A)

        l2B = latsB.max()
        l2Bs.append(l2B)

        NA = len(r1)
        NAs.append(NA)

        NB = len(r2)
        NBs.append(NB)

        print(
            f"{i:3d}: {l1A:.1f} {l1B:.1f} | {l2A:.1f} {l2B:.1f} | {NA:4d} {NB:4d}"
        )
    
    print(f"ave: {np.mean(l1As):.1f} {np.mean(l1Bs):.1f} | {np.mean(l2As):.1f} {np.mean(l2Bs):.1f} | {np.mean(NAs):.0f} {np.mean(NBs):.0f}")
    print(f"std: {np.std(l1As):.1f} {np.std(l1Bs):.1f} | {np.std(l2As):4.1f} {np.std(l2Bs):4.1f} | {np.std(NAs):4.0f} {np.std(NBs):4.0f}")