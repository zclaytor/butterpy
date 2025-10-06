import numpy as np
import matplotlib.pyplot as plt
import butterpy as bp

np.random.seed(88)
f = bp.Flutter(1000, duration=4*365)

outfile = open("diagnostics.csv", "w+")

print(f"idx,  act, range, tcyc, nspots, fmax", file=outfile)

for i, row in f.DataFrame.iterrows():
    print(i, end="\r")
    
    s = bp.Surface()

    r = s.emerge_regions(
        ndays=f.duration,
        activity_level=row["activity_level"],
        butterfly=row["butterfly"],
        cycle_period=row["cycle_period"],
        cycle_overlap=row["cycle_overlap"],
        max_lat=row["max_lat"],
        min_lat=row["min_lat"])

    nspots = len(r)
    rng = row["max_lat"] - row["min_lat"]

    if len(r) == 0:
        print(f"{i:3d}, {row['activity_level']:4.2f}, {rng:5.2f}, {row['cycle_period']:4.1f}, {nspots:6d},",
             file=outfile)
        continue

    time = np.arange(0, f.duration, f.cadence) + f.cadence

    l = s.evolve_spots(
        time=time,
        inclination=row["inclination"], 
        period=row["period"],
        shear=row["shear"], 
        tau_evol=row["tau_evol"])

    fmin = 1-l.flux.min() 
    print(f"{i:3d}, {row['activity_level']:4.2f}, {rng:5.2f}, {row['cycle_period']:4.1f}, {nspots:6d}, {fmin:.2e}",
         file=outfile)
    #continue

    s.plot_butterfly()
    plt.savefig(f"plots/bfly{i:03d}.png")

    s.plot_lightcurve()
    plt.savefig(f"plots/lc{i:03d}.png")

    plt.close("all")
    s.to_fits(f"fits/sim{i:03d}.fits")