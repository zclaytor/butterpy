import os
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from butterpy import regions, Spots, tests
from butterpy.constants import RAD2DEG, PROT_SUN, FLUX_SCALE, DAY2MIN

from config import sim_dir, simulation_properties_dir
from config import Nlc, dur, cad
from generate import generate_simdata


def simulate(s, fig, ax, out_str):
    spot_properties = regions(
        butterfly=s["Butterfly"],
        activity_rate=s["Activity Rate"],
        cycle_length=s["Cycle Length"],
        cycle_overlap=s["Cycle Overlap"],
        max_ave_lat=s["Spot Max"],
        min_ave_lat=s["Spot Min"],
        alpha_med=np.sqrt(s["Activity Rate"]) * FLUX_SCALE,
        decay_time=s["Decay Time"]*s["Period"],
        tsim=dur,
        tstart=0,
    )

    time = np.arange(0, dur, cad / DAY2MIN)

    if len(spot_properties) == 0:
        dF = np.zeros_like(time)

    else:
        lc = Spots(
            spot_properties,
            incl=s["Inclination"],
            period=s["Period"],
            diffrot_shear=s["Shear"],
            alpha_med=np.sqrt(s["Activity Rate"])*FLUX_SCALE,
            decay_timescale=s["Decay Time"],
        )

        dF = lc.calc(time)

    final = Table(np.c_[time, 1 + dF], names=["time", "flux"])
    hdu_lc = fits.BinTableHDU(final)
    hdu_spots = fits.BinTableHDU(Table.from_pandas(spot_properties))
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu_lc, hdu_spots])
    hdul.writeto(f"{sim_dir}/{out_str}.fits", overwrite=True)

    # Plot the butterfly pattern
    lat = spot_properties["lat"]

    ax[0].scatter(
        spot_properties["nday"],
        lat * RAD2DEG,
        s=spot_properties["bmax"] / np.median(spot_properties["bmax"]) * 10,
        alpha=0.5,
        c="#996699",
        lw=0.5,
    )
    ax[0].set_ylim(-90, 90)
    ax[0].set_yticks((-90, -45, 0, 45, 90))
    ax[0].set_ylabel("Latitude (deg)")
    ax[1].plot(final["time"], final["flux"], c="C2")
    ax[1].set_ylabel("Model Flux")
    ax[0].set_title(f"Simulation {out_str}")
    fig.savefig(f"{sim_dir}/{out_str}.png", dpi=150)

    for ii in range(0, len(ax)):
        ax[ii].cla()

    return 0

def read_simdata(path=simulation_properties_dir):
    try:
        sims = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            'Missing simulation properties file. '
            'Try running with keyword "new".')
    return sims        

if __name__ == "__main__":
    np.random.seed(777)

    if len(sys.argv) > 1:
        if sys.argv[1] == "new":
            sims = generate_simdata()

        elif sys.argv[1] == "test":
            sims = read_simdata(simulation_properties_dir)
            
            testno = int(sys.argv[2])

            print(f'Testing simulation {testno:04d}:')
            print(sims.iloc[testno])
            tests.test_animation(os.path.join(sim_dir, f'{testno:04d}.fits'),
                t1=0, t2=3000)
            exit()

    else:
        sims = read_simdata(simulation_properties_dir)

    # Make the light curves
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8.5, 6))
    ax[0].set_ylabel("Flux")
    ax[0].set_title("Test")
    ax[-1].set_xlabel("Test")
    ax[0].set_ylim(-1e4, 1e4)
    fig.tight_layout()
    for ii in range(len(ax)):
        ax[ii].cla()

    pbar = tqdm(total=Nlc)
    for jj, s in sims.iterrows():
        out = simulate(s, fig, ax, out_str=f"{jj:04d}")
        pbar.update(1)
