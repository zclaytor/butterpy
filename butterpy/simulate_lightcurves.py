import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from .spots import Spots
from .regions import regions
from .generate import generate_simdata
from .config import sim_dir, simulation_properties_dir
from .config import Nlc, dur, cad
from .constants import RAD2DEG, PROT_SUN, FLUX_SCALE, DAY2MIN


def scale_tess_flux(f1, f2):
    """Routine to scale the TESS ETE simulations"""
    ii = f2 > 0
    keep = f2[~ii]
    jj = f1 > 0
    f2 -= np.median(f2[ii])
    f2 /= np.std(f2[ii])
    f2 *= np.std(f1[jj])
    f2 += np.median(f1[jj])
    f2[~ii] = keep
    return f2


def simulate(s, fig, ax, out_str):
    spot_properties = regions(
        butterfly=s["Butterfly"],
        activity_rate=s["Activity Rate"],
        cycle_length=s["Cycle Length"],
        cycle_overlap=s["Cycle Overlap"],
        max_ave_lat=s["Spot Max"],
        min_ave_lat=s["Spot Min"],
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
            alpha_med=np.sqrt(s["Activity Rate"]) * FLUX_SCALE,
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


if __name__ == "__main__":
    np.random.seed(777)

    if len(sys.argv) > 1:
        if sys.argv[1] == "new":
            sims = generate_simdata()

    else:
        try:
            sims = pd.read_csv(simulation_properties_dir)
        except FileNotFoundError:
            print("Missing simulation properties file.")
            print('Try running with keyword "new".')
            exit()

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
