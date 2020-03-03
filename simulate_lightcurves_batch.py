import os
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from butterpy import regions, Spots
from butterpy.constants import RAD2DEG, PROT_SUN, FLUX_SCALE, DAY2MIN

from config import sim_dir
from config import Nlc, dur, cad
from generate import generate_simdata


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


def simulate(s, out_str):
    if os.path.exists(f"{sim_dir}/lc{out_str}.fits"):
        return 0

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
            dur=dur
        )

        dF = lc.calc(time)

    final = Table(np.c_[time, 1 + dF], names=["time", "flux"])
    hdu_lc = fits.BinTableHDU(final)
    hdu_spots = fits.BinTableHDU(Table.from_pandas(spot_properties))
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu_lc, hdu_spots])
    hdul.writeto(f"{sim_dir}/lc{out_str}.fits", overwrite=True)

    return 0

def read_simdata(path, nstart, nrows):
    try:
        sims = pd.read_csv(
            path, 
            skiprows=list(range(1, nstart+1)),
            nrows=nrows,
            index_col='Simulation Number',
        )
    except FileNotFoundError:
        print('Simulation Properties not found. Generating new file.')
        from generate import generate_simdata
        sims = generate_simdata() 
    return sims        

if __name__ == "__main__":
    np.random.seed(777)

    nrows = 1000
    task_N = int(sys.argv[1])
    nstart = task_N * nrows

    sim_props = os.path.join(sim_dir, 'simulation_properties.csv')
    sims = read_simdata(sim_props, nstart, nrows)

    num_digits = int(np.log10(Nlc))

    for jj, s in sims.iterrows():
        out_str = f'{jj:d}'.zfill(num_digits)
        out = simulate(s, out_str=out_str)
