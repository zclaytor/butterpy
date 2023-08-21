import os
import sys
import pandas as pd
import numpy as np

from astropy.io import fits
from astropy.table import Table

float_dtype = 'float32'
wt_dtype = 'uint8'

path = sys.argv[1]
task = int(sys.argv[2])
chunksize = 1000

data_path = os.path.join('/mnt/lts/nfs_fs02/ifastars_group/zclaytor', path)

"""
if mode == 'noiseless':
    wavelet_path = os.path.join(data_path, 'wavelet_arrays')
    lc_path = os.path.join(data_path, 'trimmed_lightcurves')
else:
    wavelet_path = os.path.join(data_path, f'{mode}_wavelet_arrays')
    lc_path = os.path.join(data_path, f'{mode}_lightcurves')
"""
mode = "noisy"
wavelet_path = os.path.join(data_path, f'{mode}_wavelet_arrays')
lc_path = os.path.join(data_path, f'{mode}_lightcurves')

def get_lightcurve(i):
    target = os.path.join(lc_path, f'{i//1000:03.0f}', f'{i:06.0f}lc.npy')
    flux = np.load(target).astype(float_dtype)
    time = np.arange(len(flux), dtype=float_dtype) / 48
    return time, flux

def get_wavelet(i):
    target = os.path.join(wavelet_path, f'{i//1000:03.0f}', f'{i:06.0f}wt.npy')
    return np.load(target).astype(wt_dtype)

def to_fits(i, sim, out_path, j=None):
    if j is None:
        # Don't re-number simulations unless j is set
        j = i

    p = fits.PrimaryHDU()
    p.header["DATE-BEG"] = "2018-07-25T19:29:42.708Z", "ISO-8601 formatted DateTime for obs start"
    p.header["DATE-END"] = "2019-07-17T20:29:29.973Z", "ISO-8601 formatted DateTime for obs end"
    p.header["DATE-AVG"] = "2019-01-20T07:59:36.3405Z", "ISO-8601 formatted DateTime for obs mid"
    p.header["DOI"] = "10.17909/davg-m919", "Digital Object Identifier for HLSP data"
    p.header["HLSPID"] = "SMARTS", "identifier for this HLSP collection"
    p.header["HLSPLEAD"] = "Zachary R. Claytor", "HLSP project lead"
    p.header["HLSPNAME"] = "Stellar Magnetism, Activity, and Rotation with Time Series", "title"
    p.header["HLSPTARG"] = f"SMARTS-TESS-v1.0-{j:06d}", "target designation"
    p.header["HLSPVER"] = 1.0, "version identifier"
    p.header["INSTRUME"] = "TESS", "instrument designation"
    p.header["LICENSE"] = "CC BY 4.0", "license for use of these data"
    p.header["LICENURL"] = "https://creativecommons.org/licenses/by/4.0/", "data license URL"
    p.header["OBSERVAT"] = "TESS", "observatory used to inform simulation"
    p.header["REFERENC"] = "2021arXiv210414566C", "ADS bibcode for data reference"
    p.header["TELAPSE"] = 357.04151926726604, "[d] time elapsed between start- and end-time"
    p.header["TELESCOP"] = "TESS", "telescope used to inform simulation"
    p.header["XPOSURE"] = 1800, "[s] exposure time"
    p.header["SIMULATD"] = True, "simulated, T (true) or F (false)"

    p.header["PERIOD"] = sim["Period"], "equatorial rotation period in days"
    p.header["ACTIVITY"] = sim["Activity Rate"], "solar-normalized activity level"
    p.header["CYCLE"] = sim["Cycle Length"], "magnetic cycle length in years"
    p.header["OVERLAP"] = sim["Cycle Overlap"], "magnetic cycle overlap in years"
    p.header["INCL"] = sim["Inclination"], "inclination of equator to line of sight in rad"
    p.header["MINLAT"] = sim["Spot Min"], "minimum latitude of spot emergence in deg"
    p.header["MAXLAT"] = sim["Spot Max"], "maximum latitude of spot emergence in deg"
    p.header["DIFFROT"] = sim["Shear"], "lat. rotation shear, normalized to equator"
    p.header["TSPOT"] = sim["Decay Time"], "spot decay time normalized by period"
    p.header["BFLY"] = sim["Butterfly"], "spots emerge like butterfly (T) or random (F)"

    time, flux = get_lightcurve(i)
    l = fits.BinTableHDU(
        data=Table([time, flux], names=["time", "flux"]),
        name="lightcurve",
    )
    l.header["BUNIT"] = "relative", "brightness or flux unit"
    l.header["FILTER"] = "TESS", "name of filter used"

    wt = get_wavelet(i)
    w = fits.ImageHDU(wt, name="wavelet")
    w.header["PMIN"] = 0.1, "minimum period bin edge in days"
    w.header["PMAX"] = 180, "maximum period bin edge in days"

    hdul = fits.HDUList([p, l, w])
    filepath = os.path.join(out_path, f"smarts-tess-v1.0-{j:06.0f}.fits")
    with open(filepath, "wb") as f:
        hdul.writeto(f, overwrite=True)

if __name__ == '__main__':
    sims = pd.read_csv(
        "hlsp_table.csv",
        index_col="N",
        skiprows=range(1,1+task*chunksize),
        nrows=chunksize)

    # Convert numerical columns to desired dtype
    sims.loc[:, sims.columns != 'Butterfly'] = sims.loc[:, sims.columns != 'Butterfly'].astype(float_dtype)

    out_path = os.path.join(data_path, 'hlsp', f'{task:03d}')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for j, row in sims.iterrows():
        to_fits(row["Simulation Number"], row, out_path, j=j)
