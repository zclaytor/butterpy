import os
import sys
import pandas as pd
import numpy as np

from astropy.io import fits
from astropy.table import Table

float_dtype = 'float32'
wt_dtype = 'uint8'


def to_fits(surface, filename, is_smarts=False, smarts_kw=None, **kw):
    """
    Parameters
    ----------

    Returns None.
    """
    p = fits.PrimaryHDU()

    if is_smarts:
        # These keywords are required for MAST HLSP
        p = set_smarts_keywords(p, smarts_kw)

    p = set_sim_keywords(p, surface)

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
    with open(filename, "wb") as f:
        hdul.writeto(f, **kw)


def set_smarts_keywords(hdu, smarts_kw):
    j = smarts_kw.pop("j")
    
    hdu.header["DATE-BEG"] = "2018-07-25T19:29:42.708Z", "ISO-8601 formatted DateTime for obs start"
    hdu.header["DATE-END"] = "2019-07-17T20:29:29.973Z", "ISO-8601 formatted DateTime for obs end"
    hdu.header["DATE-AVG"] = "2019-01-20T07:59:36.3405Z", "ISO-8601 formatted DateTime for obs mid"
    hdu.header["DOI"] = "10.17909/davg-m919", "Digital Object Identifier for HLSP data"
    hdu.header["HLSPID"] = "SMARTS", "identifier for this HLSP collection"
    hdu.header["HLSPLEAD"] = "Zachary R. Claytor", "HLSP project lead"
    hdu.header["HLSPNAME"] = "Stellar Magnetism, Activity, and Rotation with Time Series", "title"
    hdu.header["HLSPTARG"] = f"SMARTS-TESS-v1.0-{j:06d}", "target designation"
    hdu.header["HLSPVER"] = 1.0, "version identifier"
    hdu.header["INSTRUME"] = "TESS", "instrument designation"
    hdu.header["LICENSE"] = "CC BY 4.0", "license for use of these data"
    hdu.header["LICENURL"] = "https://creativecommons.org/licenses/by/4.0/", "data license URL"
    hdu.header["OBSERVAT"] = "TESS", "observatory used to inform simulation"
    hdu.header["REFERENC"] = "2021arXiv210414566C", "ADS bibcode for data reference"
    hdu.header["TELAPSE"] = 357.04151926726604, "[d] time elapsed between start- and end-time"
    hdu.header["TELESCOP"] = "TESS", "telescope used to inform simulation"
    hdu.header["XPOSURE"] = 1800, "[s] exposure time"
    hdu.header["SIMULATD"] = True, "simulated, T (true) or F (false)"
    return hdu

def set_sim_keywords(hdu, surface):
    hdu.header["PERIOD"] = surface.period, "equatorial rotation period in days"
    hdu.header["ACTIVITY"] = surface.activity_level, "solar-normalized activity level"
    hdu.header["CYCLE"] = surface.cycle_period, "magnetic cycle length in years"
    hdu.header["OVERLAP"] = surface.cycle_overlap, "magnetic cycle overlap in years"
    hdu.header["INCL"] = surface.incl, "inclination of equator to line of sight in rad"
    hdu.header["MINLAT"] = surface.min_lat, "minimum latitude of spot emergence in deg"
    hdu.header["MAXLAT"] = surface.max_lat, "maximum latitude of spot emergence in deg"
    hdu.header["DIFFROT"] = surface.shear, "lat. rotation shear, normalized to equator"
    hdu.header["TSPOT"] = surface.tau_decay, "spot decay time normalized by period"
    hdu.header["BFLY"] = surface.butterfly, "spots emerge like butterfly (T) or random (F)"
    return hdu

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
