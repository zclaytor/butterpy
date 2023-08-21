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
    if is_smarts:
        p = set_smarts_keywords(p, smarts_kw)

    p.header["PERIOD"] = surface.period, "equatorial rotation period in days"
    p.header["ACTIVITY"] = surface.activity_level, "solar-normalized activity level"
    p.header["CYCLE"] = surface.cycle_period, "magnetic cycle length in years"
    p.header["OVERLAP"] = surface.cycle_overlap, "magnetic cycle overlap in years"
    p.header["INCL"] = surface.incl, "inclination of equator to line of sight in rad"
    p.header["MINLAT"] = surface.min_lat, "minimum latitude of spot emergence in deg"
    p.header["MAXLAT"] = surface.max_lat, "maximum latitude of spot emergence in deg"
    p.header["DIFFROT"] = surface.shear, "lat. rotation shear, normalized to equator"
    p.header["TSPOT"] = surface.tau_decay, "spot decay time normalized by period"
    p.header["BFLY"] = surface.butterfly, "spots emerge like butterfly (T) or random (F)"

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


def set_smarts_keywords(p, smarts_kw):
    j = smarts_kw.pop("j")
    
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
