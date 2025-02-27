from astropy.io import fits


def to_fits(surface, filename, **kw):
    """
    Parameters
    ----------

    Returns None.
    """
    surface.assert_regions()
    surface.assert_lightcurve()

    p = set_sim_keywords(surface)
    s = fits.table_to_hdu(surface.regions)
    s.name = "regions"
    l = set_lightcurve_keywords(surface)

    hdul = fits.HDUList([p, s, l])
    hdul.writeto(filename, **kw)


def to_hlsp(surface, filename, hlsp_kw=None, **kw):
    """
    Parameters
    ----------

    Returns None.
    """
    surface.assert_regions()
    surface.assert_lightcurve()
    surface.assert_wavelet()

    p = fits.PrimaryHDU()

    if hlsp_kw is not None:
        # These keywords are required for MAST HLSP
        p = set_hlsp_keywords(p, hlsp_kw)

    p = set_sim_keywords(surface, hdu=p)
    l = set_lightcurve_keywords(surface)
    w = set_wavelet_keywords(surface)

    hdul = fits.HDUList([p, l, w])
    with open(filename, "wb") as f:
        hdul.writeto(f, **kw)


def set_hlsp_keywords(hdu, hlsp_kw):
    mission = hlsp_kw.pop("TELESCOP")
    if mission not in ("TESS"):
        raise NotImplementedError(f"Mission {mission} keywords not implemented.")
    
    sim_number = hlsp_kw.pop("sim_number")
    
    hdu.header["DATE-BEG"] = "2018-07-25T19:29:42.708Z", "ISO-8601 formatted DateTime for obs start"
    hdu.header["DATE-END"] = "2019-07-17T20:29:29.973Z", "ISO-8601 formatted DateTime for obs end"
    hdu.header["DATE-AVG"] = "2019-01-20T07:59:36.3405Z", "ISO-8601 formatted DateTime for obs mid"
    hdu.header["DOI"] = "10.17909/davg-m919", "Digital Object Identifier for HLSP data"
    hdu.header["HLSPID"] = "SMARTS", "identifier for this HLSP collection"
    hdu.header["HLSPLEAD"] = "Zachary R. Claytor", "HLSP project lead"
    hdu.header["HLSPNAME"] = "Stellar Magnetism, Activity, and Rotation with Time Series", "title"
    hdu.header["HLSPTARG"] = f"SMARTS-TESS-v1.0-{sim_number:06d}", "target designation"
    hdu.header["HLSPVER"] = 1.0, "version identifier"
    hdu.header["INSTRUME"] = "TESS", "instrument designation"
    hdu.header["LICENSE"] = "CC BY 4.0", "license for use of these data"
    hdu.header["LICENURL"] = "https://creativecommons.org/licenses/by/4.0/", "data license URL"
    hdu.header["OBSERVAT"] = "TESS", "observatory used to inform simulation"
    hdu.header["REFERENC"] = "2021arXiv210414566C", "ADS bibcode for data reference"
    hdu.header["TELAPSE"] = 357.04151926726604, "[d] time elapsed between start- and end-time"
    hdu.header["TELESCOP"] = mission, "telescope used to inform simulation"
    hdu.header["XPOSURE"] = 1800, "[s] exposure time"
    hdu.header["SIMULATD"] = True, "simulated, T (true) or F (false)"
    return hdu

def set_sim_keywords(surface, hdu=None):
    if hdu is None:
        hdu = fits.PrimaryHDU()

    hdu.header["PERIOD"] = surface.period, "equatorial rotation period in days"
    hdu.header["ACTIVITY"] = surface.activity_level, "solar-normalized activity level"
    hdu.header["CYCLE"] = surface.cycle_period, "magnetic cycle length in years"
    hdu.header["OVERLAP"] = surface.cycle_overlap, "magnetic cycle overlap in years"
    hdu.header["INCL"] = surface.inclination, "inclination of equator to line of sight in rad"
    hdu.header["MINLAT"] = surface.min_lat, "minimum latitude of spot emergence in deg"
    hdu.header["MAXLAT"] = surface.max_lat, "maximum latitude of spot emergence in deg"
    hdu.header["DIFFROT"] = surface.shear, "lat. rotation shear, normalized to equator"
    hdu.header["TSPOT"] = surface.tau_decay, "spot decay time normalized by period"
    hdu.header["BFLY"] = surface.butterfly, "spots emerge like butterfly (T) or random (F)"
    if surface.tsurf is not None:
        hdu.header["TSURF"] = surface.tsurf, "ambient surface temperature in K"
        hdu.header["TSPOT"] = surface.tspot, "spot temperature in K"
    return hdu

def set_lightcurve_keywords(surface):
    lc = surface.lightcurve
    hdu = fits.BinTableHDU(
        data=lc,
        name="lightcurve",
    )
    
    filters = lc.filters
    if filters is not None:
        if isinstance(filters, list):
            hdu.header["FILTER"] = "MULTI", "name of filter used"
            if len(filters) < 10:
                fmt = "d"
            else:
                fmt = "02d"
            for i, f in enumerate(filters):
                hdu.header[f"FILTER{i:{fmt}}"] = f
        else:
            hdu.header["FILTER"] = filters
    else:
        hdu.header["FILTER"] = "NONE", "name of filter used"
    hdu.header["BUNIT"] = "RELATIVE", "brightness or flux unit"
    return hdu

def set_wavelet_keywords(surface, pmin=0.1, pmax=180):
    wt = surface.wavelet_power
    w = fits.ImageHDU(wt, name="wavelet")
    w.header["PMIN"] = pmin, "minimum period bin edge in days"
    w.header["PMAX"] = pmax, "maximum period bin edge in days"
    return w