import numpy as np

def random_latitudes(minlat, maxlat):
    latavg = (maxlat + minlat) / 2.
    latrms = (maxlat - minlat)
    return latavg, latrms

def exponential_latitudes(minlat, maxlat, phase):
    # Based on Hathaway 2015, LRSP 12: 4
    latavg = maxlat * (minlat/maxlat)**phase
    # See Hathaway 2010, p. 37 for a discussion of the width
    latrms = maxlat/5 - phase*(maxlat-minlat)/7

    return latavg, latrms