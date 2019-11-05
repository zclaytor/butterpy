from time import time
import numpy as np
from pandas import DataFrame as DF
from scipy.stats import truncnorm
from .constants import RAD2DEG, YEAR2DAY
import matplotlib.pyplot as plt

import logging

n_bins = 5  # number of area bins
delta_lnA = 0.5  # bin width in log-area
max_area = 100  # original area of largest bipoles (deg^2)

tau1 = 5  # first and last times (in days) for emergence of "correlated" regions
tau2 = 15  # why these numbers ??
prob = 0.001  # total probability for "correlation" # based on what ??
nlon = 36  # number of longitude bins
nlat = 16  # number of latitude bins

dcon = 2 * np.sinh(delta_lnA / 2)  # constant from integration over area bin

fact = np.exp(
    delta_lnA * np.arange(n_bins)
)  # array of area reduction factors, = [1, 1.64, 2.71, 4.48, 7.39]
ftot = fact.sum()  # sum of reduction factors
bipole_widths = np.sqrt(max_area / fact)  # array of bipole widths (deg)


def active_latitudes(min_ave_lat, max_ave_lat, phase, butterfly=True):
    if butterfly:
        return exponential_latitudes(min_ave_lat, max_ave_lat, phase)
    return random_latitudes(min_ave_lat, max_ave_lat, phase)


def exponential_latitudes(min_lat, max_lat, phase):
    # Based on Hathaway 2015, LRSP 12: 4
    phase_scale = 1 / np.log(max_lat / min_lat)
    lat_avg = max_lat * np.exp(-phase / phase_scale)
    # See Hathaway 2010, p. 37 for a discussion of the width
    lat_rms = max_lat / 5 - phase * (max_lat - min_lat) / 7

    return lat_avg, lat_rms


def random_latitudes(min_lat, max_lat, phase):
    lat_avg = (max_lat + min_lat) / 2 * np.ones_like(phase)
    lat_rms = (max_lat - min_lat) * np.ones_like(phase)

    return lat_avg, lat_rms


def regions(
    butterfly=True,
    activity_rate=1,
    cycle_length=11,
    cycle_overlap=2,
    max_ave_lat=35,
    min_ave_lat=7,
    tsim=3650,
    tstart=0,
):
    """ 
    Simulates the emergence and evolution of starspots. 
    Output is a list of active regions.

    PARAMETERS
    ----------
    butterfly = bool - have spots decrease from maxlat to minlat or be randomly located in latitude

    activityrate = Number of magnetic bipoles, normalized such that for the Sun, activityrate = 1.

    cycle_length - length of cycle in years (Sun is 11)

    cycle_overlap - overlap of cycles in years

    max_ave_lat = maximum average latitude of spot emergence (deg)

    min_ave_lat = minimum average latitutde of emergence (deg)

    tsim = how many days to emerge spots for

    tstart = First day to simulate bipoles

    Based on Section 4 of van Ballegooijen 1998, ApJ 501: 866
    and Schrijver and Harvey 1994, SoPh 150: 1S
    Written by Joe Llama (joe.llama@lowell.edu) V 11/1/16
    # Converted to Python 3 9/5/2017

    According to Schrijver and Harvey (1994), the number of active regions
    emerging with areas in the range [A, A+dA] in time interval dt is given by

        n(A, t) dA dt = a(t) A^(-2) dA dt,

    where A is the "initial" bipole area in square degrees, and t is the time
    in days; a(t) varies from 1.23 at cycle minimum to 10 at cycle maximum.

    The bipole area is the area with the 25-Gauss contour in the "initial"
    state, i.e., at the time of maximum development of the active region.
    The assumed peak flux density in the initial state is 100 G, and
    width = 0.4*bsiz.

    """
    amplitude = 10 * activity_rate  # why 10*activity rate?
    cycle_length_days = cycle_length * YEAR2DAY
    nclen = (cycle_length + cycle_overlap) * YEAR2DAY

    # tau is time since last emergence
    tau = np.ones((nlon, nlat, 2), dtype=np.long) * tau2

    # width of latitude and longitude bins
    # Let latitude evolution for butterfly diagram go some amount
    # `lat_width` above max_ave_lat and below min_ave_lat
    lat_width = 7  # degrees
    lat_max = max_ave_lat + lat_width
    lat_min = max(min_ave_lat - lat_width, 0)
    dlat = (lat_max - lat_min) / nlat
    dlon = 360 / nlon

    n_count = 0
    n_current_cycle = 0

    spots = {}

    Nday, Icycle = np.mgrid[0:tsim, 0:2].reshape(2, 2 * tsim)
    n_current_cycle = Nday // cycle_length_days
    Nc = n_current_cycle - Icycle
    Nstart = np.fix(cycle_length_days * Nc)
    phase = (Nday - Nstart) / nclen % 1
    # Emergence rate of uncorrelated active regions, from
    # Schrijver & Harvey (1994)
    ru0_tot = amplitude * np.sin(np.pi * phase) ** 2 * dcon / max_area

    latavg, latrms = active_latitudes(
        min_ave_lat, max_ave_lat, phase, butterfly=butterfly
    )
    lat_bins = np.arange(nlat)
    lat_bins_matrix = np.outer(lat_bins, np.ones(len(Nday)))
    # Probability of spot emergence is gaussian-distributed about latavg with scatter latrms
    p = np.exp(-((lat_min + (lat_bins_matrix + 0.5) * dlat - latavg) / latrms) ** 2)

    for i_count, nday in enumerate(Nday):
        tau += 1

        # Emergence rate of correlated active regions
        rc0 = np.zeros((nlon, nlat, 2))
        index = (tau1 < tau) & (tau < tau2)
        if index.any():
            rc0[index] = prob / (tau2 - tau1)

        psum = p[:, i_count].sum()
        if psum == 0:
            ru0 = p[:, i_count]
        else:
            ru0 = ru0_tot[i_count] * p[:, i_count] / (2 * nlon * psum)

        # k = 0: Northern hemisphere; k = 1: Southern hemisphere
        for k in [0, 1]:
            for j in lat_bins:
                r0 = ru0[j] + rc0[:, j, k]
                rtot = r0.sum()
                sumv = rtot * ftot
                
                x = np.random.uniform()
                if sumv > x:  # emerge spot
                    # Add rtot*fact elements until the sum is greater than x
                    cum_sum = rtot * fact.cumsum()
                    nb = (cum_sum >= x).argmax()
                    sumb = cum_sum[nb - 1]

                    cum_sum = sumb + fact[nb] * r0.cumsum()
                    i = (cum_sum >= x).argmax()
                    sumb = cum_sum[i]

                    lon = dlon * (np.random.uniform() + i)
                    lat = lat_min + dlat * (np.random.uniform() + j)

                    if nday > tstart:
                        flux_distribution_width = 0.4 * bipole_widths[nb]
                        width_threshold = 4.0

                        # Eq. 15: B_r ~= B_max * (flux_distribution_width/width_threshold)^2
                        Bmax = (
                            250
                        )  # Solar-calibrated initial peak flux density in gauss
                        peak_magnetic_flux = (
                            Bmax * (flux_distribution_width / width_threshold) ** 2
                        )

                        lat_rad = lat / RAD2DEG
                        lon_rad = lon / RAD2DEG

                        spots[n_count] = {
                            "nday": nday,
                            "lat": (1 - 2 * k) * lat_rad,  # +lat for N, -lat for S
                            "lon": lon_rad,
                            "bmax": peak_magnetic_flux,
                        }

                        n_count += 1
                        if nb < 1:
                            tau[i, j, k] = 0

    return DF.from_dict(spots, orient="index")
