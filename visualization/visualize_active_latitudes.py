"""
Visualization tools to illustrate the active latitudes and spot emergence as a
function of time.

Intended for use as a script:

```
>>> python visualization/visualize_active_latitudes.py
```
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table

from butterpy.utils.activelat import random_latitudes, exponential_latitudes

D2S = 1*u.day.to(u.s)

PROT_SUN = 27.0
OMEGA_SUN = 2 * np.pi / (PROT_SUN * D2S)


def butterfly(r, lats, minlat, maxlat):
    '''Plot the stellar butterfly pattern '''
    lat = 90 - r['theta']*180/np.pi

    plt.figure()
    plt.scatter(r['nday'], lat,
        s=r['bmax']/np.median(r['bmax'])*10,
        alpha=0.5, c='#996699', lw=0.5)
    plt.xlim(0, r['nday'].max())
    plt.xlabel('Time (days)')
    plt.ylabel('Latitude (deg)')
    plt.tight_layout();

    cycles = set(lats['ncycle'])
    lats = lats.to_pandas()
    for c in cycles:
        l = lats.query("ncycle == @c")
        plt.plot(
            l['nday'], l['latavg'], "b",
            l['nday'], l['latavg']+l['latrms'], "b:",
            l['nday'], l['latavg']-l['latrms'], "b:",

            l['nday'], -l['latavg'], "b",
            l['nday'], -l['latavg']+l['latrms'], "b:",
            l['nday'], -l['latavg']-l['latrms'], "b:")
        
    plt.axhspan(minlat, maxlat, color="g", alpha=0.15)
    plt.axhspan(-maxlat, -minlat, color="g", alpha=0.15)


def regions(butterfly=True, activityrate=1.0, cyclelength=1.0,
    cycleoverlap=0.0, maxlat=40.0, minlat=5.0, ndays=1200):

    nbin=5 # number of area bins
    delt=0.5  # delta ln(A)
    amax=100  # orig. area of largest bipoles (deg^2)
    dcon = np.exp(0.5*delt)- np.exp(-0.5*delt)
    atm = 10*activityrate
    ncycle = 365 * cyclelength
    nclen = 365 * (cyclelength + cycleoverlap)

    fact = np.exp(delt*np.arange(nbin)) # array of area reduction factors
    ftot = fact.sum()                   # sum of reduction factors
    bsiz = np.sqrt(amax/fact)           # array of bipole separations (deg)
    tau1 = 5                            # first and last times (in days) for
    tau2 = 15                           #   emergence of an active region
    prob = 0.001                        # total probability for "correlation"
    nlon = 36                           # number of longitude bins
    nlat = 16                           # number of latitude bins
    tau = np.zeros((nlon, nlat, 2), dtype=int) + tau2
    dlon = 360 / nlon
    dlat = (maxlat-minlat)/nlat
    ncnt = 0
    ncur = 0
    spots = Table(names=('nday', 'theta', 'phi', 'width', 'bmax'),
        dtype=(int, float, float, float, float))
  
    lats = Table(names=('nday', 'ncycle', 'latavg', 'latrms'),
        dtype=(int, int, float, float))
    
    for nday in np.arange(ndays, dtype=int):
        # Emergence rates for correlated regions
        # Note that correlated emergence only occurs for the largest regions,
        # i.e., for bsiz[0]
        tau += 1
        index = (tau1 <= tau) & (tau < tau2)
        rc0 = np.where(index, prob/(tau2-tau1), 0)

        ncur = nday // ncycle # index of current active cycle

        for icycle in [0, 1]: # loop over current and previous cycle
            nc = ncur - icycle # index of current or previous cycle
            nstart = ncycle*nc # start day of cycle
            phase = (nday - nstart) / nclen # phase relative to cycle start day
            if not (0 <= phase <= 1): # phase outside of [0, 1] is nonphysical
                continue

            # Determine active latitude bins
            if butterfly:
                latavg, latrms = exponential_latitudes(minlat, maxlat, phase)
            else:
                latavg, latrms = random_latitudes(minlat, maxlat)

            lats.add_row([nday, nc, latavg, latrms])

            # Compute emergence probabilities
            
            # Emergence rate of largest uncorrelated regions (number per day,
            # both hemispheres), from Shrijver and Harvey (1994)
            ru0_tot = atm*np.sin(np.pi*phase)**2 * dcon/amax
            # Uncorrelated emergence rate per lat/lon bin, as function of lat
            jlat = np.arange(nlat, dtype=int)
            p = np.exp(-((minlat+dlat*(0.5+jlat)-latavg)/latrms)**2)
            ru0 = ru0_tot*p/(p.sum()*nlon*2)

            for k in [0, 1]: # loop over hemisphere and latitude
                for j in jlat:
                    r0 = ru0[j] + rc0[:, j, k] # rate per lon, lat, and hem
                    rtot = r0.sum() # rate per lat, hem
                    sumv = rtot * ftot
                    x = np.random.uniform()
                    if sumv > x: # emerge spot
                        # determine bipole size
                        nb = 0
                        sumb = rtot*fact[0]
                        while x > sumb:
                            nb += 1
                            sumb += rtot*fact[nb]
                        bsize = bsiz[nb]

                        # determine longitude
                        i = 0
                        sumb += (r0[0]-rtot)*fact[nb]
                        while x > sumb:
                            i += 1
                            sumb += r0[i]*fact[nb]
                        lon = dlon*(np.random.uniform() + i)
                        lat = minlat+dlat*(np.random.uniform() + j)

                        new_region = add_region(lon, lat, k, bsize)
                        spots.add_row([nday, *new_region])
                                               
                        ncnt += 1
                        if nb == 0:
                            tau[i, j, k] = 0
    print(ncnt, "spots")
    return spots, lats


def add_region(lon, lat, k, bsize):
    bmax = 2.5*bsize**2 
    
    # Convert angles to radians
    lat *= np.pi/180
    phi = lon*np.pi/180.
    bsize *= np.pi/180
    width = 4*np.pi/180

    # Compute bipole positions
    theta = 0.5*np.pi - lat + 2*k*lat # k determines hemisphere

    return theta, phi, width, bmax


if __name__ == "__main__":
    np.random.seed(88)
    
    minlat, maxlat = 15, 30
    r, l = regions(activityrate=1, minlat=minlat, maxlat=maxlat,
        cyclelength=3, cycleoverlap=1, ndays=3600, butterfly=True)

    butterfly(r, l, minlat, maxlat)
    plt.show()