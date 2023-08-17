import numpy as np
import matplotlib.pylab as plt
import astropy.units as u
from astropy.table import Table

from .utils.activelat import random_latitudes, exponential_latitudes
from .utils.spotevol import gaussian_spots
from .utils.diffrot import sin2
from .utils.joyslaw import tilt

D2S = 1*u.day.to(u.s)

PROT_SUN = 27.0
OMEGA_SUN = 2 * np.pi / (PROT_SUN * D2S)

raise DeprecationWarning("This version of the code is deprecated in favor of the implementation found in `core.py`.")

class spots(object):
    """Holds parameters for spots on a given star."""
    def __init__(
            self, 
            spot_properties, 
            alpha_med=0.0001,
            incl=90, 
            period=PROT_SUN,
            delta_omega=0.3, 
            diffrot_func=sin2,
            spot_evol=gaussian_spots,
            tau_evol=5.0, 
            threshold=0.1):
        """
        Generate initial parameter set for spots. Emergence times
        and initial locations are read from the user-provided 
        `spot_properties` table.

        Parameters
        ----------
        spot_properties (astropy Table): 
            Table containing spot properties such as emergence time, 
            latitude, longitude, and peak magnetic flux.

        alpha_med (float, optional, default=0.0001):
            Median value of an individual spot filling factor.

        incl (float, optional, default=90):
            Inclination angle of the star in degrees, where inclination is
            the angle between the pole and the line of sight.

        period (float, optional, default=PROT_SUN):
            Rotation period of the star in days.

        delta_omega (float, optional, default=0.3):
            Differential rotation rate of the star in units of equatorial
            rotation velocity.

        diffrot_func (function, optional, default=`utils.diffrot.sin2`):
            Differential rotation function. Default is sin^2 (latitude).

        spot_evol (function, optional, default=`utils.spotevol.gaussian_spots`):
            Spot evolution function. Default is double-sided gaussian with time.

        tau_evol (float, optional, default=5.0):
            Spot decay timescale in units of the equatorial rotation period.

        threshold (float, optional, default=0.1):
            Minimum peak magnetic flux for a spot to be considered.


        Returns None.
        """
        
        # set global stellar parameters which are the same for all spots
        # inclination
        self.spot_properties = spot_properties
        self.incl = incl * np.pi/180 # in radians
        # rotation and differential rotation
        self.period = period # in days
        self.omega = 2*np.pi/(self.period * D2S) # in radians/s
        self.delta_omega = delta_omega * self.omega # in radians/s
        self.diffrot_func = diffrot_func
        # spot emergence and decay
        self.spot_evol = spot_evol
        self.tau_em = min(2, self.period * tau_evol / 10)
        self.tau_decay = self.period * tau_evol
        # Convert spot properties
        tmax = spot_properties['nday']
        lat = 0.5*(spot_properties['thpos'] + spot_properties['thneg'])
        lat = np.pi/2 - lat
        l = lat < 0
        lat[l] *= -1
        lon = 0.5*(spot_properties['phpos'] + spot_properties['phneg'])
        Bem = spot_properties['bmax']

        # keep only spots with peak B-field > threshold
        l = Bem > threshold
        self.nspot = l.sum()
        self.tmax = tmax[l]
        self.lat = lat[l]
        self.lon = lon[l]
        self.amax = Bem[l] * alpha_med / np.median(Bem[l]) 
        # scale amax to achieve desired median alpha, 
        # where alpha = spot contrast * spot area

    def calci(self, time, i):
        """
        Helper function to evolve one spot and calculate its impact on the flux.

        Includes rotation and foreshortening (i.e., spots in the center cause
        more modulation than spots at the limb, and spots out of view do not
        contribute flux modulation). Also includes spot emergence and decay.

        Currently there is no spot drift or shear (within an active region).

        Parameters
        ----------
        time (numpy array):
            the array of time values at which to compute the flux modulation.

        i (int):
            spot index, which can have integer values of [0, self.nspot].

        Returns
        -------
        dF_i (numpy array):
            the time-varying flux modulation from spot `i`.
        """
        # Spot area
        tt = time - self.tmax[i]
        area = self.amax[i] * self.spot_evol(tt, self.tau_em, self.tau_decay)
        # Rotation rate
        omega_lat = self.diffrot_func(self.omega, self.delta_omega, self.lat[i])
        # Foreshortening
        phase = omega_lat * time * D2S + self.lon[i]
        beta = np.cos(self.incl) * np.sin(self.lat[i]) + \
            np.sin(self.incl) * np.cos(self.lat[i]) * np.cos(phase)
        # Differential effect on stellar flux
        dF_i = - area * beta
        dF_i[beta < 0] = 0
        return dF_i

    def calc(self, time):
        """
        Calculates the flux modulation for all spots.

        Includes rotation and foreshortening (i.e., spots in the center cause
        more modulation than spots at the limb, and spots out of view do not
        contribute flux modulation). Also includes spot emergence and decay.

        Currently there is no spot drift or shear (within an active region).

        Parameters
        ----------
        time (numpy array):
            the array of time values at which to compute the flux modulation.

        Returns
        -------
        lc (numpy array):
            the light curve: time-varying flux modulation from all spots.
        """
        lc = np.ones_like(time, dtype="float32")
        for i in np.arange(self.nspot):
            lc += self.calci(time, i)
        return lc

    def butterfly(self):
        '''Plot the stellar butterfly pattern '''
        lat = 0.5*(self.spot_properties['thpos'] + self.spot_properties['thneg'])
        lat = np.pi/2. - lat
        l = lat < 0
        plt.figure()
        plt.scatter(self.spot_properties['nday'],lat*u.rad.to(u.deg), \
            s=self.spot_properties['bmax']/np.median(self.spot_properties['bmax'])*10,
            alpha=0.5, c='#996699', lw=0.5)
        plt.xlim(0, self.spot_properties['nday'].max())
        plt.ylim(-90, 90)
        plt.yticks((-90, -45, 0, 45, 90))
        plt.xlabel('Time (days)')
        plt.ylabel('Latitude (deg)')
        plt.tight_layout();


def regions(butterfly=True, activityrate=1.0, cyclelength=1.0,
    cycleoverlap=0.0, maxlat=40.0, minlat=5.0, ndays=1200):
    """     
    Simulates the emergence and evolution of starspots. 
    Output is a Table of active regions.

    Parameters
    ----------
    butterfly (bool, optional, default=True): 
        Have spots decrease from maxlat to minlat (True)
        or be randomly located in latitude (False)

    activityrate (float, optional, default=1.0): 
        Number of magnetic bipoles, normalized such that 
        for the Sun, activityrate = 1.

    cyclelength (float, optional, default=1.0): 
        Interval (years) between cycle starts (Sun is 11)

    cycleoverlap (float, optional, default=1.0): 
        Overlap of cycles in years.

    maxlat (float, optional, default=40):
        Maximum latitude of spot emergence in degrees.

    minlat (float, optional, default=5):
        Minimum latitutde of spot emergence in degrees.

    ndays (int, optional, default=1200):
        Number of days to emerge spots. 

    Returns
    -------
    spots (astropy Table): 
        Each row is an active region with the following parameters:

        nday  = day of emergence
        thpos = theta of positive pole (radians)
        phpos = phi   of positive pole (radians)
        thneg = theta of negative pole (radians)
        phneg = phi   of negative pole (radians)
        width = width of each pole (radians)
        bmax  = maximum flux density (Gauss)

    Notes
    -----
    Based on Section 4 of van Ballegooijen 1998, ApJ 501: 866
    and Schrijver and Harvey 1994, SoPh 150: 1S.

    Written by Joe Llama (joe.llama@lowell.edu) V 11/1/16
    Converted to Python 3 9/5/2017

    According to Schrijver and Harvey (1994), the number of active regions
    emerging with areas in the range [A,A+dA] in a time dt is given by 

        n(A,t) dA dt = a(t) A^(-2) dA dt ,

    where A is the "initial" area of a bipole in square degrees, and t is
    the time in days; a(t) varies from 1.23 at cycle minimum to 10 at cycle
    maximum.

    The bipole area is the area within the 25-Gauss contour in the
    "initial" state, i.e. time of maximum development of the active region.
    The assumed peak flux density in the initial state is 1100 G, and
    width = 0.4*bsiz. The parameters are corrected for further diffusion and 
    correspond to the time when width = 4 deg, the smallest width that can be 
    resolved with lmax=63.

    In our simulation we use a lower value of a(t) to account for "correlated"
    regions.

    """
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
    if butterfly:                       # Really we want spots to emerge in a
        l1 = max(minlat-7, 0)           # range around the average active lat,
        l2 = min(maxlat+7, 90)          # so we bump the boundaries a bit.
    else:                               
        l1, l2 = minlat, maxlat
    dlat = (l2-l1)/nlat                 
    ncur = 0
    spots = Table(names=('nday', 'thpos', 'phpos','thneg','phneg', 'width', 'bmax', 'ang'),
        dtype=(int, float, float, float, float, float, float, float))

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

            # Compute emergence probabilities
            
            # Emergence rate of largest uncorrelated regions (number per day,
            # both hemispheres), from Shrijver and Harvey (1994)
            ru0_tot = atm*np.sin(np.pi*phase)**2 * dcon/amax
            # Uncorrelated emergence rate per lat/lon bin, as function of lat
            jlat = np.arange(nlat, dtype=int)
            p = np.exp(-((l1 + dlat*(0.5+jlat) - latavg)/latrms)**2)
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
                        lat = l1 + dlat*(np.random.uniform() + j)

                        new_region = add_region(nc, lon, lat, k, bsize)
                        spots.add_row([nday, *new_region])

                        if nb == 0:
                            tau[i, j, k] = 0
    return spots


def add_region(nc, lon, lat, k, bsize):
    """
    Add one active region of a particular size at a particular location.

    Joy's law tilt angle is computed here as well. 
    For tilt angles, see 
        Wang and Sheeley, Sol. Phys. 124, 81 (1989)
        Wang and Sheeley, ApJ. 375, 761 (1991)
        Howard, Sol. Phys. 137, 205 (1992)

    Parameters
    ----------
    nc (int): cycle index
    lon (float): longitude
    lat (float): latitude
    k (int): hemisphere index (0 for North, 1 for South)
    bsize (float): the size of the bipole

    Returns
    -------
    thpos (float): theta of positive bipole
    phpos (float): longitude of positive bipole
    thneg (float): theta of negative bipole
    phneg (float): longitude of negative bipole
    width (float): bipole width threshold, always 4...?
    bmax (float): magnetic field strength of bipole
    ang (float): Joy's law bipole angle
    """
    ic = 1. - 2.*(nc % 2) # +1 for even, -1 for odd cycle
    width = 4.0 # this is not actually used... remove?
    bmax = 2.5*bsize**2 # original was bmax = 250*(0.4*bsize / width)**2, this is equivalent
    ang = tilt(lat)
    
    # Convert angles to radians
    ang *= np.pi/180
    lat *= np.pi/180
    phcen = lon*np.pi/180.
    bsize *= np.pi/180
    width *= np.pi/180

    # Compute bipole positions
    dph = ic*0.5*bsize*np.cos(ang)/np.cos(lat)
    dth = ic*0.5*bsize*np.sin(ang)
    thcen = 0.5*np.pi - lat + 2*k*lat # k determines hemisphere
    phpos = phcen + dph
    phneg = phcen - dph
    thpos = thcen + dth
    thneg = thcen - dth

    return thpos, phpos, thneg, phneg, width, bmax, ang