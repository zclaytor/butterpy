import numpy as np
import matplotlib.pylab as plt
import astropy.units as u
from astropy.table import Table

from .utils.activelat import random, exponential
from .utils.diffrot import sin2
from .utils.joyslaw import tilt

D2S = 1*u.day.to(u.s)

PROT_SUN = 27.0
OMEGA_SUN = 2 * np.pi / (PROT_SUN * D2S)


class spots(object):
    """Holds parameters for spots on a given star."""
    def __init__(self, spot_properties, dur=None, alpha_med=0.0001, 
        incl=np.pi/2, omega=2.0, delta_omega=0.3, diffrot_func=sin2,
        tau_evol=5.0, threshold=0.1):
        """
        Generate initial parameter set for spots. Emergence times
        and initial locations are read from the user-provided 
        `spot_properties` table.

        Parameters:
        ----------
        spot_properties (astropy Table): 
            DataFrame containing spot properties such as emergence time, 
            latitude, longitude, and peak magnetic flux.

        dur (float, optional):
            Duration of time to consider for spot emergence and decay.
            Default is the maximum emergence time in `spot_properties`.

        alpha_med (float, optional, default=0.0001):
            Median value of an individual spot filling factor.

        incl (float, optional, default=pi/2):
            Inclination angle of the star in radians, where inclination is
            the angle between the pole and the line of sight.

        omega (float, optional, default=2.0):
            Rotation rate of the star in solar units.

        delta_omega (float, optional, default=0.3):
            Differential rotation rate of the star in solar units.

        diffrot_func (function, optional, default=`utils.diffrot.sin2`):
            Differential rotation function. Default is sin^2 (latitude).

        tau_evol (float, optional, default=5.0):
            Spot decay timescale in units of the equatorial rotation period.

        threshold (float, optional, default=0.1):
            Minimum peak magnetic flux for a spot to be considered.


        Returns None.
        """
        
        # set global stellar parameters which are the same for all spots
        # inclination
        self.spot_properties = spot_properties
        self.incl = incl
        # rotation and differential rotation (supplied in solar units)
        self.omega = omega * OMEGA_SUN # in radians
        self.delta_omega = delta_omega * OMEGA_SUN
        self.per_eq = 2 * np.pi / self.omega / D2S # in days
        self.per_pole = 2 * np.pi / (self.omega - self.delta_omega) / D2S
        self.diffrot_func = diffrot_func
        # spot emergence and decay timescales
        self.tau_em = min(2.0, self.per_eq * tau_evol / 10.0)
        self.tau_decay = self.per_eq * tau_evol
        # Convert spot properties
        t0 = spot_properties['nday']
        lat = 0.5*(spot_properties['thpos'] + spot_properties['thneg'])
        lat = np.pi/2. - lat
        l = lat < 0
        lat[l] *= -1
        lon = 0.5*(spot_properties['phpos'] + spot_properties['phneg'])
        Bem = spot_properties['ang']

        # keep only spots emerging within specified time-span, with peak B-field > threshold
        if dur == None:
            self.dur = t0.max()
        else:
            self.dur = dur
        l = (t0 <= self.dur) * (Bem > threshold)
        self.nspot = l.sum()
        self.t0 = t0[l]
        self.lat = lat[l]
        self.lon = lon[l]
        self.amax = Bem[l] \
            * alpha_med #/ np.median(Bem[l]) # scale to achieve desired median alpha, # where alpha = spot contrast * spot area

    def calci(self, time, i):
        '''Evolve one spot and calculate its impact on the stellar flux'''
        '''NB: Currently there is no spot drift or shear'''
        # Spot area
        area = np.ones(len(time)) * self.amax[i]
        tt = time - self.t0[i]
        l = tt<0
        area[l] *= np.exp(-tt[l]**2 / 2. / self.tau_em**2) # emergence
        l = tt>0
        area[l] *= np.exp(-tt[l]**2 / 2. / self.tau_decay**2) # decay
        # Rotation rate
        ome = self.diffrot_func(self.omega, self.delta_omega, self.lat[i])
        # Fore-shortening
        phase = ome * time * D2S + self.lon[i]
        beta = np.cos(self.incl) * np.sin(self.lat[i]) + \
            np.sin(self.incl) * np.cos(self.lat[i]) * np.cos(phase)
        # Differential effect on stellar flux
        dF = - area * beta
        dF[beta < 0] = 0
        return area, ome, beta, dF

    def calc(self, time):
        '''Calculations for all spots'''
        N = len(time)
        M = self.nspot
        area = np.zeros((M, N))
        ome = np.zeros(M)
        beta = np.zeros((M, N))
        dF = np.zeros((M, N))
        for i in np.arange(M):
            area_i, omega_i, beta_i, dF_i = self.calci(time, i)
            area[i,:] = area_i
            ome[i] = omega_i
            beta[i,:] = beta_i
            dF[i,:] = dF_i
        return area, ome, beta, dF

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
    Output is a list of active regions.

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
    width = 0.4*bsiz (see disp_region). The parameters are corrected for 
    further diffusion and correspond to the time when width = 4 deg, the 
    smallest width that can be resolved with lmax=63.

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
    dlat = (maxlat-minlat)/nlat
    ncnt = 0
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
                latavg, latrms = exponential(minlat, maxlat, phase)
            else:
                latavg, latrms = random(minlat, maxlat)

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

                        new_region = add_region(nc, lon, lat, k, bsize)
                        spots.add_row([nday, *new_region])

                        ncnt += 1
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
    width = 4.0 # this is no longer needed... remove?
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