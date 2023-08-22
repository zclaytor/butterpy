import pickle

import numpy as np
import matplotlib.pylab as plt
import astropy.units as u
from astropy.table import Table

from .utils.activelat import random_latitudes, exponential_latitudes
from .utils.spotevol import gaussian_spots
from .utils.diffrot import sin2
from .utils.joyslaw import tilt

from .io.pkl import pickle, unpickle


D2S = 1*u.day.to(u.s)

PROT_SUN = 27.0
OMEGA_SUN = 2 * np.pi / (PROT_SUN * D2S)


class Surface(object):
    def __init__(
        self,
        nbins=5,
        delta_lnA=0.5,
        max_area=100,
        tau1=5,
        tau2=15,
        nlon=36,
        nlat=16,
    ):     
        #self.areas = max_area / np.exp(delta_lnA * np.arange(nbins))
        self.nbins=nbins # number of area bins
        self.delta_lnA=delta_lnA  # delta ln(A)
        self.max_area=max_area  # orig. area of largest bipoles (deg^2)
        self.tau1 = tau1
        self.tau2 = tau2
        self.nlon = nlon
        self.nlat = nlat

        self.regions = None
        self.nspots = None
        self.lightcurve = None

    def emerge_regions(
        self,
        ndays=1000,
        activity_level=1,
        butterfly=True,
        cycle_period=11,
        cycle_overlap=2,
        max_lat=28,
        min_lat=7,
        prob_corr=0.001,
    ):
        """     
        Simulates the emergence and evolution of starspots. 
        Output is a Table of active regions.

        Parameters
        ----------
        butterfly (bool, optional, default=True): 
            Have spots decrease from maxlat to minlat (True)
            or be randomly located in latitude (False)

        activity_level (float, optional, default=1.0): 
            Number of magnetic bipoles, normalized such that 
            for the Sun, activityrate = 1.

        cycle_period (float, optional, default=1.0): 
            Interval (years) between cycle starts (Sun is 11)

        cycle_overlap (float, optional, default=1.0): 
            Overlap of cycles in years.

        max_lat (float, optional, default=40):
            Maximum latitude of spot emergence in degrees.

        min_lat (float, optional, default=5):
            Minimum latitutde of spot emergence in degrees.

        ndays (int, optional, default=1200):
            Number of days to emerge spots. 

        prob_corr (float, optional, default=0.001):
            The probability of correlated active region emergence
            (relative to uncorrelated emergence).


        Returns
        -------
        regions (astropy Table): 
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
        # set attributes
        self.duration = ndays
        self.activity_level = activity_level
        self.butterfly = butterfly
        self.cycle_period = cycle_period
        self.cycle_overlap = cycle_overlap
        self.max_lat = max_lat
        self.min_lat = min_lat
        self.prob_corr = prob_corr

        # factor from integration over bin size (I think)
        dcon = np.exp(0.5*self.delta_lnA)- np.exp(-0.5*self.delta_lnA)

        amplitude = 10*activity_level
        ncycle = 365 * cycle_period
        nclen = 365 * (cycle_period + cycle_overlap)

        fact = np.exp(self.delta_lnA*np.arange(self.nbins)) # array of area reduction factors
        ftot = fact.sum()                   # sum of reduction factors
        bsiz = np.sqrt(self.max_area/fact)  # array of bipole separations (deg)
        tau = np.zeros((self.nlon, self.nlat, 2), dtype=int) + self.tau2
        dlon = 360 / self.nlon

        if butterfly:                       # Really we want spots to emerge in a
            l1 = max(min_lat-7, 0)          # range around the average active lat,
            l2 = min(max_lat+7, 90)         # so we bump the boundaries a bit.
        else:                               
            l1, l2 = min_lat, max_lat
        dlat = (l2-l1)/self.nlat                 

        self.regions = Table(names=('nday', 'thpos', 'phpos','thneg','phneg', 'width', 'bmax', 'ang'),
            dtype=(int, float, float, float, float, float, float, float))

        for nday in np.arange(ndays, dtype=int):
            # Emergence rates for correlated regions
            # Note that correlated emergence only occurs for the largest regions,
            # i.e., for bsiz[0]
            tau += 1
            index = (self.tau1 <= tau) & (tau < self.tau2)
            rc0 = np.where(index, prob_corr/(self.tau2-self.tau1), 0)

            ncur = nday // ncycle # index of current active cycle
            for icycle in [0, 1]: # loop over current and previous cycle
                nc = ncur - icycle # index of current or previous cycle
                nstart = ncycle*nc # start day of cycle
                phase = (nday - nstart) / nclen # phase relative to cycle start day
                if not (0 <= phase <= 1): # phase outside of [0, 1] is nonphysical
                    continue

                # Determine active latitude bins
                if butterfly:
                    latavg, latrms = exponential_latitudes(min_lat, max_lat, phase)
                else:
                    latavg, latrms = random_latitudes(min_lat, max_lat)

                # Compute emergence probabilities
                
                # Emergence rate of largest uncorrelated regions (number per day,
                # both hemispheres), from Shrijver and Harvey (1994)
                ru0_tot = amplitude*np.sin(np.pi*phase)**2 * dcon/self.max_area
                # Uncorrelated emergence rate per lat/lon bin, as function of lat
                jlat = np.arange(self.nlat, dtype=int)
                p = np.exp(-((l1 + dlat*(0.5+jlat) - latavg)/latrms)**2)
                ru0 = ru0_tot*p/(p.sum()*self.nlon*2)

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

                            self.add_region(nday, nc, lon, lat, k, bsize)

                            if nb == 0:
                                tau[i, j, k] = 0
        return self.regions

    def add_region(self, nday, nc, lon, lat, k, bsize):
        """
        Add one active region of a particular size at a particular location.

        Joy's law tilt angle is computed here as well. 
        For tilt angles, see 
            Wang and Sheeley, Sol. Phys. 124, 81 (1989)
            Wang and Sheeley, ApJ. 375, 761 (1991)
            Howard, Sol. Phys. 137, 205 (1992)

        Parameters
        ----------
        nday (int): day index
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
        self.assert_regions()

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

        self.regions.add_row([nday, thpos, phpos, thneg, phneg, width, bmax, ang])

    def evolve_spots(
        self,
        time,
        incl=90, 
        period=PROT_SUN,
        shear=0.3, 
        diffrot_func=sin2,
        spot_evol=gaussian_spots,
        tau_evol=5.0,
        alpha_med=0.0001,       
        threshold=0.1
    ):
        """
        Generate initial parameter set for spots and compute light curve. 
        Emergence times and initial locations are read from `self.regions`. 

        Includes rotation and foreshortening (i.e., spots in the center cause
        more modulation than spots at the limb, and spots out of view do not
        contribute flux modulation). Also includes spot emergence and decay.

        Currently there is no spot drift or shear (within an active region).

        Parameters
        ----------
        time (float array):
            The array of time values at which to compute the light curve.

        incl (float, optional, default=90):
            Inclination angle of the star in degrees, where inclination is
            the angle between the pole and the line of sight.

        period (float, optional, default=PROT_SUN):
            Rotation period of the star in days.

        shear (float, optional, default=0.3):
            Differential rotation rate of the star in units of equatorial
            rotation velocity. I.e., `shear` is alpha = delta_omega / omega.

        diffrot_func (function, optional, default=`utils.diffrot.sin2`):
            Differential rotation function. Default is sin^2 (latitude).

        spot_evol (function, optional, default=`utils.spotevol.gaussian_spots`):
            Spot evolution function. Default is double-sided gaussian with time.

        tau_evol (float, optional, default=5.0):
            Spot decay timescale in units of the equatorial rotation period.

        alpha_med (float, optional, default=0.0001):
            Spot filling factor, equal to spot area * contrast.
            E.g., a 50%-contrast spot subtending 1% of the stellar disk
            has alpha_med = 0.01 * 0.5 = 0.005.

        threshold (float, optional, default=0.1):
            Minimum peak magnetic flux for a spot to be considered.

        Returns
        -------
        lc (numpy array):
            the light curve: time-varying flux modulation from all spots.        
        """
        self.assert_regions()

        # set global stellar parameters which are the same for all spots
        # inclination
        self.incl = incl * np.pi/180 # in radians
        # rotation and differential rotation
        self.period = period # in days
        self.omega = 2*np.pi/(self.period * D2S) # in radians/s
        self.shear = shear # in radians/s
        self.diffrot_func = diffrot_func
        # spot emergence and decay
        self.spot_evol = spot_evol
        self.tau_emerge = min(2, self.period * tau_evol / 10)
        self.tau_decay = self.period * tau_evol
        # Convert spot properties
        tmax = self.regions['nday']
        lat = 0.5*(self.regions['thpos'] + self.regions['thneg'])
        lat = np.pi/2 - lat
        l = lat < 0
        lat[l] *= -1
        lon = 0.5*(self.regions['phpos'] + self.regions['phneg'])
        Bem = self.regions['bmax']

        # keep only spots with peak B-field > threshold
        l = Bem > threshold
        self.nspots = l.sum()
        self.tmax = tmax[l]
        self.lat = lat[l]
        self.lon = lon[l]
        self.amax = Bem[l] * alpha_med / np.median(Bem[l]) 
        # scale amax to achieve desired median alpha, 
        # where alpha = spot contrast * spot area 

        self.lightcurve = self.compute_lightcurve(time)
        return self.lightcurve       
        
    def compute_lightcurve(self, time):
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
        self.assert_regions()
        self.assert_spots()

        lc = np.ones_like(time, dtype="float32")
        for i in np.arange(self.nspots):
            lc += self.calci(time, i)
        return lc

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
        area = self.amax[i] * self.spot_evol(tt, self.tau_emerge, self.tau_decay)
        # Rotation rate
        omega_lat = self.diffrot_func(self.omega, self.shear, self.lat[i])
        # Foreshortening
        phase = omega_lat * time * D2S + self.lon[i]
        beta = np.cos(self.incl) * np.sin(self.lat[i]) + \
            np.sin(self.incl) * np.cos(self.lat[i]) * np.cos(phase)
        # Differential effect on stellar flux
        dF_i = - area * beta
        dF_i[beta < 0] = 0
        return dF_i

    def assert_regions(self):
        """ If `regions` hasn't been run, raise an error
        """
        assert self.regions is not None, "Set `regions` first with `Surface.emerge_regions`."

    def assert_spots(self):
        """Assert that `evolve_spots` has been run by checking the value of nspots
        """
        assert self.nspots is not None, "Run `evolve_spots` first to initialize spot parameters."

    def plot_butterfly(self):
        """Plot the stellar butterfly pattern.
        """
        self.assert_regions()

        lat = 0.5*(self.regions['thpos'] + self.regions['thneg'])
        lat = (np.pi/2 - lat)*u.rad.to(u.deg)

        fig, ax = plt.subplots()
        ax.scatter(self.regions['nday'], lat,
            s=self.regions['bmax']/np.median(self.regions['bmax'])*10,
            alpha=0.5, c='#996699', lw=0.5)
        ax.set_xlim(0, self.regions['nday'].max())
        ax.set_ylim(-90, 90)
        ax.set_yticks((-90, -45, 0, 45, 90))
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Latitude (deg)')
        fig.tight_layout()

        return fig, ax
    
    def pickle(self, filename):
        """
        Write Surface object to pickle file.

        Parameters
        ----------
        filename (str): output file path.

        Returns None.
        """
        pickle(self, filename)