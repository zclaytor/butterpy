import warnings

import numpy as np
import matplotlib.pylab as plt
import astropy.units as u
from astropy.table import Table
from astropy.io import fits

from .utils.activelat import random_latitudes, exponential_latitudes
from .utils.spotevol import gaussian_spots
from .utils.diffrot import sin2
from .utils.joyslaw import tilt
from .utils.animation import animate_spots
from .io.pkl import to_pickle, read_pickle
from .io.fits import to_fits


D2S = 1*u.day.to(u.s)

PROT_SUN = 24.5


class Surface(object):
    """Create a blank surface to emerge active regions and evolve star spots.

    The `Surface` consists of a grid of `nlat` latitude bins by `nlon`
    longitude bins, over which emergence probabilities are to be computed.

    Active region areas are drawn from a log-uniform distribution consisting
    of `nbins` values, starting at area `max_area` and spacing `delta_lnA`.

    Active regions can be "uncorrelated" or "correlated" to other regions.
    "Correlated" regions can only emerge near regions with the largest area
    (`max_area`) between `tau1` and `tau2` days after the preexisting region's
    emergence.

    Attributes:
        nbins (int): the number of discrete active region areas.
        delta_lnA (float): logarithmic spacing of area values.
        max_area (float): maximum active region area in square degrees.
        tau1 (int): first allowable day of "correlated" emergence, after
            preexisting region's emergence.
        tau2 (int): last allowable day of "correlated" emergence, after
            preexisting region's emergence.
        nlon (int): number of longitude bins in the Surface grid.
        nlat (int): number of latitude bins in the Surface grid.

        duration (int): number of days to emerge regions.
        activity_level (float): Number of magnetic bipoles, normalized such 
            that for the Sun, activity_level = 1.
        butterfly (bool): Have spots decrease from maxlat to minlat (True) or 
            be randomly located in latitude (False).
        cycle_period (float): Interval (years) between cycle starts (Sun is 11).
        cycle_overlap (float): Overlap of cycles in years.
        max_lat (float): Maximum latitude of spot emergence in degrees.
        min_lat (float): Minimum latitutde of spot emergence in degrees. 
        prob_corr (float): The probability of correlated active region 
            emergence (relative to uncorrelated emergence).      
        
        regions (astropy Table): list of active regions with timestamp,
            asterographic coordinates of positive and negative bipoles,
            magnetic field strength, and bipole tilt relative to equator.

        inclination (float): Inclination angle of the star in radians, where 
            inclination is the angle between the pole and the line of sight.
        period (float): Equatorial rotation period of the star in days.
        omega (float): Equatorial angular velocity in rad/s, equal to 2*pi/period.
        shear (float): Differential rotation rate of the star in units of 
            equatorial rotation velocity. I.e., `shear` is alpha = delta_omega / omega.
        diffrot_func (function): Differential rotation function. 
            Default is sin^2 (latitude).
        spot_func (function): Spot evolution function. 
            Default is double-sided gaussian with time.
        tau_emerge (float): Spot emergence timescale in days.
        tau_decay (float): Spot decay timescale in days.
        nspots (int): total number of star spots.
        tmax (numpy array): time of maximum area for each spot.
        lat (numpy array): latitude of each spot.
        lon (numpy array): longitude of each spot.
        amax (numpy array): maximum area of each spot in millionths of 
            solar hemisphere.

        lightcurve (LightCurve): time and flux modulation for all spots.

        wavelet_power (numpy array): the wavelet transform of the lightcurve.

    """

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
        """
        Note:
            You usually don't need to change the defaults for `Surface`.

        Args:
            nbins (int): the number of discrete active region areas.
            delta_lnA (float): logarithmic spacing of area values.
            max_area (float): maximum active region area in square degrees.
            tau1 (int): first allowable day of "correlated" emergence, after
                preexisting region's emergence.
            tau2 (int): last allowable day of "correlated" emergence, after
                preexisting region's emergence.
            nlon (int): number of longitude bins in the Surface grid.
            nlat (int): number of latitude bins in the Surface grid.
            
        """
        #self.areas = max_area / np.exp(delta_lnA * np.arange(nbins))
        self.nbins = nbins # number of area bins
        self.delta_lnA = delta_lnA  # delta ln(A)
        self.max_area = max_area  # orig. area of largest bipoles (deg^2)
        self.tau1 = tau1
        self.tau2 = tau2
        self.nlon = nlon
        self.nlat = nlat

        self.regions = None
        self.nspots = None
        self.lightcurve = None
        self.wavelet_power = None

    def __repr__(self):
        """Representation method for Surface.
        """
        repr = f"butterpy Surface from {type(self)} with:"

        repr += f"\n    {self.nlat} latitude bins by {self.nlon} longitude bins"

        if self.regions is not None:
            repr += f"\n    N regions = {len(self.regions)}"

        if self.lightcurve is not None:
            repr += f"\n    lightcurve length = {len(self.time)}, duration = {self.time.max() - self.time.min()}"

        if self.wavelet_power is not None:
            repr += f"\n    wavelet_power shape = {self.wavelet_power.shape}"
        
        return repr

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

        Args:
            ndays (int, optional, default=1000): Number of days to emerge spots.
            activity_level (float, optional, default=1): Number of magnetic 
                bipoles, normalized such that for the Sun, activity_level = 1.
            butterfly (bool, optional, default=True): Have spots decrease 
                from maxlat to minlat (True) or be randomly located in 
                latitude (False).
            cycle_period (float, optional, default=11): Interval (years) 
                between cycle starts (Sun is 11).
            cycle_overlap (float, optional, default=2): Overlap of cycles in 
                years.
            max_lat (float, optional, default=28): Maximum latitude of spot 
                emergence in degrees.
            min_lat (float, optional, default=7): Minimum latitutde of spot 
                emergence in degrees. 
            prob_corr (float, optional, default=0.001): The probability of 
                correlated active region emergence (relative to uncorrelated 
                emergence).

        Returns:
            regions: astropy Table where each row is an active region with 
                the following parameters:

                nday  = day of emergence
                thpos = theta of positive pole (radians)
                phpos = phi   of positive pole (radians)
                thneg = theta of negative pole (radians)
                phneg = phi   of negative pole (radians)
                width = width of each pole (radians)
                bmax  = maximum flux density (Gauss)

        Notes:
            Based on Section 4 of van Ballegooijen 1998, ApJ 501: 866
            and Schrijver and Harvey 1994, SoPh 150: 1S.

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

            We use a lower value of a(t) to account for "correlated" regions.

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

                            self._add_region_cycle(nday, nc, lat, lon, k, bsize)

                            if nb == 0:
                                tau[i, j, k] = 0
        return self.regions

    def _add_region_cycle(self, nday, nc, lat, lon, k, bsize):
        """
        Add one active region of a particular size at a particular location,
        caring about the cycle (for tilt angles).

        Joy's law tilt angle is computed here as well. 
        For tilt angles, see 
            Wang and Sheeley, Sol. Phys. 124, 81 (1989)
            Wang and Sheeley, ApJ. 375, 761 (1991)
            Howard, Sol. Phys. 137, 205 (1992)

        Args:
            nday (int): day index
            nc (int): cycle index
            lat (float): latitude
            lon (float): longitude
            k (int): hemisphere index (0 for North, 1 for South)
            bsize (float): the size of the bipole

        Adds a row with the following values to `self.regions`:
        
            nday (int): day index
            thpos (float): theta of positive bipole
            phpos (float): longitude of positive bipole
            thneg (float): theta of negative bipole
            phneg (float): longitude of negative bipole
            width (float): bipole width threshold, always 4...?
            bmax (float): magnetic field strength of bipole
            ang (float): Joy's law bipole angle (from equator)

        Returns None.
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

    def add_region(self, nday, lat, lon, bmax):
        """
        Add one active region of a particular size at a particular location,
        ignoring Joy's law tilt and cycle.

        This is meant to be a user-facing function.

        Args:
            nday (int): day index
            lat (float): latitude
            lon (float): longitude
            bmax (float): magnetic field strength of bipole

        Adds a row with the following values to `self.regions`:
        
            nday (int): day index
            thpos (float): theta of positive bipole
            phpos (float): longitude of positive bipole
            thneg (float): theta of negative bipole
            phneg (float): longitude of negative bipole
            bmax (float): magnetic field strength of bipole

        Returns None.
        """
        if self.regions is None:
            self.regions = Table(names=('nday', 'thpos', 'phpos','thneg','phneg', 'bmax'),
                dtype=(int, float, float, float, float, float))
            
        # Convert angles to radians
        lat *= np.pi/180
        phcen = lon*np.pi/180.
        bsize = np.sqrt(bmax/2.5) * np.pi/180

        # Compute bipole positions
        dph = 0
        dth = 0.5*bsize
        thcen = 0.5*np.pi - lat # k determines hemisphere
        phpos = phcen + dph
        phneg = phcen - dph
        thpos = thcen + dth
        thneg = thcen - dth

        self.regions.add_row([nday, thpos, phpos, thneg, phneg, bmax])

    def evolve_spots(
        self,
        time=None,
        inclination=90, 
        period=PROT_SUN,
        shear=0.3, 
        diffrot_func=sin2,
        spot_func=gaussian_spots,
        tau_evol=5.0,
        alpha_med=0.0001,       
        threshold=0.1,
    ):
        """
        Generate initial parameter set for spots and compute light curve. 
        Emergence times and initial locations are read from `self.regions`. 

        Includes rotation and foreshortening (i.e., spots in the center cause
        more modulation than spots at the limb, and spots out of view do not
        contribute flux modulation). Also includes spot emergence and decay.

        Currently there is no spot drift or shear (within an active region).

        Args:
            time (float array, default=None):
                The array of time values at which to compute the light curve.
                If no time is supplied, defaults to 0.1-day cadence and duration
                of `self.duration`: `np.arange(0, self.duration, 0.1)`.
            inclination (float, optional, default=90):
                Inclination angle of the star in degrees, where inclination is
                the angle between the pole and the line of sight.
            period (float, optional, default=PROT_SUN):
                Rotation period of the star in days.
            shear (float, optional, default=0.3):
                Differential rotation rate of the star in units of equatorial
                rotation velocity. I.e., `shear` is alpha = delta_omega / omega.
            diffrot_func (function, optional, default=`utils.diffrot.sin2`):
                Differential rotation function. Default is sin^2 (latitude).
            spot_func (function, optional, default=`utils.spotevol.gaussian_spots`):
                Spot evolution function. Default is double-sided gaussian with time.
            tau_evol (float, optional, default=5.0):
                Spot decay timescale in units of the equatorial rotation period.
            alpha_med (float, optional, default=0.0001):
                Spot filling factor, equal to spot area * contrast.
                E.g., a 50%-contrast spot subtending 1% of the stellar disk
                has alpha_med = 0.01 * 0.5 = 0.005.
            threshold (float, optional, default=0.1):
                Minimum peak magnetic flux for a spot to be considered.
            
        Returns:
            lc (LightCurve): observation times and flux modulation for all spots.        
        """
        self.assert_regions()

        # set global stellar parameters which are the same for all spots
        # inclination
        self.inclination = inclination * np.pi/180 # in radians
        # rotation and differential rotation
        self.period = period # in days
        self.omega = 2*np.pi/(self.period * D2S) # in radians/s
        self.shear = shear # in radians/s
        self.diffrot_func = diffrot_func
        # spot emergence and decay
        self.spot_func = spot_func
        self.tau_emerge = min(2, self.period * tau_evol / 10)
        self.tau_decay = self.period * tau_evol
        # Convert spot properties
        tmax = self.regions['nday']
        lat = 0.5*(self.regions['thpos'] + self.regions['thneg'])
        lat = np.pi/2 - lat

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

        Args:
            time (numpy array):
                The array of time values at which to compute the flux modulation.
                If `None` is passed, defaults to 0.1-day cadence and duration
                of `self.duration`: `np.arange(0, self.duration, 0.1)`.

        Returns:
            lc (LightCurve): observation times and flux modulation for all spots.        
        """
        self.assert_regions()
        self.assert_spots()

        if time is None:
            time = np.arange(0, self.duration, 0.1, dtype="float32")
        else:
            # ensure time values are floats
            time = time.astype("float32")
            
            if time.max() > self.duration:
                # warn if time array exceeds spot emergence duration
                warnings.warn("`time` array exceeds duration of regions computation.\n"
                              "No new spots will emerge after this time, and the "
                              "light curve will relax back to unity.", 
                              UserWarning)

        flux = np.ones_like(time, dtype="float32")
        for i in np.arange(self.nspots):
            flux += self.calc_i(time, i)

        lc = LightCurve(time, flux)
        return lc

    def calc_i(self, time, i):
        """
        Helper function to evolve one spot and calculate its impact on the flux.

        Includes rotation and foreshortening (i.e., spots in the center cause
        more modulation than spots at the limb, and spots out of view do not
        contribute flux modulation). Also includes spot emergence and decay.

        Currently there is no spot drift or shear (within an active region).

        Args:   
            time (numpy array): array of time values at which to compute the flux modulation.
            i (int): spot index, which can have integer values of [0, self.nspot].

        Returns:
            dF_i (numpy array): time-varying flux modulation from spot `i`.
        """
        tt = time - self.tmax[i]
        # Spot area
        area = self.amax[i] * self.spot_func(tt, self.tau_emerge, self.tau_decay)
        # Rotation rate
        omega_lat = self.diffrot_func(self.omega, self.shear, self.lat[i])
        phase = omega_lat * time * D2S + self.lon[i]
        # Foreshortening
        beta = np.cos(self.inclination) * np.sin(self.lat[i]) + \
            np.sin(self.inclination) * np.cos(self.lat[i]) * np.cos(phase)
        # Differential effect on stellar flux
        dF_i = -area*beta
        dF_i[beta < 0] = 0
        return dF_i
    
    def _calc_t(self, t, animate=False):
        """
        Helper function to calculate flux modulation for all spots at a single
        time step. This is much slower than `calc_i`, so use is only recommended
        for illustrative purposes such as in visualization tools that need to
        compute on time steps.

        Includes rotation and foreshortening (i.e., spots in the center cause
        more modulation than spots at the limb, and spots out of view do not
        contribute flux modulation). Also includes spot emergence and decay.

        Currently there is no spot drift or shear (within an active region).

        Args:
            t (float): the time value at which to compute the flux modulation.
            animate (bool, False): whether the function is being called for 
                animation purposes. If True, returns current latitude, longitude,
                area, and flux for each spot. If False, returns only the flux.

        Returns:
            dF_t (numpy array): single-epoch flux modulation from all spots.

        """
        tt = t - self.tmax
        # Spot area
        area = self.amax * self.spot_func(tt, self.tau_emerge, self.tau_decay)
        # Rotation rate
        omega_lat = self.diffrot_func(self.omega, self.shear, self.lat)
        phase = omega_lat * t * D2S + self.lon
        # Foreshortening
        beta = np.cos(self.inclination) * np.sin(self.lat) + \
            np.sin(self.inclination) * np.cos(self.lat)*np.cos(phase)
        # Differential effect on stellar flux
        dF_t = -area*beta
        dF_t[beta < 0] = 0

        if animate:
            return self.lat, phase, area, dF_t, 
        return dF_t

    def compute_wps(self, bin_size=None):
        """
        Computes the (optionally binned) Morlet wavelet power spectrum.
        Not yet implemented!
        """
        raise NotImplementedError("Method is not yet implemented.")
    
    @property
    def time(self):
        """Return the light curve time array.
        """
        self.assert_lightcurve()
        return self.lightcurve.time

    @property
    def flux(self):
        """Return the light curve flux array.
        """
        self.assert_lightcurve()
        return self.lightcurve.flux

    def plot_lightcurve(self, *args, **kw):
        """Wrapper for `self.lightcurve.plot`.
        """
        self.assert_lightcurve()
        return self.lightcurve.plot(*args, **kw)

    def assert_regions(self):
        """ If `regions` hasn't been run, raise an error
        """
        assert self.regions is not None, "Set `regions` first with `Surface.emerge_regions`."

    def assert_spots(self):
        """Assert that `evolve_spots` has been run by checking the value of `self.nspots`.
        """
        assert self.nspots is not None, "Run `evolve_spots` first to initialize spot parameters."

    def assert_lightcurve(self):
        """Assert that `compute_lightcurve` has been run by checking the value of `self.lightcurve`.
        """
        assert self.lightcurve is not None, "The light curve has not been set."

    def assert_wavelet(self):
        """Assert that `compute_wavelet_power` has been run.
        """
        assert self.wavelet_power is not None, "The wavelet power spectrum has not been computed."

    def plot_butterfly(self):
        """Plot the stellar butterfly pattern.
        """
        self.assert_regions()

        lat = 90*(1 - (self.regions["thpos"] + self.regions["thneg"])/np.pi)

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
    
    def animate_spots(self, *args, **kw):
        """
        Animate a plot of spot and light curve evolution. 
        Wrapper for `utils.visualization.animate_spots`.

        Args:
            time (numpy array):
                Array of time values to animate. Ideally you should only animate
                a subset of time steps, otherwise the animation will be too large.
            projection (str, "ortho"):
                Projection for animation. Can be orthographic ("ortho" or 
                "orthographic"), which requires cartopy, or cartesian ("cart" or
                "cartesian").
            window_size (float, 50):
                Window size, in days, for light curve viewing.
            fig_kw: dict of kwargs to be passed to `matplotlib.figure`.
            kw: remaining kwargs to be passed to `matplotlib.animation.FuncAnimation`.

        Returns:
            ani (matplotlib.animation.FuncAnimation):
                animated plot of spot and light curve evolution.
        """
        return animate_spots(self, *args, **kw)
    
    def to_pickle(self, filename):
        """
        Write Surface object to pickle file.

        Args:
            filename (str): output file path.

        Returns None.
        """
        to_pickle(self, filename)

    def to_fits(self, filename, filter="NONE", **kw):
        """
        Write Surface object to fits file.

        Args:
            filename (str): output file path.
            filter (str): filter for photometry. Default is "NONE".
            **kw: keyword arguments to be passed to HDUList.writeto
            
        Returns None.
        """
        to_fits(self, filename, filter=filter, **kw)


def read_fits(filename):
    """
    Reads a butterpy Surface from fits file.

    Args:
        filename (str): the path to the fits file.

    Returns:
        s (Surface): the read-in Surface.

    """
    with fits.open(filename) as hdul:   
        s = Surface()

        s.period = hdul[0].header["PERIOD"]
        s.activity_level = hdul[0].header["ACTIVITY"]
        s.cycle_period = hdul[0].header["CYCLE"]
        s.cycle_overlap = hdul[0].header["OVERLAP"]
        s.inclination = hdul[0].header["INCL"]
        s.min_lat = hdul[0].header["MINLAT"]
        s.max_lat = hdul[0].header["MAXLAT"]
        s.shear = hdul[0].header["DIFFROT"]
        s.tau_decay = hdul[0].header["TSPOT"]
        s.butterfly = hdul[0].header["BFLY"]

        s.regions = Table(hdul[1].data)

        s.lightcurve = LightCurve(hdul[2].data["time"], hdul[2].data["flux"])

    return s


class LightCurve(object):
    """
    Most basic light curve class with time and flux attributes.
    For more mature features, use Lightkurve.
    """
    def __init__(self, time, flux):
        """
        Initialize the light curve.

        Args:
            time (numpy array): array of time values corresponding to flux measurements.
            flux (numpy array): array of flux measurements.
        """
        self.time = time
        self.flux = flux

    def __repr__(self):
        """Representation method for LightCurve.
        """
        return repr(Table(data=[self.time, self.flux], names=["time", "flux"]))

    def plot(self, time_unit=None, flux_unit=None, **kw):
        """
        Plot flux versus time.

        Args:
            time_unit (str, None): time unit for plot label.
            flux_unit (str, None): flux unit for plot label.
            **kw: kwargs for `Axes.plot`.

        Returns:
            fig: Matplotlib figure object.
            ax: Matplotlib axes object.

        """
        fig, ax = plt.subplots()

        ax.plot(self.time, self.flux, **kw)
        
        xlabel = "Time"
        if time_unit is not None:
            xlabel += f" ({time_unit})"

        ylabel = "Flux"
        if flux_unit is not None:
            ylabel += f" ({flux_unit})"

        ax.set(xlabel=xlabel, ylabel=ylabel)

        return fig, ax