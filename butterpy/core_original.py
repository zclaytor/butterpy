import numpy as np
import matplotlib.pylab as plt
import astropy.units as u
from astropy.table import Table

D2S = 1*u.day.to(u.s)

PROT_SUN = 27.0
OMEGA_SUN = 2 * np.pi / (27.0 * D2S)


def diffrot_sin2(omega_0, delta_omega, lat):
    return omega_0 - delta_omega * np.sin(lat)**2


class spots():
    """Holds parameters for spots on a given star"""
    def __init__(self,spot_properties,  dur = None, alpha_med = 0.0001, incl = None, \
                 omega = 2.0, delta_omega = 0.3, diffrot_func = diffrot_sin2, \
                 tau_evol = 5.0, threshold = 0.1):
        '''Generate initial parameter set for spots (emergence times
        and initial locations are read from regions file)'''
        # set global stellar parameters which are the same for all spots
        # inclination
        self.spot_properties = spot_properties
        if incl == None:
            self.incl = np.arcsin(np.random.uniform())
        else:
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
        l = (t0 < self.dur) * (Bem > threshold)
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


def regions(randspots=False, activityrate=1, cyclelength=1, \
    cycleoverlap=0, maxlat=40, minlat=5, ndays=1200):
    ''' Routine to produce a butterfly pattern and save it in regions.txt
        The inputs are:
        randspots=True / False - have spots decrease from maxlat to minlat or be randomly located in latitude
        activityrate = No. of spots x Solar rate
        cyclelength - length of cycle in years (Sun is 11)
        cycleovelap - overlap of cycles in years
        maxlat = maximum latitude of spot emergence (deg)
        minlat = minimum latitutde of emergence (deg)
        ndays = how many days to emerge spots for
        Based on Section 4 of van Ballegooijen 1998
        Written by Joe Llama (joe.llama@lowell.edu) V 11/1/16
        # Converted to Python 3 9/5/2017
    '''
    nbin=5 # number of area bins
    delt=0.5  # delta ln(A)
    amax=100.  # orig. area of largest bipoles (deg^2)
    dcon = np.exp(0.5*delt)- np.exp(-0.5*delt)
    deviation = (maxlat-minlat) / 7.
    atm = 10*activityrate
    ncycle = 365 * cyclelength
    nclen = 365 * (cyclelength + cycleoverlap)
    latrmsd = deviation
    fact = np.exp(delt*np.arange(nbin)) #array of area reduction factors
    ftot = np.sum(fact)             #sum of reduction factors
    bsiz = np.sqrt(amax/fact)         #array of bipole separations (deg)
    tau1 = 5                       #first and last times (in days) for
    tau2 = 15                      #  emergence of "correlated" regions
    prob = 0.0001                   #total probability for "correlation"
    nlon = 36                      #number of longitude bins
    nlat = 16                      #number of latitude bins
    tau = np.zeros((nlon,nlat,2), dtype=int)+tau2
    dlon = 360. / nlon
    dlat = maxlat/nlat
    ncnt = 0
    ncur = 0
    cycle_days = ncycle
    start_day  = 0
    spots = Table(names=('nday', 'thpos', 'phpos','thneg','phneg', 'width', 'bmax', 'ang'),
        dtype=(int, float, float, float, float, float, float, float))
    for nday in np.arange(ndays, dtype=int):
        if nday % cycle_days == 0:
            ncur += 1
            cycle_days += ncycle
        tau += 1
        rc0 = np.zeros((nlon, nlat, 2))
        index = (tau1 < tau) & (tau < tau2)
        if index.any():
            rc0[index] = prob / (tau2 - tau1)
        for icycle in [0, 1]:
            nc = ncur - icycle
            if ncur == 1:
                if icycle == 0:
                    start_day = ncycle*nc
                if icycle == 1:
                    start_day = 0
            else:
                start_day = ncycle*nc
            nstart = start_day
            ic = 1. - 2.*((nc + 2.) % 2) # This might be wrong
            phase = (nday - nstart) / nclen
            #print(nday, ncur, cycle_days, icycle, nc, ic, start_day, phase)
            #input()
            ru0_tot = atm*np.sin(np.pi*phase)**2.*(dcon)/amax
            if randspots == False:
                #This is a bit of a fudge. For the sun, y =35 - 48x + 20x^2
                latavg = maxlat - (maxlat+minlat)*phase + \
                        +2*minlat*phase**2.
                latrms = (maxlat/5.) - latrmsd*phase
                nlat1 = np.fix(np.max([(maxlat*0.9) - (1.2*maxlat)*phase, 0.])/dlat)
                nlat2 = np.fix(np.min([(maxlat + 6.) - maxlat*phase, maxlat])/dlat)
                nlat2 = np.min([nlat2, nlat-1])
            else:
                latavg = (maxlat - minlat) / 2.
                latrms = (maxlat - minlat)
                nlat1 = np.fix(minlat / dlat)
                nlat2 = np.fix(maxlat / dlat)
                nlat2 = np.min([nlat2, nlat-1])
            p = np.zeros(nlat)
            #print(phase, latavg, latrms, dlat, nlat1, nlat2)
            #exit()
            for j in np.arange(nlat1, nlat2, dtype=int):
                p[j] = np.exp(-((dlat*(0.5+j)-latavg)/latrms)**2.)
                ru0 = ru0_tot*p/(np.sum(p)*nlon*2)
            for k in [0, 1]:
                for j in np.arange(nlat1, nlat2, dtype=int):
                    r0 = ru0[j] + rc0[:, j, k]
                    rtot = np.sum(r0)
                    sumv = rtot * ftot
                    x = np.random.uniform()
                    if x < sumv:
                        nb = 0
                        sumb = rtot*fact[0]
                        while x > sumb:
                            nb += 1
                            sumb += rtot*fact[nb]
                        i = 0
                        sumb += (r0[0]-rtot)*fact[nb]
                        while x > sumb:
                            i += 1
                            sumb += r0[i]*fact[nb]
                        lon = dlon*(np.random.uniform() + i)
                        lat = dlat*(np.random.uniform() + j)

                        w_org = 0.4*bsiz[nb]
                        width = 4.0
                        bmax = 250.*(w_org / width)**2.
                        bsizr = np.pi * bsiz[nb] / 180.
                        width *= np.pi / 180.
                        while True:
                            x = np.random.normal()
                            if np.abs(x) < 1.6:
                                break
                        while True:
                            y = np.random.normal()
                            if np.abs(y) < 1.8:
                                break
                        z = np.random.uniform()
                        if z > 0.14:
                            ang = (0.5*lat + 2.0) + 27.*x*y
                        else:
                            while True:
                                z = np.random.normal()
                                if np.abs(z) < 0.5:
                                    break
                            ang = z*np.pi/180.
                        lat = np.pi * lat / 180.
                        ang *= np.pi/180.
                        dph = ic*0.5*bsizr*np.cos(ang)/np.cos(lat)
                        dth = ic*0.5*bsizr*np.sin(ang)
                        phcen = np.pi*lon/180.
                        if k == 0:
                            thcen = 0.5*np.pi - lat
                        else:
                            thcen = 0.5*np.pi + lat
                        phpos = phcen + dph
                        phneg = phcen - dph
                        thpos = thcen + dth
                        thneg = thcen - dth
                        spots.add_row([nday, thpos, phpos, thneg, phneg, width, bmax, ang])
                        ncnt += 1
                        if nb < 1:
                            tau[i, j, k] = 0
    return spots