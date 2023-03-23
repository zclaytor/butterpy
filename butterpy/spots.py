# pylint: disable=no-member
"""
spots.py
Contains the Spots class, which holds parameters for spots on a given star.
"""
import os

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec, patches

from .constants import RAD2DEG, DAY2SEC, PROT_SUN, FLUX_SCALE


def diffrot(omega0, delta_omega, lat):
    """
    Default differential rotation function:
    Gives angular velocity as a function of latitude.
    """
    return omega0 - delta_omega * np.sin(lat) ** 2


class Spots(object):
    """
    Holds parameters for spots on a given star.

    PARAMETERS
    ----------
    spot_properties

    dur

    alpha_med

    incl

    omega

    delta_omega

    diffrot_func

    tau_evol

    threshold

    """

    def __init__(
        self,
        spot_properties,
        dur=None,
        alpha_med=FLUX_SCALE,
        incl=1,
        period=PROT_SUN,
        diffrot_shear=0.2,
        diffrot_func=diffrot,
        decay_timescale=5.0,
        threshold=0.1,
    ):

        ## global stellar parameters, same for all spots
        self.spot_properties = spot_properties
        self.inclination = incl

        # rotation
        self.omega = 2 * np.pi / (period * DAY2SEC)
        self.delta_omega = diffrot_shear * self.omega
        self.diffrot_func = diffrot_func

        # spot emergence and decay timescales
        self.emergence_timescale = max(
            2.0, period * decay_timescale / 5
        )
        self.decay_timescale = period * decay_timescale

        # convert active region properties
        time = spot_properties["nday"].values.astype(float)
        latitude = spot_properties["lat"].values / RAD2DEG
        longitude = spot_properties["lon"].values / RAD2DEG
        peak_magnetic_flux = spot_properties["bmax"].values

        # keep only spots emerging within specified timespan and with peak B-field > threshold
        if dur is None:
            self.dur = time.max()
        else:
            self.dur = dur

        keep = (time <= self.dur) & (peak_magnetic_flux > threshold)
        self.nspot = keep.sum()
        self.t0 = time[keep]
        self.latitude = latitude[keep]
        self.longitude = longitude[keep]
        self.area_max = (
            alpha_med * (peak_magnetic_flux / np.median(peak_magnetic_flux))[keep]
        )
        # alpha is spot contrast * spot area

    def calc_i(self, time, animate=False):
        """Modulate flux for single time step"""
        tt = time - self.t0
        t_emerge = self.emergence_timescale
        t_decay = self.decay_timescale
        timescale = np.where(tt < 0, t_emerge, t_decay)
        current_area = self.area_max * np.exp(-0.5 * (tt / timescale) ** 2)

        omega_lat = self.diffrot_func(self.omega, self.delta_omega, self.latitude)
        current_lon = omega_lat * time * DAY2SEC + self.longitude

        cos_beta = np.cos(self.inclination) * np.sin(self.latitude) + np.sin(
            self.inclination
        ) * np.cos(self.latitude) * np.cos(current_lon)

        dF = -current_area * np.maximum(cos_beta, 0)

        if animate:
            return self.latitude, current_lon, current_area, dF
        return dF.sum()

    def calc(self, time, max_memory=1000):
        """Modulate flux using chunks for all spots"""
        M = self.nspot
        N = len(time)
        memory_ceil = max_memory * (1024 ** 2) # bytes

        if 8 * M * N < memory_ceil:
            dFlux = self.matrix_calc(time)
        else:
            nbins = int(np.ceil(8 * M * N / memory_ceil))
            n = int(N / nbins)
            dFlux = np.zeros(N)
            for i in range(nbins):
                dFlux[i * n : (i + 1) * n] = self.matrix_calc(time[i * n : (i + 1) * n])
            if nbins * n < N:
                dFlux[nbins * n :] = self.matrix_calc(time[nbins * n :])
        return dFlux

    def matrix_calc(self, time):
        """Modulate flux for all spots"""
        M = self.nspot
        N = len(time)

        # spot area
        tt = np.outer(np.ones(M), time) - np.outer(self.t0, np.ones(N))
        area = np.outer(self.area_max, np.ones(N))
        emerge = np.outer(self.emergence_timescale, np.ones(N))
        decay = np.outer(self.decay_timescale, np.ones(N))
        timescale = np.where(tt < 0, emerge, decay)
        area *= np.exp(-0.5 * (tt / timescale) ** 2)

        # rotation rate
        omega = self.diffrot_func(self.omega, self.delta_omega, self.latitude)
        # here 'phase' is the longitude as a function of time
        phase = np.outer(omega, time * DAY2SEC) + np.outer(self.longitude, np.ones(N))

        # foreshortening
        # if inclination is angle between pole and line of sight,
        # cos(beta) = cos(i)*sin(lat) + sin(i)*cos(lat)*cos(lon(t)).
        cos_beta = np.cos(self.inclination) * np.outer(
            np.sin(self.latitude), np.ones(N)
        ) + np.sin(self.inclination) * np.outer(
            np.cos(self.latitude), np.ones(N)
        ) * np.cos(
            phase
        )

        # Total modulation on stellar flux
        dFlux = -area * np.maximum(cos_beta, 0)

        # return area, omega, beta, dFlux
        return dFlux.sum(axis=0)

    def plot_butterfly(self, fig=None, ax=None):
        """Plot the butterfly diagram"""
        latitude = self.spot_properties["lat"]

        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot()

        ax.scatter(
            self.spot_properties["nday"],
            latitude,
            s=10
            * self.spot_properties["bmax"]
            / np.median(self.spot_properties["bmax"]),
            alpha=0.5,
            c="#996699",
            lw=0.5,
        )
        ax.set_xlim(0, self.spot_properties["nday"].max())
        ax.set_ylim(-90, 90)
        ax.set_yticks((-90, -45, 0, 45, 90))
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Latitude (deg)")

    def ortho_animation(self, time, lightcurve, window_size=50, 
        fig_kw={"figsize": (5, 9)}, **kw):
        try:
            import cartopy.crs as ccrs
        except:
            raise ImportError(
                "butterpy requires the use of cartopy to run orthographic animations, "
                "but cartopy is not installed. Please install using "
                "`conda install -c conda-forge cartopy`, or if using pip, see "
                "https://scitools.org.uk/cartopy/docs/latest/installing.html for details."
            )

        fig = plt.figure(**fig_kw)
        fig.subplots_adjust(
            top=0.93, bottom=0.12, left=0.21, right=0.95, hspace=0.05)
        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=(1, 0.3))
        ax1 = fig.add_subplot(gs[0], 
            projection=ccrs.Orthographic(0, 90 - RAD2DEG * self.inclination))
        ax2 = fig.add_subplot(gs[1])

        ax1.set_global()
        ax1.gridlines(color='k', linestyle='dotted')
        ax1.scatter([], [], s=1, alpha=0.5, c="#996699", lw=0.5,
            transform=ccrs.PlateCarree())

        ax2.plot(lightcurve["time"], lightcurve["flux"])
        ax2.add_artist(
            patches.Ellipse(
                (0.5, 0.5),
                width=0.02,
                height=0.05,
                color="r",
                transform=ax2.transAxes,
                zorder=2,
            )
        )
        width = window_size / 2
        ax2.set_ylim(0.97, 1)
        ax2.set_xticks(np.arange(time[0] - width / 2, time[-1] + width / 2, width / 2))
        ax2.set_xlim(time[0] - width, time[0] + width)
        ax2.set_xlabel("Time (days)")
        ax2.set_ylabel("Relative Flux")
        ax2.vlines(0.5, color="r", ymin=0, ymax=1, transform=ax2.transAxes)
        title = fig.suptitle("", x=0.57)
        fig.align_labels()

        ax = (ax1, ax2)
        ani = animation.FuncAnimation(
            fig,
            _update_figure,
            frames=time,
            blit=False,
            repeat=False,
            fargs=(self, ax, title, 'ortho'),
            **kw,
        )
        return ani

    def animate_evolution(self, time, sim, lightcurve, window_size=50, **kw):
        nlat = 16
        nlon = 36
        dlon = 360 / nlon

        lat_width = 7
        lat_min = max(sim["Spot Min"] - lat_width, 0)
        lat_max = sim["Spot Max"] + lat_width

        fig = plt.figure(figsize=(5, 6), constrained_layout=False)
        fig.subplots_adjust(
            top=0.95, bottom=0.08, left=0.16, right=0.95, hspace=0.3, wspace=0.2
        )
        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=(1, 0.5))
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        ax1.set_xlim(-180, 180)
        ax1.set_xticks(np.linspace(-180, 180, 7))
        ymax = np.ceil(lat_max / 5) * 5
        ax1.set_ylim(-ymax, ymax)
        ax1.set_yticks(np.linspace(-ymax, ymax, 5))
        ax1.xaxis.set_ticklabels([180, 240, 300, 0, 60, 120, 180])

        ax1.vlines(
            np.linspace(dlon - 180, 180 - dlon, nlon - 1),
            ymin=lat_min,
            ymax=lat_max,
            lw=0.2,
        )
        ax1.vlines(
            np.linspace(dlon - 180, 180 - dlon, nlon - 1),
            ymin=-lat_max,
            ymax=-lat_min,
            lw=0.2,
        )
        ax1.hlines(np.linspace(lat_min, lat_max, nlat + 1), xmin=-180, xmax=180, lw=0.2)
        ax1.hlines(
            -np.linspace(lat_min, lat_max, nlat + 1), xmin=-180, xmax=180, lw=0.2
        )

        ax1.set_xlabel("Longitude (degrees)")
        ax1.set_ylabel("Latitude (degrees)")
        ax1.scatter([], [], s=1, alpha=0.5, c="#996699", lw=0.5)

        ax2.plot(lightcurve["time"], lightcurve["flux"])
        ax2.add_artist(
            patches.Ellipse(
                (0.5, 0.5),
                width=0.02,
                height=0.05,
                color="r",
                transform=ax2.transAxes,
                zorder=2,
            )
        )
        width = window_size / 2
        ax2.set_ylim(0.97, 1)
        ax2.set_xticks(np.arange(time[0] - width / 2, time[-1] + width / 2, width / 2))
        ax2.set_xlim(time[0] - width, time[0] + width)
        ax2.set_xlabel("Time (days)")
        ax2.set_ylabel("Relative Flux")
        ax2.vlines(0.5, color="r", ymin=0, ymax=1, transform=ax2.transAxes)
        title = fig.suptitle("")
        fig.align_labels()

        ax = (ax1, ax2)
        ani = animation.FuncAnimation(
            fig,
            _update_figure,
            frames=time,
            blit=False,
            repeat=False,
            fargs=(self, ax, title),
            **kw,
        )
        return ani


def _update_figure(time, spots, ax, title, projection=None):
    ax1, ax2 = ax

    latitudes, longitudes, areas, fluxes = spots.calc_i(time, animate=True)
    longitudes = (longitudes * RAD2DEG) % 360
    longitudes[longitudes >= 180] -= 360

    if projection == 'ortho':
        coll_idx = 0
        sizes = -10 * fluxes / FLUX_SCALE
    else:
        coll_idx = -1
        sizes = 10 * areas / FLUX_SCALE

    im1 = ax1.collections[coll_idx]
    im1.set_offsets(np.c_[longitudes, latitudes * RAD2DEG])
    im1.set_sizes(sizes)
    new_flux = 1 + fluxes.sum()

    y1, y2 = ax2.get_ylim()
    if new_flux < y1 + 0.01:
        y1 = new_flux - 0.01
        ax2.set_ylim(y1, y2)
    circle_y = (new_flux - y1) / (y2 - y1)
    circle, = ax2.patches
    circle.set_center((0.5, circle_y))

    t1, t2 = ax2.get_xlim()
    t0 = (t1 + t2) / 2
    delta_t = time - t0
    ax2.set_xlim(t1 + delta_t, t2 + delta_t)

    time_minutes = time * 24 * 60
    hours, minutes = divmod(time_minutes, 60)
    days, hours = divmod(hours, 24)
    title.set_text(f"Elapsed: {days:4.0f}d{hours:02.0f}h{minutes:02.0f}m")
    return (im1,)


def get_animation(path, time, projection=None, verbose=False, **kw):
    root, fname = os.path.split(path)
    sim_number = int("".join([char for char in fname if char.isdigit()]))
    path_to_sims = os.path.join(root, 'simulation_properties.csv')

    my_sim = pd.read_csv(path_to_sims).iloc[sim_number]

    if verbose:
        for label, item in my_sim.iteritems():
            print(f'{label.ljust(20, ".")}{item}')

    with fits.open(path) as f:
        lightcurve = f[1].data
        spot_properties = f[2].data

    my_spots = Spots(
        pd.DataFrame(spot_properties),
        incl=my_sim["Inclination"],
        period=my_sim["Period"],
        diffrot_shear=my_sim["Shear"],
        alpha_med=np.sqrt(my_sim["Activity Rate"]) * FLUX_SCALE,
        decay_timescale=my_sim["Decay Time"],
    )

    if projection == 'ortho':
        ani = my_spots.ortho_animation(time, lightcurve, **kw)
    else:
        ani = my_spots.animate_evolution(time, my_sim, lightcurve, **kw)
    return ani
