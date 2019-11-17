"""
spots.py
Contains the Spots class, which holds parameters for spots on a given star.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from .constants import RAD2DEG, DAY2SEC, OMEGA_SUN


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
        alpha_med=0.0001,
        incl=None,
        omega=2.0,
        delta_omega=0.3,
        diffrot_func=diffrot,
        decay_timescale=5.0,
        threshold=0.1,
    ):

        ## global stellar parameters, same for all spots

        self.spot_properties = spot_properties
        if incl is None:
            self.inclination = np.arcsin(np.random.uniform())
        else:
            self.inclination = incl

        # rotation
        self.omega = omega * OMEGA_SUN
        self.delta_omega = delta_omega * OMEGA_SUN
        self.equatorial_period = 2 * np.pi / self.omega / DAY2SEC
        self.diffrot_func = diffrot_func

        # spot emergence and decay timescales
        self.emergence_timescale = max(
            2.0, self.equatorial_period * decay_timescale / 5
        )
        self.decay_timescale = self.equatorial_period * decay_timescale

        # convert active region properties
        time = spot_properties["nday"].values
        latitude = spot_properties["lat"].values
        longitude = spot_properties["lon"].values
        peak_magnetic_flux = spot_properties["bmax"]

        # keep only spots emerging within specified timespan and with peak B-field > threshold
        if dur is None:
            self.dur = time.max()
        else:
            self.dur = dur

        keep = (time < self.dur) & (peak_magnetic_flux > threshold)
        self.nspot = keep.sum()
        self.t0 = time[keep]
        self.latitude = latitude[keep]
        self.longitude = longitude[keep]
        # self.area_max = active_region_tilt[keep] * alpha_med / np.median(active_region_tilt[keep]) # median-divide to scale.
        # Aigrain et al. (2015) has this value as alpha_med*Bmax/mean(Bmax)
        self.area_max = (
            alpha_med * (peak_magnetic_flux / np.median(peak_magnetic_flux))[keep]
        )
        # alpha is spot contrast * spot area

    def calc(self, time):
        """Modulate flux using chunks for all spots"""
        N = len(time)

        if N > 365 * 48:
            nbins = 100
            n = int(N / nbins)
            dFlux = np.zeros(N)
            for i in range(nbins):
                dFlux[i * n : (i + 1) * n] = self.matrix_calc(time[i * n : (i + 1) * n])
        else:
            dFlux = self.matrix_calc(time)

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


def butterfly(self):
    """Plot the butterfly diagram"""
    latitude = self.spot_properties["lat"]

    fig, ax = subplots()
    ax.scatter(
        self.spot_properties["nday"],
        latitude * RAD2DEG,
        s=10 * self.spot_properties["bmax"] / np.median(self.spot_properties["bmax"]),
        alpha=0.5,
        c="#996699",
        lw=0.5,
    )
    ax.set_xlim(0, self.spot_properties["nday"].max())
    ax.set_ylim(-90, 90)
    ax.set_yticks((-90, -45, 0, 45, 90))
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Latitude (deg)")
    fig.tight_layout()


def main():
    print('"main" functionality not implemented yet!')


if __name__ == "__main__":
    main()
