import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .distributions import Uniform, LogUniform, SineSquared, Boolean, Composite, Fixed
from .core import Surface

class Flutter(object):
    """Sets the environment for a group of Butterpy Surface simulations.
    """
    def __init__(
        self,
        n_sims=10,
        duration=365,
        cadence=1/48,
        activity_level=LogUniform(0.1, 10),
        butterfly=Boolean(p=0.8),
        cycle_period=LogUniform(1, 40),
        cycle_overlap=None,
        min_lat=Uniform(0, 40),
        max_lat=None,
        inclination=SineSquared(0, np.pi/2),
        period=Uniform(0.1, 180),
        shear=Composite(
            [LogUniform(-1, -0.1), LogUniform(0.1, 1), Fixed(0)],
            weights=[0.25, 0.5, 0.25]),
        tau_evol=LogUniform(1, 10)):
        """DOCSTRING
        """
        self.duration = duration
        self.cadence = cadence
        self.n_sims = n_sims
        
        self.activity_level = activity_level
        self.butterfly = butterfly
        self.cycle_period = cycle_period
        self.cycle_overlap = cycle_overlap
        self.min_lat = min_lat
        self.max_lat = max_lat

        self.inclination = inclination
        self.period = period
        self.shear = shear
        self.tau_evol = tau_evol

        self.sample(self.n_sims)

    def sample(self, n_sims=None):
        if n_sims is None:
            n_sims = self.n_sims

        self.DataFrame = pd.DataFrame([])

        self.DataFrame["activity_level"] = self.activity_level.sample(n_sims)
        cycle_period = self.cycle_period.sample(n_sims) # may be used by cycle_overlap
        self.DataFrame["cycle_period"] = cycle_period
        if self.cycle_overlap is None:
            # By default, cycle overlap should never be longer than cycle period
            # On second thought, the way cycle period is defined should make that fine.
            # Cycle period is the interval between cycle starts. The cycle can start every
            # year, and last for 4 years, giving 3 years overlap. I'll fix that later.
            self.cycle_overlap = LogUniform(0.1, cycle_period)
        self.DataFrame["cycle_overlap"] = self.cycle_overlap.sample(n_sims)
        self.DataFrame["inclination"] = self.inclination.sample(n_sims) * 180/np.pi
        min_lat = self.min_lat.sample(n_sims) # may be used by max_lat
        self.DataFrame["min_lat"] = min_lat
        if self.max_lat is None:
            # If max_lat is too close to min_lat, no spots get emerged.
            self.max_lat = Uniform(min_lat+5, 85)
        self.DataFrame["max_lat"] = self.max_lat.sample(n_sims)
        self.DataFrame["period"] = self.period.sample(n_sims)
        self.DataFrame["shear"] = self.shear.sample(n_sims)
        self.DataFrame["tau_evol"] = self.tau_evol.sample(n_sims)
        self.DataFrame["butterfly"] = self.butterfly.sample(n_sims)

        self.DataFrame.index.name = "simulation_number"
        return self.DataFrame

    def _assert_dataframe(self):
        assert self.DataFrame is not None, "Flutter.DataFrame is not set."

    def plot_histograms(self, nbins=20):
        """DOCSTRING
        """
        self._assert_dataframe()

        plt.figure(figsize=(12, 7))
        plt.subplot2grid((2, 3), (0, 0))
        plt.hist("period", nbins, color="C0", data=self.DataFrame, 
                 range=(self.period.min, self.period.max))
        plt.xlabel("Rotation Period (days)")
        plt.ylabel("N")
        plt.subplot2grid((2, 3), (0, 1))
        plt.hist("tau_evol", nbins, color="C1", data=self.DataFrame,
                 range=(self.tau_evol.min, self.tau_evol.max))
        plt.xlabel("Spot lifetime (Prot)")
        plt.ylabel("N")
        plt.subplot2grid((2, 3), (0, 2))
        plt.hist("inclination", nbins, color="C3", data=self.DataFrame,
                 range=(self.inclination.min*180/np.pi, self.inclination.max*180/np.pi))
        plt.xlabel("Stellar inclincation (deg)")
        plt.ylabel("N")
        plt.subplot2grid((2, 3), (1, 0))
        plt.hist("activity_level", nbins, color="C4", data=self.DataFrame,
                 range=(self.activity_level.min, self.activity_level.max))
        plt.xlabel("Stellar activity rate (Solar units)")
        plt.ylabel("N")
        plt.subplot2grid((2, 3), (1, 1))
        plt.hist("shear", nbins, color="C5", data=self.DataFrame,
                 range=(self.shear.min, self.shear.max))
        plt.xlabel(r"Differential Rotation Shear $\Delta \Omega / \Omega$")
        plt.ylabel("N")
        plt.subplot2grid((2, 3), (1, 2))
        plt.hist(self.DataFrame.eval("max_lat - min_lat"), nbins, color="C6")
        plt.xlabel("Spot latitude range (deg)")
        plt.ylabel("N")
        plt.tight_layout()
        ax = plt.gca()
        return ax

    def to_csv(self, path, **kw):
        """
        Outputs sampled distributions to CSV using `pandas.DataFrame.to_csv`.

        Args:
            path (str): The relative path, including filename, to save the data.
            **kw: keyword arguments for `pandas.DataFrame.to_csv`.

        Returns: 
            None or str: If path_or_buf is None, returns the resulting csv 
            format as a string. Otherwise returns None.
        """
        self._assert_dataframe()
        return self.DataFrame.to_csv(path, **kw)

    def __repr__(self):
        """Gonna need a repr to print out what distros are set.
        """

    def run(self):
        """Run the butterpy simulations with given input
        """
        self._assert_dataframe()
        for i, row in self.DataFrame.iterrows():
            self._run_one(row, i)

    def fly(self):
        """Alias for `Flutter.run` to fit with the butterfly theme.
        See docs for `Flutter.run`.
        """
        return self.run()
    
    def _run_one(self, row, i=None):
        s = Surface()

        r = s.emerge_regions(
            ndays=self.duration,
            activity_level=row["activity_level"],
            butterfly=row["butterfly"],
            cycle_period=row["cycle_period"],
            cycle_overlap=row["cycle_overlap"],
            max_lat=row["max_lat"],
            min_lat=row["min_lat"])

        time = np.arange(0, self.duration, self.cadence) + self.cadence

        l = s.evolve_spots(
            time=time,
            inclination=row["inclination"], 
            period=row["period"],
            shear=row["shear"], 
            tau_evol=row["tau_evol"])
        
        #s.plot_butterfly()
        #plt.savefig(f"butterfly{i}.png")

        #s.plot_lightcurve()
        #plt.savefig(f"lightcurve{i}.png")

        #s.to_fits(f"sim{i}.fits")

        return s