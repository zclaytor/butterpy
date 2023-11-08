import numpy as np
import pandas as pd

from .distributions import Uniform, LogUniform, SineSquared, Boolean, Composite, Fixed

class Flutter(object):
    """Sets the environment for a group of Butterpy Surface simulations.
    """
    def __init__(
        self,
        duration=365,
        cadence=1/48,
        n_sims=10,
        activity_level=LogUniform(0.1, 10),
        butterfly=Boolean(p=0.8),
        cycle_period=LogUniform(1, 40),
        cycle_overlap=None,
        min_lat=Uniform(0, 40),
        max_lat=None,
        inclination=SineSquared(0, np.pi/2),
        period=Uniform(0.1, 180),
        shear=None,
        tau_evol=LogUniform(1, 10)):
        """DOCSTRING
        """
        self.duration = duration
        self.cadence = cadence
        self.n_sims = n_sims
        
        self.activity_level = activity_level
        self.butterfly = butterfly
        self.cycle_period = cycle_period
        if cycle_overlap is None:
            cycle_overlap = LogUniform(0.1, self.cycle_period)
        self.cycle_overlap = cycle_overlap
        self.min_lat = min_lat
        if max_lat is None:
            max_lat = Uniform(self.min_lat+5, 85)
        self.max_lat = max_lat

        self.inclination = inclination
        self.period = period
        if shear is None:
            shear = Composite(
                [LogUniform(-1, 0), LogUniform(0, 1), Fixed(0)],
                weights=[0.25, 0.5, 0.25])
        self.shear = shear
        self.tau_evol = tau_evol

        self.DataFrame = None

    def sample(self):
        self.DataFrame = pd.DataFrame([])
        
        self.DataFrame["activity_level"] = self.activity_level.sample(self.n_sims)
        self.DataFrame["cycle_period"] = self.cycle_period.sample(self.n_sims)
        self.DataFrame["cycle_overlap"] = self.cycle_overlap.sample(self.n_sims)
        self.DataFrame["inclination"] = self.inclination.sample(self.n_sims)
        self.DataFrame["min_lat"] = self.min_lat.sample(self.n_sims)
        self.DataFrame["max_lat"] = self.max_lat.sample(self.n_sims)
        self.DataFrame["period"] = self.period.sample(self.n_sims)
        self.DataFrame["shear"] = self.shear.sample(self.n_sims)
        self.DataFrame["tau_evol"] = self.tau_evol.sample(self.n_sims)
        self.DataFrame["butterfly"] = self.butterfly.sample(self.n_sims)

        self.DataFrame.index.name = "simulation_number"


    def make_plots(self):
        """DOCSTRING
        """
        pass

    def to_csv(self, path, **kw):
        """DOCSTRING
        """
        self.DataFrame.to_csv(path, **kw)        

    def __repr__(self):
        """Gonna need a repr to print out what distros are set.
        """