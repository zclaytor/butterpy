import numpy as np
import pandas as pd


class Flutter(object):
    """Sets the environment for a group of Butterpy Surface simulations.
    """
    def __init__(self):
        """DOCSTRING
        """
        self.duration = None
        self.cadence = None
        self.n_sims = None
        
        self.activity_level = None
        self.butterfly = None
        self.cycle_period = None
        self.cycle_overlap = None
        self.max_lat = None
        self.min_lat = None

        self.incl = None 
        self.period = None
        self.shear = None
        self.tau_evol = None

        # inclination ~ uniform in sin^2(i)
        incl = np.arcsin(np.sqrt(np.random.uniform(0, 1, Nlc)))
        # activity rate ~ uniform
        # ar = 10**np.random.uniform(low=-2, high=1, size=Nlc)
        ar = np.random.uniform(low=0.1, high=10, size=Nlc)
        # cycle length ~ log uniform
        clen = 10 ** np.random.uniform(low=0, high=1.6, size=Nlc)
        # cycle overlap ~ log uniform, but max out at cycle length.
        max_cover = np.maximum(np.log10(clen), 0.5)
        # cover = 10**np.random.uniform(low=-1, high=0.5, size=Nlc)
        cover = 10 ** np.random.uniform(low=-1, high=max_cover)
        # minimum and maximum spot latitudes
        theta_low = np.random.uniform(low=0, high=40, size=Nlc)
        theta_high = np.random.uniform(low=theta_low + 5, high=80)
        # period ~ uniform
        # period = 10.0**np.random.uniform(low=-1, high=2, size=Nlc)
        period = np.random.uniform(low=0.1, high=365, size=Nlc)
        # spot decay timescale ~ log uniform
        tau_evol = 10.0 ** np.random.uniform(low=0, high=1, size=Nlc)
        # if butterfly==True, spot emergence latitude has cycle phase dependence
        butterfly = np.random.choice([True, False], size=Nlc, p=[0.8, 0.2])
        # differential rotation shear ~ log uniform, and allow for negative values
        diffrot_shear = np.zeros(Nlc)
        n_pos = int(Nlc * 0.5)
        n_neg = int(Nlc * 0.25)
        diffrot_shear[:n_pos] = 10 ** np.random.uniform(-1, 0, size=n_pos)
        diffrot_shear[n_pos : n_pos + n_neg] = -10 ** np.random.uniform(-1, 0, size=n_neg)
        np.random.shuffle(diffrot_shear)

        omega = 2 * np.pi / period # rad / day


        # Stitch this all together and write the simulation properties to file
        sims = {}
        sims["Activity Rate"] = ar
        sims["Cycle Length"] = clen
        sims["Cycle Overlap"] = cover
        sims["Inclination"] = incl
        sims["Spot Min"] = theta_low
        sims["Spot Max"] = theta_high
        sims["Period"] = period
        sims["Omega"] = omega
        sims["Shear"] = diffrot_shear
        sims["Decay Time"] = tau_evol
        sims["Butterfly"] = butterfly
        sims = pd.DataFrame.from_dict(sims)
        sims.to_csv(sim_dir + "simulation_properties.csv", float_format="%5.4f",
            index_label="Simulation Number")


    def make_plots(self):
        """DOCSTRING
        """
        pass

    def to_csv(self):
        """DOCSTRING
        """
        pass

    def __repr__(self):
        """Gonna need a repr to print out what distros are set.
        """