import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from butterpy.constants import RAD2DEG
from config import Nlc, sim_dir

def generate_simdata():
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

    plt.figure(figsize=(12, 7))
    plt.subplot2grid((2, 3), (0, 0))
    plt.hist(period, 20, color="C0")
    plt.xlabel("Rotation Period (days")
    plt.ylabel("N")
    plt.subplot2grid((2, 3), (0, 1))
    plt.hist(tau_evol, 20, color="C1")
    plt.xlabel("Spot lifetime (Prot)")
    plt.ylabel("N")
    plt.subplot2grid((2, 3), (0, 2))
    plt.hist(incl * RAD2DEG, 20, color="C3")
    plt.xlabel("Stellar inclincation (deg)")
    plt.ylabel("N")
    plt.subplot2grid((2, 3), (1, 0))
    plt.hist(ar, 20, color="C4")
    plt.xlabel("Stellar activity rate (x Solar)")
    plt.ylabel("N")
    plt.subplot2grid((2, 3), (1, 1))
    plt.hist(diffrot_shear, 20, color="C5")
    plt.xlabel(r"Differential Rotation Shear $\Delta \Omega / \Omega$")
    plt.ylabel("N")
    plt.subplot2grid((2, 3), (1, 2))
    plt.hist(theta_high - theta_low, 20, color="C6")
    plt.xlabel("Spot latitude range")
    plt.ylabel("N")
    plt.tight_layout()
    plt.savefig(f"{sim_dir:s}distributions.png", dpi=150)

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

    return sims


if __name__ == "__main__":
    sims = generate_simdata()
