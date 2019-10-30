import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from butterpy.spots import Spots
from butterpy.regions import regions
from butterpy.constants import RAD2DEG, PROT_SUN, FLUX_SCALE, DAY2MIN

np.random.seed(777)

dur = 3650  # Duration in days
cad = 30  # cadence in minutes


def generate_simdata(Nlc):
    incl = np.arcsin(np.sqrt(np.random.uniform(0, 1, Nlc)))
    ar = 10 ** np.random.uniform(low=-2, high=1, size=Nlc)
    clen = 10 ** np.random.uniform(low=0, high=1.6, size=Nlc)
    cover = 10 ** np.random.uniform(low=-1, high=0.5, size=Nlc)
    theta_low = np.random.uniform(low=0, high=40, size=Nlc)
    theta_high = np.random.uniform(low=theta_low + 5, high=80)
    period = 10.0 ** np.random.uniform(low=-1, high=2, size=Nlc)
    tau_evol = 10.0 ** np.random.uniform(low=0, high=1, size=Nlc)
    butterfly = np.random.choice([True, False], size=Nlc, p=[0.8, 0.2])
    delta_omega = 10.0 ** (np.random.uniform(-1, 0, size=Nlc))

    omega = PROT_SUN / period
    delta_omega *= omega

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
    sims["Delta Omega"] = delta_omega
    sims["Decay Time"] = tau_evol
    sims["Butterfly"] = butterfly
    sims = pd.DataFrame.from_dict(sims)
    return sims


def simulate(s):
    spot_properties = regions(
        butterfly=s["Butterfly"],
        activity_rate=s["Activity Rate"],
        cycle_length=s["Cycle Length"],
        cycle_overlap=s["Cycle Overlap"],
        max_ave_lat=s["Spot Max"],
        min_ave_lat=s["Spot Min"],
        tsim=dur,
        tstart=0,
    )

    time = np.arange(0, dur, cad / DAY2MIN)

    if len(spot_properties) == 0:
        dF = np.zeros_like(time)

    else:
        lc = Spots(
            spot_properties,
            incl=s["Inclination"],
            omega=s["Omega"],
            delta_omega=s["Delta Omega"],
            alpha_med=np.sqrt(s["Activity Rate"]) * FLUX_SCALE,
            decay_timescale=s["Decay Time"],
        )

        dF = lc.calc(time)

    return 0


if __name__ == "__main__":
    lengths = [1, 2, 3]
    ts = []
    for l in tqdm(lengths):
        sim = generate_simdata(l)
        t0 = datetime.now()
        for _, s in sim.iterrows():
            out = simulate(s)
        t1 = datetime.now()
        ts.append(t1 - t0)

    df = pd.DataFrame.from_dict({"num lightcurves": lengths, "time": ts})
    df.to_csv("python_times.csv")
