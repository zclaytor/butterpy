import os
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from butterpy.spots import Spots
from butterpy.regions import regions
from butterpy.constants import FLUX_SCALE, DAY2MIN

np.random.seed(777)

DIR = os.path.dirname(__file__)


def simulate(s, dur=3650, cad=30):
    spot_properties = regions(
        butterfly=s["butterfly"],
        activity_rate=s["ar"],
        cycle_length=s["clen"],
        cycle_overlap=s["cover"],
        max_ave_lat=s["θ_high"],
        min_ave_lat=s["θ_low"],
        tsim=dur,
        tstart=0,
    )

    time = np.arange(0, dur, cad / DAY2MIN)
    if len(spot_properties) == 0:
        dF = np.zeros_like(time)

    else:
        lc = Spots(
            spot_properties,
            incl=s["inclination"],
            omega=s["ω"],
            delta_omega=s["Δω"],
            alpha_med=np.sqrt(s["ar"]) * FLUX_SCALE,
            decay_timescale=s["τ_decay"],
        )
        dF = lc.calc(time)

    return dF


if __name__ == "__main__":
    simdata = pd.read_csv(os.path.join(DIR, "benchmark_data.csv"))
    times = []
    fnames = []
    for i, s in tqdm(simdata.iterrows(), total=len(simdata)):
        t0 = datetime.now()
        out = simulate(s)
        t1 = datetime.now()
        times.append(t1 - t0)
        fname = os.path.join(DIR, "data", f"python_{i+1}.npy")
        np.save(fname, out)
        fnames.append(fname)

    df = pd.DataFrame.from_dict({"time": times, "datafile": fnames})
    df.to_csv(os.path.join(DIR, "python_times.csv"))
