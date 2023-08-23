import timeit
import pytest

import numpy as np
import butterpy as bp


np.random.seed(88)

s = bp.Surface()
s.emerge_regions(activity_level=5)


def spots():
    time = np.arange(0.0, 1000.0)
    lc = s.evolve_spots(time)
    return lc.time, lc.flux

def times():
    time = np.arange(0.0, 1000.0)
    flux = np.array([1 + s._calc_t(t).sum() for t in time], dtype="float32")
    return time, flux

def time_function(func, repeat=10, number=1):
    print(f"timing {str(func)}...")
    duration = timeit.Timer(func).repeat(repeat=repeat, number=number)
    avg_duration = np.average(duration)
    std = np.std(duration, ddof=1)
    print(f'{avg_duration:.3f} s ± {1000*std:.1f} ms per loop')


if __name__ == "__main__":
    time_function(spots)
    time_function(times)

    t1, f1 = spots()
    t2, f2 = times()

    assert f1 == pytest.approx(f2), "Fluxes do not match."



