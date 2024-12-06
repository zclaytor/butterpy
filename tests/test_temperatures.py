import os
import pytest
import numpy as np
from astropy.table import Table
from butterpy import Surface, read_fits


cwd = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def default_surface_regions():
    np.random.seed(88)
    s = Surface(tsurf=5800, tspot=4500)
    s.emerge_regions(
        activity_level=1, min_lat=5, max_lat=35,
        cycle_period=3, cycle_overlap=1, ndays=3600)
    return s

@pytest.fixture(scope="session")
def load_test_roman_flux():
    return np.load(os.path.join(cwd, "data/default_flux_roman.npy"))

def test_roman_filters(default_surface_regions, load_test_roman_flux):
    surface = default_surface_regions
    l = surface.evolve_spots(time=np.arange(0, 3600, 0.1), filters=["roman.f062", "roman.f146"])
    l_expected = load_test_roman_flux
    assert np.array(l) == pytest.approx(l_expected), "Roman filter light curves do not match."