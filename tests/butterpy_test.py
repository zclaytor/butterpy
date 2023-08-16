import os
import pytest
import numpy as np
from astropy.table import Table
from butterpy.core import regions, spots


cwd = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def default_surface():
    np.random.seed(88)
    return regions(
        activityrate=1, minlat=5, maxlat=35, 
        cyclelength=3, cycleoverlap=1, ndays=3600)
    
@pytest.fixture
def load_test_surface():
    return Table.read(os.path.join(cwd, "data/default_surface.fits"))

@pytest.fixture
def load_test_flux():
    return np.load(os.path.join(cwd, "data/default_spots.npy"))

def test_regions(default_surface):
    surface = default_surface
    assert isinstance(surface, Table), "Result of `regions` call is not an astropy Table."

def test_regions_output(default_surface, load_test_surface):
    surface = default_surface
    expected = load_test_surface
    assert surface == pytest.approx(expected), "Calculated surface does not match expectation."

def test_flux(default_surface, load_test_flux):
    s = spots(default_surface)
    time = np.arange(0, 3600, 0.1)
    f_calc = s.calc(time)
    f_expected = load_test_flux
    assert f_calc == pytest.approx(f_expected), "Calculated flux does not match expectation."