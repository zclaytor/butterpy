import os
import pytest
import numpy as np
from astropy.table import Table
from butterpy.core import regions


cwd = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def default_surface():
    np.random.seed(88)
    return regions(
        activityrate=1, minlat=5, maxlat=35, 
        cyclelength=3, cycleoverlap=1, ndays=3600)
    
@pytest.fixture
def load_test_data():
    return Table.read(os.path.join(cwd, "data/default_surface.fits"))

def test_regions(default_surface):
    surface = default_surface
    assert isinstance(surface, Table), "Result of `regions` call is not an astropy Table."

def test_output(default_surface, load_test_data):
    surface = default_surface
    expected = load_test_data
    assert len(surface) == len(expected), "Length of `regions` output does not match test data."
    assert surface["thpos"][0] == expected["thpos"][0], "First `thpos` values do not match."
    assert surface["nday"][-1] == expected["nday"][-1], "Final `nday` values do not match."