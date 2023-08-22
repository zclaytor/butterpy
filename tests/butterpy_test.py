import os
import pytest
import numpy as np
from astropy.table import Table
from butterpy import Surface, unpickle


cwd = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def default_surface_regions():
    np.random.seed(88)
    s = Surface()
    s.emerge_regions(
        activity_level=1, min_lat=5, max_lat=35,
        cycle_period=3, cycle_overlap=1, ndays=3600)
    return s
    
@pytest.fixture(scope="session")
def default_surface(default_surface_regions):
    s = default_surface_regions
    s.evolve_spots(time=np.arange(0, 3600, 0.1))
    return s

@pytest.fixture(scope="session")
def load_test_surface():
    return Table.read(os.path.join(cwd, "data/default_surface.fits"))

@pytest.fixture(scope="session")
def load_test_flux():
    return np.load(os.path.join(cwd, "data/default_spots.npy"))

def test_regions(default_surface_regions):
    regions = default_surface_regions.regions
    assert isinstance(regions, Table), "Result of `regions` call is not an astropy Table."

def test_regions_output(default_surface_regions, load_test_surface):
    regions = default_surface_regions.regions
    expected = load_test_surface
    for i, j in zip(regions.iterrows(), expected.iterrows()):
        assert i == pytest.approx(j), "Rows from calculated surface do not match expectation."

def test_flux(default_surface, load_test_flux):
    f_calc = default_surface.lightcurve.flux
    f_expected = load_test_flux
    assert f_calc == pytest.approx(f_expected), "Calculated flux does not match expectation."

def test_lightcurve(default_surface):
    s = default_surface
    assert isinstance(s.time, np.ndarray), "`Surface.lightcurve.time` does not appear to be correctly set."
    assert isinstance(s.flux, np.ndarray), "`Surface.lightcurve.flux` does not appear to be correctly set."

def test_plots():
    assert 0, "tests not yet implemented."

def test_pickle(default_surface, tmp_path):
    s = default_surface
    fname = tmp_path / "test-surface.pkl"
    s.pickle(fname)
    sprime = unpickle(fname)

    for i, j in zip(s.regions.iterrows(), sprime.regions.iterrows()):
        assert i == pytest.approx(j), "Rows from unpickled surface do not match original."

    assert s.lightcurve.flux == pytest.approx(sprime.lightcurve.flux), "Unpickled flux does not match original."