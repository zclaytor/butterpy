import pytest

import numpy as np

import butterpy.distributions as dist


def basic_test_helper(d):
    assert isinstance(repr(d), str), f"{d}: `repr` method does not return a string."
    try:
        assert isinstance(d.sample(), float), f"{d}: default sample is not a float."
    except AssertionError:
        assert isinstance(d.sample(), int), f"{d}: default sample is neither float nor int."
    assert len(d.sample(1)) == 1, f"{d}: default sample is not length 1."
    assert len(d.sample(2)) == 2, f"{d}: default sample is not length 2."

def test_uniform():
    basic_test_helper(dist.Uniform())

def test_loguniform():
    basic_test_helper(dist.LogUniform())

def test_sinesquared():
    basic_test_helper(dist.SineSquared())

def test_boolean():
    d = dist.Boolean()
    assert isinstance(repr(d), str), f"{d}: `repr` method does not return a string."
    assert isinstance(d.sample(), np.int64), f"{d}: default sample is not an int."

def test_fixed():
    basic_test_helper(dist.Fixed())

def test_composite():
    d = dist.Composite(
        [dist.Uniform(0, 1), dist.LogUniform(1, 10)],
        weights=[1, 3],
    )
    basic_test_helper(d)

    # Check unshuffled behavior
    s = d.sample(4, shuffle=False)
    assert (s[0] < s[1:]).all(), "Composite distribution is shuffled when it shouldn't be."

def test_composite_rounding():
    d = dist.Composite(
        [dist.Uniform(0, 1), dist.LogUniform(1, 10), dist.Uniform(10, 20)],
        weights=[1, 1, 1],
    )
    # Just rounding makes sample(4) --> 3 and sample(5) --> 6, so test the fixes
    assert len(d.sample(4)) == 4, f"{d}: default sample is not length 4."
    assert len(d.sample(5)) == 5, f"{d}: default sample is not length 5."

def test_exceptions():
    d = dist.Composite([dist.Uniform(-1, 0), dist.Uniform(0, 1)]
    )
    try:
        d._sample_one(size=2)
    except ValueError:
        pass

