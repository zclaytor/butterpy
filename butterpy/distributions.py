import numpy as np


class Distribution(object):
    """Base probability distribution class.
    Distribution has shape `shape` and range [min, max).
    """
    def __init__(self, min, max, shape):
        """
        Initialize distribution shape.

        Args:
            min (float): distribution range minimum.
            max (float): distribution range maximum.
            shape (str): distribution shape, e.g., 'Uniform'.           

        """
        self.min = min
        self.max = max
        self.shape = shape

    def __repr__(self):
        """Print distribution shape and range.
        """
        repr = f"butterpy.Distribution with shape `{self.shape}` from {self.min} to {self.max}"
        return repr
    

class Uniform(Distribution):
    """Uniform distribution from `min` to `max`.
    """
    def __init__(self, min=0, max=1):
        """Creates a Uniform distribution with range [min, max).
        That is, the values are uniformly distributed between min (inclusive)
        and max (exclusive):
            x ~ U(min, max).
        """
        super().__init__(min, max, shape="Uniform")

    def sample(self, size=None):
        """Sample the distribution, with optional `size` argument. 
        `size` is passed directly to `numpy.random.uniform`, so the behavior
        matches the `numpy` behavior.

        Args:
            size (int): The number of times to sample the distribution. 
                Defaults to None, in which case a single float is returned.
                Otherwise, an array with length `size` is returned.

        Returns:
            sample (float or numpy.ndarray): The samples from the distribution.
        """
        return np.random.uniform(low=self.min, high=self.max, size=size)
    

class LogUniform(Distribution):
    """Log Uniform distribution from `min` to `max`.
    That is, the logarithms of the values are uniformly distributed.

    This is accomplished using inverse transform sampling:
        log10(x) ~ U(log10(min), log10(max)).
    """
    def __init__(self, min=1, max=10):
        """Creates a LogUniform distribution with range [min, max).
        """        
        super().__init__(min, max, shape="LogUniform")

    def sample(self, size=None):
        """Sample the distribution, with optional `size` argument. 
        `size` is passed directly to `numpy.random.uniform`, so the behavior
        matches the `numpy` behavior.

        Args:
            size (int): The number of times to sample the distribution. 
                Defaults to None, in which case a single float is returned.
                Otherwise, an array with length `size` is returned.

        Returns:
            sample (float or numpy.ndarray): The samples from the distribution.
        """
        return 10**np.random.uniform(
            low=np.log10(self.min), high=np.log10(self.max), size=size)
    

class SineSquared(Distribution):
    """Uniform in sin^2 from `min` to `max`.
    That is, the squared sines of the values are uniformly distributed.

    This is accomplished using inverse transform sampling:
        sin^2 (x) ~ U(sin^2 (min), sin^2 (max)).
    """
    def __init__(self, min=0, max=1):
        """Creates a SineSqaured distribution with range [min, max).
        """        
        super().__init__(min, max, shape="SineSquared")

    def sample(self, size=None):
        """Sample the distribution, with optional `size` argument. 
        `size` is passed directly to `numpy.random.uniform`, so the behavior
        matches the `numpy` behavior.

        Args:
            size (int): The number of times to sample the distribution. 
                Defaults to None, in which case a single float is returned.
                Otherwise, an array with length `size` is returned.

        Returns:
            sample (float or numpy.ndarray): The samples from the distribution.
        """        
        return np.arcsin(np.sqrt(
            np.random.uniform(np.sin(self.min)**2, np.sin(self.max)**2, size)))
    

class Composite(Distribution):
    """Composite distribution with specified weights for each part.
    The weights are internally renormalized to add to unity.

    Example:
        c = Composite(
            [Uniform(0, 1), LogUniform(1, 10)],
            weights=[1, 3])

    `c.sample(100)` will return 25 values uniformly sampled from [0, 1)
    and 75 values logarithmically sampled from [1, 10).
    """
    def __init__(self, distributions, weights):
        """Initialize Composite distribution.

        Args:
            distributions (list-like): List of initialized Distributions.
            weights (list-like): List of relative weights corresponding to
                each distribution.
        """
        self.distributions = distributions
        self.weights = np.asarray(weights)/sum(weights)

    def __repr__(self):
        """Print each distribution shape and range.
        """
        repr = f"butterpy.Composite distribution with:\n" \
            + "\n".join([f"  {w*100:2.0f}%: {d.__repr__()}" 
                         for w, d in zip(self.weights, self.distributions)])
        return repr
    
    def _sample_one(self, size=None):
        """Specialized behavior for a single sample,
        using `numpy.random.choice` with weights to choose which
        distribution to sample.

        Args:
            size (int): The number of samples, which must be `None` or 1.
                For `None`, a single float is returned. For 1, an array
                with length 1 is returned.

        Returns:
            sample (float or numpy.ndarray): The single sample value.

        """
        if size not in [None, 1]:
            raise ValueError("`size` must be either `None` or 1.")
        
        d = np.random.choice(self.distributions, p=self.weights)
        return d.sample(size=size)
    
    def sample(self, size=None, shuffle=True):
        """Sample the Composite distribution, with optional `size` argument. 
        `size` behavior is intended to mimic that of `numpy.random.uniform`.

        Args:
            size (int): The number of times to sample the distribution. 
                Defaults to None, in which case a single float is returned.
                Otherwise, an array with length `size` is returned.
            shuffle (bool): Whether to shuffle the samples between distributions.
                True by default, but False will return, e.g.,
                array([*sample1, *sample2, ...]),
                with the samples ordered by the distribution they're pulled from.

        Returns:
            sample (float or numpy.ndarray): The samples from the distribution.
        """                
        if size is None or size == 1:
            return self._sample_one(size)
        
        n_samples = (size*self.weights).round().astype(int)
        samples = np.concatenate(
            [d.sample(n) for d, n in zip(self.distributions, n_samples)]
        )

        # If under the requested size by 1, generate a bonus sample
        if len(samples) == size - 1:
            samples = np.append(samples, self._sample_one())

        # Everything past here can be shuffled
        if shuffle:
            np.random.shuffle(samples)

        # If over the requested size by 1, shuffle and truncate
        if len(samples) == size + 1:
            samples = samples[:-1]

        if len(samples) != size:
            # If we make it here, something went horribly wrong.
            raise ValueError(f"Something has gone horribly wrong. "
                             f"{size} samples requested; {len(samples)} returned.")

        return samples
    

class Boolean(Distribution):
    """docs
    """
    def __init__(self, p=0.5):
        """docs
        """
        assert 0 <= p <= 1, "`p` must be between 0 and 1."
        self.p = p

    def __repr__(self):
        """docs
        """
        return f"Boolean distribution with p(True) = {self.p}"
    
    def sample(self, size=None):
        return np.random.choice(
            [1, 0], p=[self.p, 1-self.p], size=size)