import numpy as np


class Distribution(object):
    """Base probability distribution class.
    """
    def __init__(self, min, max, shape):
        """DOCSTRING
        """
        self.min = min
        self.max = max
        self.shape = shape

    def __repr__(self):
        """DOCSTRING
        """
        repr = f"butterpy.Distribution with shape `{self.shape}` from {self.min} to {self.max}"
        return repr
    

class Uniform(Distribution):
    """Uniform distribution from `min` to `max`.
    """
    def __init__(self, min=0, max=1):
        super().__init__(min, max, shape="Uniform")

    def sample(self, size=None):
        return np.random.uniform(low=self.min, high=self.max, size=size)
    

class LogUniform(Distribution):
    """Log Uniform distribution from `min` to `max`.
    """
    def __init__(self, min=1, max=10):
        super().__init__(min, max, shape="LogUniform")

    def sample(self, size=None):
        return 10**np.random.uniform(
            low=np.log10(self.min), high=np.log10(self.max), size=size)
    

class SineSquared(Distribution):
    """Uniform in sin^2 from `min` to `max`.
    """
    def __init__(self, min=0, max=1):
        super().__init__(min, max, shape="SineSquared")

    def sample(self, size=None):
        return np.arcsin(np.sqrt(
            np.random.uniform(np.sin(self.min)**2, np.sin(self.max)**2, size)))
    

class Composite(Distribution):
    """Composite distribution with specified weights for each part.
    """
    def __init__(self, dists, weights):
        self.distributions = dists
        self.weights = np.asarray(weights)/sum(weights)

    def __repr__(self):
        repr = f"butterpy.Composite distribution with:\n" \
            + "\n".join([f"  {w*100:2.0f}%: {d.__repr__()}" 
                         for w, d in zip(self.weights, self.distributions)])
        return repr
    
    def _sample_one(self, size=None):
        d = np.random.choice(self.distributions, p=self.weights)
        return d.sample(size=size)
    
    def sample(self, size=None, shuffle=True):
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


if __name__ == "__main__":
    np.random.seed(88)

    d1 = Distribution(1, 10, "weird")
    d2 = Uniform()
    d3 = LogUniform()
    d4 = SineSquared()

    print(d1, d2, d3)
    print(d3.sample(10))
    print(d4.sample(10))

    dcomp = Composite([Uniform(-1, 0), Uniform(0, 1)], [0.25, 0.75])
    print(dcomp)
    s = dcomp.sample(12)
    print(s)