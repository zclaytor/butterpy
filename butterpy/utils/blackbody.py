import matplotlib.pyplot as plt

from astropy.modeling import models
import astropy.units as u

_bb_base_unit = u.erg / (u.cm**2 * u.s * u.sr)


class BlackBody:
    def __init__(self, temperature, wavelength=None, frequency=None):
        if not hasattr(temperature, "unit"):
            temperature *= u.K
        self.temperature = temperature

        if wavelength is not None:
            if not hasattr(wavelength, "unit"):
                wavelength *= u.AA
            scale = 1*_bb_base_unit/wavelength.unit
            self.wavelength = wavelength
            self.kind = "lambda"
        elif frequency is not None:
            if not hasattr(frequency, "unit"):
                frequency *= u.Hz
            scale = 1*_bb_base_unit/frequency.unit
            self.frequency = frequency
            self.kind = "nu"
        else:
            raise ValueError("Either `wavelength` or `frequency` must be set.")
        
        bb = models.BlackBody(temperature=temperature, scale=scale)
        self.flux = bb(wavelength)

    def plot(self, ax=None, **kw):
        if ax is None:
            fig, ax = plt.subplots()
        if self.kind == "lambda":
            spec_coord = self.wavelength
        elif self.kind == "nu":
            spec_coord = self.frequency

        ax.plot(spec_coord, self.flux, **kw)
        return ax
    
    def __repr__(self):
        return f"{type(self)} of form B_{self.kind} and temperature {self.temperature:.0f}."
    


if __name__ == "__main__":
    import numpy as np
    bb = BlackBody(5000, np.arange(1000, 10000, 1))
    print(bb)
    bb.plot()
    plt.show()