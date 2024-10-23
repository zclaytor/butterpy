"""
core.py

Core module of the photometry subpackage containing a class definition for a
set of filters.
"""

import os
import matplotlib.pyplot as plt

from astropy.io import ascii


_root = os.path.abspath(os.path.dirname(__file__))


class Filter:
    def __init__(self, wavelength, response, name=None):
        self.wavelength = wavelength
        self.response = response
        self.name = name
    
    def plot(self, ax=None, **kw):
        if ax is None:
            fig, ax = plt.subplots()
   
        ax.plot(self.wavelength, self.response, **kw)
        return ax

    def __repr__(self):
        return f"{type(self)} with name '{self.name}'"


def get_filter(name):
    mission, filter = name.split(".")
    if mission.lower() == "roman":
        try:
            datapath = os.path.join(_root, "filterdata/Roman_effarea_v8_median_20240301.csv")
            fdata = ascii.read(datapath, include_names=["Wave", filter.upper()])
            f = Filter(fdata["Wave"].value, fdata[filter].value, name=name)
        except KeyError:
            raise NotImplementedError(f"Filter '{name}' not implemented.")
        return f
    else:
        raise NotImplementedError(f"Filter '{name}' not implemented.")


if __name__ == "__main__":
    filter = get_filter("roman.F184")
    print(filter)
    filter.plot()
    plt.show()

    get_filter("roman.other")