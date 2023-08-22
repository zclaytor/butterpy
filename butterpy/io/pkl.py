"""The .pkl module contains functions for reading and writing pickle files."""


import pickle as pk


def to_pickle(surface, filename):
    """
    Write Surface object to pickle file.

    Parameters
    ----------
    surface (butterpy.Surface): the surface object to be saved.

    filename (str): output file path.

    Returns None.
    """
    with open(filename, "wb") as f:
        pk.dump(surface, f)


def read_pickle(filename):
    """
    Read Surface object from pickle file.

    Parameters
    ----------
    filename (str): file path to be read.

    Returns
    -------
    surface (butterpy.Surface): the Surface object read from file.
    """
    with open(filename, "rb") as f:
        surface = pk.load(f)
    
    return surface