# constants.py
from numpy import pi

DAY2MIN = 24 * 60  # days to minutes
DAY2SEC = DAY2MIN * 60  # days to seconds
YEAR2DAY = 365  # years to days
RAD2DEG = 180 / pi  # radians to degrees

PROT_SUN = 24.5  # siderial rotation period at the equator
OMEGA_SUN = 2 * pi / (PROT_SUN * DAY2SEC)

FLUX_SCALE = 3e-4  # Scaled to Sun
