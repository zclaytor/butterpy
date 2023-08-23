import timeit
import numpy as np
from butterpy.core import Surface

def default_regions():
    return Surface().emerge_regions(activity_level=1, min_lat=5, max_lat=35, 
        cycle_period=3, cycle_overlap=1, ndays=3600)

def high_activity_regions():
    return Surface().emerge_regions(activity_level=5, min_lat=5, max_lat=35, 
        cycle_period=3, cycle_overlap=1, ndays=3600)

def low_activity_regions():
    return Surface().emerge_regions(activity_level=0.3, min_lat=5, max_lat=35, 
        cycle_period=3, cycle_overlap=1, ndays=3600)

def time(func, repeat=10, number=1):
    print(f"timing {str(func)}...")
    duration = timeit.Timer(func).repeat(repeat=repeat, number=number)
    avg_duration = np.average(duration)
    std = np.std(duration, ddof=1)
    print(f'{avg_duration:.3f} s ± {1000*std:.1f} ms per loop')

if __name__ == "__main__":
    time(low_activity_regions)
    time(default_regions)
    time(high_activity_regions)