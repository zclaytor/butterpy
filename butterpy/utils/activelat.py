import numpy as np

# def active_latitudes(min_ave_lat, max_ave_lat, phase, butterfly=True):
#     if butterfly:
#         return exponential_latitudes(min_ave_lat, max_ave_lat, phase)
#     return random_latitudes(min_ave_lat, max_ave_lat, phase)

# def exponential_latitudes(min_lat, max_lat, phase):
#     # Based on Hathaway 2015, LRSP 12: 4
#     phase_scale = 1 / np.log(max_lat / min_lat)
#     lat_avg = max_lat * np.exp(-phase / phase_scale)
#     # See Hathaway 2010, p. 37 for a discussion of the width
#     lat_rms = max_lat / 5 - phase * (max_lat - min_lat) / 7

#     return lat_avg, lat_rms

# def random_latitudes(min_lat, max_lat, phase):
#     lat_avg = (max_lat + min_lat) / 2 * np.ones_like(phase)
#     lat_rms = (max_lat - min_lat) * np.ones_like(phase)

#     return lat_avg, lat_rms

def random(minlat, maxlat):
    nlat = 16
    dlat = maxlat/nlat
    latavg = (maxlat - minlat) / 2.
    latrms = (maxlat - minlat)
    nlat1 = int(minlat / dlat)
    nlat2 = int(maxlat / dlat)
    nlat2 = np.min([nlat2, nlat-1])
    return latavg, latrms, nlat1, nlat2

def quadratic(minlat, maxlat, phase):
    #This is a bit of a fudge. For the sun, y =35 - 48x + 20x^2
    nlat = 16
    dlat = maxlat/nlat
    latrmsd = (maxlat-minlat) / 7.
    latavg = maxlat - (maxlat+minlat)*phase + \
            +2*minlat*phase**2.
    latrms = (maxlat/5.) - latrmsd*phase
    nlat1 = int(np.max([(maxlat*0.9) - (1.2*maxlat)*phase, 0.])/dlat)
    nlat2 = int(np.min([(maxlat + 6.) - maxlat*phase, maxlat])/dlat)
    nlat2 = np.min([nlat2, nlat-1])
    return latavg, latrms, nlat1, nlat2

# def exponential(minlat, maxlat, phase):
#     # Based on Hathaway 2015, LRSP 12: 4
#     phase_scale = 1 / np.log(maxlat / minlat)
#     latavg = maxlat * np.exp(-phase / phase_scale)
#     # See Hathaway 2010, p. 37 for a discussion of the width
#     latrms = maxlat / 5 - phase * (maxlat - minlat) / 7
#     nlat1 = 

#     return latavg, latrms