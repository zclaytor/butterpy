import os

root = '/home/zclaytor/'
sim_dir = os.path.join(root, 'nfs_fs02/simulated_lightcurves/')
spots_dir = os.path.join(root, 'nfs_fs02/spot_properties/')

# Create directories if they don't exist
if not os.path.exists(sim_dir):
    os.makedirs(sim_dir)
if not os.path.exists(spots_dir):
    os.makedirs(spots_dir)

Nlc = 1000000 # How many light curves are we making? 
dur = 3650 # Duration in days
cad = 30 # cadence in minutes
