import os

root = "/home/zach/Desktop/tesstoys/"
sim_dir = os.path.join(root, "lightcurves/")
simulation_properties_dir = os.path.join(sim_dir, "simulation_properties.csv")

# Create directories if they don't exist
if not os.path.exists(sim_dir):
    os.makedirs(sim_dir)

Nlc = 100  # How many light curves are we making?
dur = 3650  # Duration in days
cad = 30  # cadence in minutes