import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from spots import Spots
from regions import regions
from config import tess_stars_dir, sim_dir, simulation_properties_dir
from config import Nlc, dur, cad
from constants import RAD2DEG, PROT_SUN, FLUX_SCALE, DAY2MIN


jstart = 0


def scale_tess_flux(f1, f2):
    '''Routine to scale the TESS ETE simulations'''
    ii = f2 > 0
    keep = f2[~ii]
    jj = f1 > 0
    f2 -= np.median(f2[ii])
    f2 /= np.std(f2[ii])
    f2 *= np.std(f1[jj])
    f2 += np.median(f1[jj])
    f2[~ii] = keep
    return f2


def generate_simdata():
    incl = np.arcsin(np.sqrt(np.random.uniform(0, 1, Nlc)))
    ar = 10**np.random.uniform(low=-2, high=1, size=Nlc)
    clen = 10**np.random.uniform(low=0, high=1.6, size=Nlc)
    cover = 10**np.random.uniform(low=-1, high=0.5, size=Nlc)
    theta_low = np.random.uniform(low=0, high=40, size=Nlc)
    theta_high = np.random.uniform(low=theta_low+5, high=80)
    period = 10.0**np.random.uniform(low=-1, high=2, size=Nlc)
    tau_evol = 10.0**np.random.uniform(low=0, high=1, size=Nlc)
    butterfly = np.random.choice([True,False], size=Nlc, p=[0.8, 0.2])
    delta_omega = 10.0**(np.random.uniform(-1, 0, size=Nlc))

    omega = PROT_SUN / period
    delta_omega *= omega

    plt.figure(figsize=(12, 7))
    plt.subplot2grid((2, 3), (0, 0))
    plt.hist(period, 20, color='C0')
    plt.xlabel("Rotation Period (days")
    plt.ylabel("N")
    plt.subplot2grid((2, 3), (0, 1))
    plt.hist(tau_evol, 20, color='C1')
    plt.xlabel("Spot lifetime (Prot)")
    plt.ylabel("N")
    plt.subplot2grid((2, 3), (0, 2))
    plt.hist(incl*RAD2DEG, 20, color='C3')
    plt.xlabel("Stellar inclincation (deg)")
    plt.ylabel("N")
    plt.subplot2grid((2, 3), (1, 0))
    plt.hist(ar, 20, color='C4')
    plt.xlabel("Stellar activity rate (x Solar)")
    plt.ylabel("N")
    plt.subplot2grid((2, 3), (1, 1))
    plt.hist(delta_omega, 20, color='C5')
    plt.xlabel(r"Differential Rotation Shear $\Delta \Omega$ (x Solar)")
    plt.ylabel("N")
    plt.subplot2grid((2, 3), (1, 2))
    plt.hist(theta_high - theta_low, 20, color='C6')
    plt.xlabel("Spot latitude range")
    plt.ylabel("N")
    plt.tight_layout()
    plt.savefig(f"{sim_dir:s}distributions.png", dpi=150)


    # Stitch this all together and write the simulation properties to file
    sims = {}
    sims['Activity Rate'] = ar
    sims['Cycle Length'] = clen
    sims['Cycle Overlap'] = cover
    sims['Inclination'] = incl
    sims['Spot Min'] = theta_low
    sims['Spot Max'] = theta_high   
    sims['Period'] = period   
    sims['Omega'] = omega      
    sims['Delta Omega'] = delta_omega
    sims['Decay Time'] = tau_evol
    sims['Butterfly'] = butterfly
    sims = pd.DataFrame.from_dict(sims)
    sims.to_csv(simulation_properties_dir, float_format='%5.4f')

    return sims


def simulate(s, fig, ax, out_str):
    spot_properties = regions(butterfly=s['Butterfly'],
                              activity_rate=s['Activity Rate'],
                              cycle_length=s['Cycle Length'],
                              cycle_overlap=s['Cycle Overlap'],
                              max_ave_lat=s['Spot Max'],
                              min_ave_lat=s['Spot Min'],
                              tsim=dur,
                              tstart=0)

    time = np.arange(0, dur, cad/DAY2MIN)

    if len(spot_properties) == 0:
        dF = np.zeros_like(time)

    else:
        lc = Spots(spot_properties,
                   incl = s['Inclination'],
                   omega = s['Omega'],
                   delta_omega=s['Delta Omega'],
                   alpha_med=np.sqrt(s['Activity Rate'])*FLUX_SCALE,
                   decay_timescale=s['Decay Time'])
               
        dF = lc.calc(time)

    final = Table(np.c_[time, 1+dF], names=['TIME', 'MODEL_FLUX'])
    hdu_lc = fits.BinTableHDU(final)
    hdu_spots = fits.BinTableHDU(Table.from_pandas(spot_properties))
    hdul = fits.HDUList([fits.PrimaryHDU(),hdu_lc, hdu_spots])
    hdul.writeto(f'{sim_dir}/{out_str}.fits', overwrite=True)
    
    # Plot the butterfly pattern
    lat = spot_properties['lat']
    
    ax[0].scatter(spot_properties['nday'], lat*RAD2DEG,
        s=spot_properties['bmax']/np.median(spot_properties['bmax'])*10,
        alpha=0.5, c='#996699', lw=0.5)
    ax[0].set_ylim(-90, 90)
    ax[0].set_yticks((-90, -45, 0, 45, 90))
    ax[0].set_ylabel('Latitude (deg)')
    ax[1].plot(final['TIME'], final['MODEL_FLUX'], c='C2')
    ax[1].set_ylabel("Model Flux")
    ax[0].set_title(f"Simulation {out_str}")
    fig.savefig(f'{sim_dir}/{out_str}.png', dpi=150)
    
    for ii in range(0, len(ax)):
        ax[ii].cla()

    return 0


if __name__ == "__main__":
    np.random.seed(7)

    if len(sys.argv) > 1:
        if sys.argv[1] == 'new': 
            sims = generate_simdata()
            jstart = 0

    else:
        try:
            sims = pd.read_csv(simulation_properties_dir)
        except FileNotFoundError:
            print('Missing simulation properties file.')
            print('Try running with keyword "new".')
            exit()           

    
    # Make the light curves
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8.5, 6))
    ax[0].set_ylabel("Flux")
    ax[0].set_title("Test")
    ax[-1].set_xlabel("Test")
    ax[0].set_ylim(-1e4,1e4)
    fig.tight_layout()
    for ii in range(0, len(ax)):
        ax[ii].cla()

    pbar = tqdm(total=Nlc)
    pbar.update(jstart)
    for jj, s in sims.iloc[jstart:].iterrows():
        out = simulate(s, fig, ax, out_str=f'{jj:04d}')
        pbar.update(1)