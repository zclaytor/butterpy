import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec, patches


def animate_spots(surface, time, projection="ortho", window_size=50, fig_kw=None, **kw):
    """
    Animate a plot of spot and light curve evolution. 
    Wrapper for `utils.visualization.animate_spots`.

    Parameters
    ----------
    time (numpy array):
        Array of time values to animate. Ideally you should only animate
        a subset of time steps, otherwise the animation will be too large.

    projection (str, "ortho"):
        Projection for animation. Can be orthographic ("ortho" or 
        "orthographic"), which requires cartopy, or cartesian ("cart" or
        "cartesian").

    window_size (float, 50):
        Window size, in days, for light curve viewing.

    fig_kw: dict of kwargs to be passed to `matplotlib.figure`.
    
    kw: remaining kwargs to be passed to `matplotlib.animation.FuncAnimation`.

    Returns
    -------
    ani (matplotlib.animation.FuncAnimation):
        animated plot of spot and light curve evolution.
    """
    if projection in ["ortho", "orthographic"]:
        return ortho_animation(surface, time, window_size, fig_kw, **kw)
    elif projection in ["cart", "cartesian"]:
        return cart_animation(surface, time, window_size, fig_kw, **kw)

def ortho_animation(surface, time, window_size=50, fig_kw=None, **kw):
    surface.assert_regions()
    surface.assert_lightcurve()

    try:
        import cartopy.crs as ccrs
    except:
        raise ImportError(
            "butterpy requires the use of cartopy to run orthographic animations, "
            "but cartopy is not installed. Please install using "
            "`conda install -c conda-forge cartopy`, or if using pip, see "
            "https://scitools.org.uk/cartopy/docs/latest/installing.html for details."
        )

    if fig_kw is None:
        fig = plt.figure(figsize=(5, 6))
    else:
        fig = plt.figure(**fig_kw)

    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=(1, 0.5))
    fig.subplots_adjust(
        top=0.93, bottom=0.12, left=0.21, right=0.95, hspace=0.05)
    ax1 = fig.add_subplot(gs[0], 
        projection=ccrs.Orthographic(0, 90 - surface.incl*180/np.pi))
    ax2 = fig.add_subplot(gs[1])

    ax1.set_global()
    ax1.gridlines(color='k', linestyle='dotted')
    ax1.scatter([], [], s=1, alpha=0.5, c="#996699", lw=0.5,
        transform=ccrs.PlateCarree())

    ax2.plot(surface.time, surface.flux)
    ax2.add_artist(
        patches.Ellipse(
            (0.5, 0.5),
            width=0.02,
            height=0.05,
            color="r",
            transform=ax2.transAxes,
            zorder=2,
        )
    )
    width = window_size / 2

    tmin = time[0] - width
    tmax = time[0] + width
    flux = surface.flux[(tmin <= surface.time) & (surface.time < tmax)]

    ax2.set_ylim(0.995*flux.min(), 1.005*flux.max())
    ax2.set_xticks(np.arange(time[0] - width / 2, time[-1] + width / 2, width / 2))
    ax2.set_xlim(time[0] - width, time[0] + width)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Relative Flux")
    ax2.vlines(0.5, color="r", ymin=0, ymax=1, transform=ax2.transAxes)
    title = fig.suptitle("", x=0.57)
    fig.align_labels()

    ax = (ax1, ax2)
    ani = animation.FuncAnimation(
        fig,
        _update_figure,
        frames=time,
        blit=False,
        repeat=False,
        fargs=(surface, ax, title, 'ortho'),
        **kw,
    )
    return ani

def cart_animation(surface, time, window_size=50, fig_kw=None, **kw):
    surface.assert_regions()
    surface.assert_lightcurve()

    nlat = surface.nlat
    nlon = surface.nlon
    dlon = 360 / nlon

    lat_width = 7
    lat_min = max(surface.min_lat - lat_width, 0)
    lat_max = surface.max_lat + lat_width

    if fig_kw is None:
        fig = plt.figure(figsize=(5, 6))
    else:
        fig = plt.figure(**fig_kw)
    
    fig.subplots_adjust(
        top=0.95, bottom=0.08, left=0.16, right=0.95, hspace=0.3, wspace=0.2
    )
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=(1, 0.5))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.set_xlim(-180, 180)
    ax1.set_xticks(np.linspace(-180, 180, 7))
    ymax = np.ceil(lat_max / 5) * 5
    ax1.set_ylim(-ymax, ymax)
    ax1.set_yticks(np.linspace(-ymax, ymax, 5))
    ax1.xaxis.set_ticklabels([180, 240, 300, 0, 60, 120, 180])

    ax1.vlines(
        np.linspace(dlon - 180, 180 - dlon, nlon - 1),
        ymin=lat_min,
        ymax=lat_max,
        lw=0.2,
    )
    ax1.vlines(
        np.linspace(dlon - 180, 180 - dlon, nlon - 1),
        ymin=-lat_max,
        ymax=-lat_min,
        lw=0.2,
    )
    ax1.hlines(np.linspace(lat_min, lat_max, nlat + 1), xmin=-180, xmax=180, lw=0.2)
    ax1.hlines(
        -np.linspace(lat_min, lat_max, nlat + 1), xmin=-180, xmax=180, lw=0.2
    )

    ax1.set_xlabel("Longitude (degrees)")
    ax1.set_ylabel("Latitude (degrees)")
    ax1.scatter([], [], s=1, alpha=0.5, c="#996699", lw=0.5)

    ax2.plot(surface.time, surface.flux)
    ax2.add_artist(
        patches.Ellipse(
            (0.5, 0.5),
            width=0.02,
            height=0.05,
            color="r",
            transform=ax2.transAxes,
            zorder=2,
        )
    )
    width = window_size / 2

    tmin = time[0] - width
    tmax = time[0] + width
    flux = surface.flux[(tmin <= surface.time) & (surface.time < tmax)]

    ax2.set_ylim(0.995*flux.min(), 1.005*flux.max())
    ax2.set_xticks(np.arange(time[0] - width / 2, time[-1] + width / 2, width / 2))
    ax2.set_xlim(tmin, tmax)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Relative Flux")
    ax2.vlines(0.5, color="r", ymin=0, ymax=1, transform=ax2.transAxes)
    title = fig.suptitle("")
    fig.align_labels()

    ax = (ax1, ax2)
    ani = animation.FuncAnimation(
        fig,
        _update_figure,
        frames=time,
        blit=False,
        repeat=False,
        fargs=(surface, ax, title),
        **kw,
    )
    return ani


def _update_figure(time, surface, ax, title, projection=None):
    ax1, ax2 = ax

    latitudes, longitudes, areas, fluxes = surface._calc_t(time, animate=True)
    longitudes = (longitudes * 180/np.pi) % 360
    longitudes[longitudes >= 180] -= 360

    if projection == 'ortho':
        coll_idx = 0
        sizes = -100_000 * fluxes
    else:
        coll_idx = -1
        sizes = 100_000 * areas

    im1 = ax1.collections[coll_idx]
    im1.set_offsets(np.c_[longitudes, latitudes * 180/np.pi])
    im1.set_sizes(sizes)
    new_flux = 1 + fluxes.sum()

    t1, t2 = ax2.get_xlim()
    t0 = (t1 + t2) / 2
    delta_t = time - t0
    t1new, t2new = t1 + delta_t, t2 + delta_t
    ax2.set_xlim(t1new, t2new)

    flux = surface.flux[(t1new <= surface.time) & (surface.time < t2new)]
    y1new, y2new = 0.995*flux.min(), 1.005*flux.max()
    ax2.set_ylim(y1new, y2new)

    circle_y = (new_flux - y1new) / (y2new - y1new)
    circle, = ax2.patches
    circle.set_center((0.5, circle_y))

    time_minutes = time * 24 * 60
    hours, minutes = divmod(time_minutes, 60)
    days, hours = divmod(hours, 24)
    title.set_text(f"Elapsed: {days:4.0f}d{hours:02.0f}h{minutes:02.0f}m")
    return (im1,)


if __name__ == "__main__":
    from butterpy import Surface

    np.random.seed(88)

    s = Surface()
    s.emerge_regions(activity_level=5, min_lat=10, max_lat=60, cycle_period=5, cycle_overlap=1)
    s.evolve_spots(period=10, incl=45)

    s.plot_butterfly()
    anim = animate_spots(s, np.arange(100, 200, 0.5), projection="ortho")
    plt.show()