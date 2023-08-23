"""Functions to calculate spot evolution.
"""
import numpy as np

def gaussian_spots(t, tau_emerge, tau_decay):
    area_factor = np.ones_like(t)

    l = t < 0
    area_factor[l] *= np.exp(-t[l]**2 / tau_emerge**2) # emergence

    l = t > 0
    area_factor[l] *= np.exp(-t[l]**2 / tau_decay**2) # decay

    return area_factor

def exponential_spots(t, tau_emerge, tau_decay):
    area_factor = np.ones_like(t)

    l = t < 0
    area_factor[l] *= np.exp(t[l] / tau_emerge) # emergence

    l = t > 0
    area_factor[l] *= np.exp(-t[l] / tau_decay) # decay

    return area_factor    


def plot_spot_evolution():
    import matplotlib.pyplot as plt
    time = np.arange(-10, 10, 0.1)
    t1, t2 = 3, 5
    plt.plot(time, exponential_spots(time, t1, t2), label="exponential evolution")
    plt.plot(time, gaussian_spots(time, t1, t2), label="gaussian evolution")
    plt.axvline(-3, color="k", linestyle=":", label="emergence time")
    plt.axvline(5, color="k", linestyle="--", label="decay time")
    plt.xlabel("time")
    plt.ylabel("relative area")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_spot_evolution()