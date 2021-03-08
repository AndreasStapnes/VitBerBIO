import numpy as np
import matplotlib.pyplot as plt

t = 10
dx = 0.001
mu = 0.1

trajectory = np.arange(0, t+dx, dx)


def MonteCarlo(N):
    photons_remaining = np.zeros(len(trajectory))
    removed_photons = 0
    remaining_photons = N
    p = mu*dx
    for i in range(len(trajectory)):
        r = np.random.rand(remaining_photons)
        for j in range(remaining_photons):
            if p > r[j]:
                removed_photons += 1
                remaining_photons = N - removed_photons
                
        photons_remaining[i] = remaining_photons
    return photons_remaining


N_list = [100, 1000, 5000, 10000]
N_colors = ["blue", "red", "green", "yellow"]  


def prettyPlot():
    plt.figure(figsize=(12,8))
    plt.suptitle("Monte Carlo simulation of photon attenuation")
    plt.xlabel(r"$x[cm]$")
    plt.ylabel(r"$\frac{I}{I_0}$")
    plt.plot(trajectory, np.exp(-trajectory*mu), label="Analytical", color="black")
    for n in range(len(N_list)):
        plt.plot(trajectory, MonteCarlo(N_list[n])/N_list[n], color=N_colors[n], label=N_list[n])
    plt.legend()
    plt.show()

prettyPlot()
