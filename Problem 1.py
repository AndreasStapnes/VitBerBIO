# Numpy to make life easy
import numpy as np

# Matplotlib for plotting
import matplotlib.pyplot as plt

# Increase default font size
plt.rcParams.update({'font.size': 14})

# Import for progress meter
from tqdm import tqdm, trange



dx = 0.01; t = 10; mu = 0.1; p = mu*dx #Definerer gitte verdier

trajectory = np.arange(0, t + dx, dx) #Definerer de forskjellie stegene gjennom matrialet

def propagation(): #returnere true når fotonet ikke scatterer
    r = np.random.rand()
    if(r > p):
        return True

def MonteCarlo(): #returnerer true når fotonet kommer gjennom hele matrialet
    i = 0
    while(propagation() and i < len(trajectory)):
        i += 1
    if(i == len(trajectory)):
        return True
    else:
        return i


def MonteCarloN(N):
    start = np.ones(len(trajectory))*N
    for i in range(N):
        out = MonteCarlo()
        if(out != 1):
            start[out:] -= 1
    return start


N_list = [100, 1000, 5000, 10000]
N_farge = ["blue", "red", "green", "yellow"]

def prettyPlot():
    plt.figure(figsize=(12,8))
    plt.suptitle("Monte Carlo simulering av foton demping")
    plt.xlabel(r"$x[cm]$")
    plt.ylabel(r"$\frac{I}{I_0}$")
    plt.plot(trajectory, np.exp(-trajectory*mu), label="Analytisk", color="magenta")
    for n in range(len(N_list)):
        plt.plot(trajectory, MonteCarloN(N_list[n])/N_list[n], color=N_farge[n], label=N_list[n])
    plt.legend()
    plt.show()

prettyPlot()