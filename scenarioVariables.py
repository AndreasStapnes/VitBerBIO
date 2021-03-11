xLim = (-2.0, 2.0);
yLim = (-2.0, 2.0);
zLim = (-2.0, 2.0);
timestep = 2.6e-11;
c = 3e8

K1 = 9e9
K2 = 0.02e9

from numba import jit
import numpy as np
import math

with open("data/object1_50keV.npy", "rb") as file:
    box = np.load(file)
    box *= 100 #For å gjøre om fra cm^-1 til m^-1

pixelSize = 1/128;
xlen, ylen, zlen = np.shape(box)
@jit(nopython=True)
def dissipationDensity(x,y,z):
    xind = int(math.floor(x / pixelSize))+xlen//2
    yind = int(math.floor(y / pixelSize))+ylen//2
    zind = int(math.floor(z / pixelSize))+zlen//2
    if(0<=xind<xlen and 0<=yind<ylen and 0<=zind<zlen):
        return box[xind, yind, zind]
    return 0



@jit(nopython=True)
def dissipation_density(x,y,z):
    #return K1*timestep if ((math.sqrt(x**2+z**2)-0.3)**2+y**2<0.0025 and z < 0) or (math.sqrt((abs(x)-0.15)**2+y**2+(z-0.2)**2) < 0.07) else 0;  #Dette er gammel dissipationdensity for testing (lager en smiley)
    return dissipationDensity(x,y,z)*c*timestep