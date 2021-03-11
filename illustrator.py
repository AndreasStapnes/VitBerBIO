import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from simulator import plane, normalize, photon, unitSphericalDistribution, crossProd, photonSource

#pl = plane(np.array((0.7, 0.0, 0.0)), normalize(np.array((-1.0, -1.0, 0.0))))
#photon.planes.append(pl)





fig, ax = plt.subplots(1,1)
#ax.set_aspect("equal")
xi = np.linspace(-0.5,0.5,256)
eta = np.linspace(-0.5,0.5,256)
xis, etas = np.meshgrid(xi, eta)
zetas = np.sin(xis)
dots = ax.pcolormesh(xis, etas ,zetas, shading="auto")


def init():
    dots.set_array(zetas.ravel())
    return dots


frames = 15
def expose(i):
    theta = i *2*np.pi/frames
    r = np.array([np.cos(theta), np.sin(theta), 0])
    normal = r
    basis = [crossProd(r, np.array([0,0,1])), np.array([0,0,1]), normal]
    pl = plane(r, normalize(normal), basis=basis, marker=False)
    photon.planes.append(pl);
    photon.updatePlanes()
    print("!", end="")

    phPl = photonSource(-r, normal, basis=basis, xiActiveArea=[-0.5,0.5], etaActiveArea=[-0.5,0.5])
    global grid
    probabilityGrid = phPl.generatePhotons(random=False, dpm=256, discretePhotons=False)
    dots.set_array(probabilityGrid.T.ravel())
    photon.planes.pop()
    del pl
    print("*", end="")
    return dots

anim = animation.FuncAnimation(fig, expose, interval=20, frames=frames, init_func=init, blit=False)

anim.save("anim2.gif", fps=10);



'''
theta = 0
r = np.array([np.cos(theta), np.sin(theta), 0])
normal = r
basis = [crossProd(r, np.array([0,0,1])), np.array([0,0,1]), normal]
pl = plane(r, normalize(normal), basis=basis, marker=False)
photon.planes.append(pl);
photon.updatePlanes()

print("!", end="")
phPl = photonSource(-r, normal, basis=basis, xiActiveArea=[-0.6,0.6], etaActiveArea=[-0.6,0.6])
global grid
probabilityGrid = phPl.generatePhotons(random=False, dpm=200, discretePhotons=False)
plt.figure(1)
plt.pcolormesh(probabilityGrid)
plt.show()

'''