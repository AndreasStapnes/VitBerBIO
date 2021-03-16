import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors
import matplotlib as mpl
from simulator import plane, normalize, photon, crossProd, photonSource
from jitSpeedup import speedup
from scenarioVariables import obj1_20kev, obj1_50kev, obj1_100kev, obj2_25kev, obj2_50kev, obj2_75kev, test_array
from tqdm import tqdm, trange

dpm = 128
fig, ax = plt.subplots(1,1)
#ax.set_aspect("equal")
xi = np.linspace(-0.5,0.5,dpm*1+1)
eta = np.linspace(-0.5,0.5,dpm*1+1)
xis, etas = np.meshgrid(xi, eta)
zetas = np.zeros_like(xis)[:-1,:-1]
dots = ax.pcolormesh(xis, etas ,zetas, shading="flat", norm=colors.Normalize(0,1))
cb2 = fig.colorbar(dots)


def init():
    dots.set_array(zetas.ravel())
    return dots


speedup.reloadJit(obj1_20kev)
frames = 1
loopt = trange(frames)
loop = iter(loopt)
def prog(*args):
    try:
        next(loop)
    except:
        pass
def expose(i):
    theta = i *2*np.pi/frames
    r = np.array([np.cos(theta), np.sin(theta), 0])
    normal = r
    basis = [crossProd(r, np.array([0,0,1])), np.array([0,0,1]), normal]
    pl = plane(location=r, direction=normal, basis=basis, marker=False)
    photon.planes.append(pl);
    photon.updatePlanes()
    phPl = photonSource(location=-r, direction=normal, basis=basis, xiActiveArea=[-0.5,0.5], etaActiveArea=[-0.5,0.5])
    probabilityGrid = phPl.generatePhotons(random=False, dpm=dpm, discretePhotons=False)
    dots.set_array(probabilityGrid.T.ravel())
    photon.planes.pop()
    del pl
    return dots

mpl.rcParams['animation.ffmpeg_path'] = r"../../ffmpegLibFiler/bin/ffmpeg.exe";
anim = animation.FuncAnimation(fig, expose, frames=frames, save_count=frames, init_func=init, blit=False)
writer = animation.FFMpegWriter(fps=10);
anim.save("anim11.mp4", writer=writer, progress_callback=prog); prog();

#from IPython.display import Video
#Video("anim11.mp4", embed=True, html_attributes="loop autoplay")


