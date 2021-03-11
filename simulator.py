# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:04:37 2021

@author: astap
"""
timestep = 1e-11;
c = 3e8


import numpy as np

import jitSpeedup as speedUp
from numba import jit
from scenarioVariables import xLim, yLim, zLim, c, timestep

import numba
from numba.typed import List

isInLim = lambda D1Limits, D1Position: D1Limits[0]<D1Position<D1Limits[1]
enclosed = lambda D3Position: isInLim(xLim, D3Position[0]) & isInLim(yLim, D3Position[1]) & isInLim(zLim, D3Position[2])
crossProd = lambda a, b: np.array(((a[1]*b[2]-a[2]*b[1]), (a[2]*b[0]-a[0]*b[2]), (a[0]*b[1]-a[1]*b[0])))
dotProd = lambda a,b: a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
norm = lambda a: np.sqrt(dotProd(a,a))
orthoProjection = lambda a, b: a - b*dotProd(a,b)/(norm(b)**2)
normalize = lambda a: a/norm(a);

def generateUnitBasis(normal):
    zetaHat = normalize(normal)
    iHat = np.array([1,0,0]); jHat = np.array([0,1,0])
    if(dotProd(zetaHat, iHat) > 1/np.sqrt(2)):
        xiHat = orthoProjection(iHat, zetaHat)
    else:
        xiHat = orthoProjection(jHat, zetaHat)
    xiHat /= norm(xiHat)
    etaHat = crossProd(zetaHat, xiHat)
    return np.array((zetaHat, xiHat, etaHat))


def unitSphericalDistribution():
    latDist = np.random.random()*2 - 1
    sgn = np.sign(latDist); latDist *= sgn
    latitude = np.pi*(np.sqrt(latDist)/2 if sgn>0 else 1-np.sqrt(latDist)/2)
    longitude = 2*np.pi*np.random.random()
    x,y,z = np.sin(latitude)*np.cos(longitude), np.sin(latitude)*np.sin(longitude), np.cos(latitude)
    return x,y,z

class plane:
    def __init__(self, location, direction, **kwargs):
        if("basis" in kwargs):
            self.xi, self.eta, self.zeta = kwargs["basis"];
        else:
            self.zeta, self.xi, self.eta = generateUnitBasis(direction)

        self.transparent = {"opaque" : 0, "transparent" : 1}[kwargs.get("opacity", "opaque")] #Defaultverdi opaque == absorberende plan
        self.marker = kwargs.get("marker", True)
        self.location = location
        self.markings = []
        #skaper en basis (xi, eta, zeta) for planet med 'direction'(=zeta) som enhetsvektor
        #i retning av planets normal-akse. Planet defineres
        #til å krysse location, og baserer sine koordinater ut fra dette


    def __le__(self, photon): #Sjekker om fotonet nettop skar gjennom et plan
        relativeLocationFinal = photon.location - self.location
        zetaCoordinateFinal = dotProd(relativeLocationFinal, self.zeta)
        relativeLocationPrevious = photon.nextStep(-c*timestep) - self.location
        zetaCoordinatePrevious = dotProd(relativeLocationPrevious, self.zeta)
        return np.sign(zetaCoordinatePrevious) != np.sign(zetaCoordinateFinal)
        #Returnerer sann dersom fortegnet til zeta-koordinatet endret
        #seg forrige timestep. Nå vil altså 'plan <= foton' == 'True'

    def __lt__(self, photon):
        if(self.marker):
            relativeLocationInitial = photon.location - self.location
            planePhotonDistance = dotProd(relativeLocationInitial, self.zeta)
            requiredTravelDistance = planePhotonDistance/dotProd(photon.direction, self.zeta)
            inPlanePosition = photon.nextStep(distance=-requiredTravelDistance);
            relativeLocationFinal = inPlanePosition - self.location
            self.markings.append(np.array((dotProd(relativeLocationFinal, self.xi), dotProd(relativeLocationFinal, self.eta))))
        if(self.transparent):
            return False
        return True
    #Sannhetsverdien til plan<foton angir om fotonet absorberes. plan < foton legger til fotonets projeksjon i det aktuelle planet sine 'markings'.


class photonSource():
    def __init__(self, position, normal, **kwargs):
        self.position = position
        if("basis" in kwargs):
            self.xi, self.eta, self.zeta = kwargs["basis"]
        else:
            self.zeta, self.xi, self.eta = generateUnitBasis(normal)
        self.xiActiveArea = kwargs.get("xiActiveArea", [-1,1])
        self.etaActiveArea = kwargs.get("etaActiveArea", [-1,1])

    def generatePhotons(self, amount=1, **kwargs):
        random = kwargs.get("random", True)
        xiLen = self.xiActiveArea[1]-self.xiActiveArea[0]
        xiNought = self.xiActiveArea[0]
        etaLen = self.etaActiveArea[1]-self.etaActiveArea[0]
        etaNought = self.etaActiveArea[0]
        if (random):
            for i in range(amount): #I random må fotonene være diskrete
                xiRand = np.random.random()*xiLen + xiNought
                etaRand = np.random.random()*etaLen + etaNought
                pos = self.position + self.xi * xiRand + self.eta * etaRand
                ph = photon(pos, self.zeta)
                ph.jitPrimer()
                return None #Ingenting aktuelt å returnere
        else:
            discrete = kwargs.get("discretePhotons", True)
            dpm = kwargs.get("dpm", 200)
            def approxDiscretize(start, end, dpm): #Diskretiserer et intervall slik du ville med piksler
                length = end-start
                totPts = round(dpm*length)
                stepLen = length/totPts
                return np.linspace(start+stepLen/2, end-stepLen/2, totPts)
            xiDiscretePoints = approxDiscretize(*self.xiActiveArea, dpm)
            etaDiscretePoints = approxDiscretize(*self.etaActiveArea, dpm)
            xiAmt = len(xiDiscretePoints); etaAmt = len(etaDiscretePoints)
            probabilisticSurvivalRates = np.zeros((xiAmt, etaAmt))
            for xiIndex in range(xiAmt):
                for etaIndex in range(etaAmt):
                    xiPt = xiDiscretePoints[xiIndex]
                    etaPt = etaDiscretePoints[etaIndex]
                    pos = self.position + self.xi * xiPt + self.eta * etaPt
                    ph = photon(pos, self.zeta, discrete)
                    throwawayPosition, probSurvivalRate = ph.jitPrimer()
                    probabilisticSurvivalRates[xiIndex,etaIndex] = probSurvivalRate

            return probabilisticSurvivalRates
            #Returnerer probsurvrates, hvilket i grunnen kun er interresante dersom det er ikkediskrete fotoner involvert





class photon:
    planes = []; #Alle aktuelle plan hvilket fotonet kan krysse
    planesCoordinates = List([])
    planesDirections = List([])

    @classmethod
    def updatePlanes(cls):
        cls.planesCoordinates = List([List(plane.location) for plane in photon.planes])
        cls.planesDirections = List([List(plane.zeta) for plane in photon.planes])

    def __init__(self, location, direction, discreteHits=True):
        self.location = location
        self.direction = normalize(direction)
        self.discrete = discreteHits

    def nextStep(self, distance=c*timestep):
        return self.location + self.direction * distance

    def jitPrimer(self):
        hitPlane, position, probabilisticSurvival = speedUp.jitTilHit(photon.planesCoordinates, photon.planesDirections,
                          self.location, self.direction, self.discrete)

        if not hitPlane:
            del self
            return position, probabilisticSurvival
        self.location = position


        for plane in photon.planes:
            if(plane <= self):
                if(plane < self): #Marker punktet i planet
                    del self #Dersom absorbert, slett
                else:
                    position = self.jitPrimer()
                return position, probabilisticSurvival

        print("failed")
        raise Exception("Disagreement of hit between jit-nopython and python")



'''
pl = plane(np.array((0.7,0.0,0.0)), normalize(np.array((-1.0,-1.0,0.0))))
photon.planes.append(pl)

poses = []
nonvis_pts = 200
for j in range(nonvis_pts):
    for i in range(1000):
        dir = unitSphericalDistribution()
        pos = np.array((-.9,-.9,.0))
        ph = photon(pos.copy(), dir)
        current_poses = [pos, ph.jitPrimer()]
        poses.append(np.array(current_poses))
    print("*", end='')

poses = poses[::nonvis_pts]






import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(0)
ax = fig.add_subplot(111,projection='3d')
for path in poses:
    ax.plot(*path.T, ':', c="k")

scatterpos = []
for mrk in pl.markings:
    pos = pl.location + pl.xi*mrk[0] + pl.eta*mrk[1]
    scatterpos.append(pos)
scatterpos = np.array(scatterpos)
ax.scatter(*scatterpos.T,c='r',marker='o', s=0.1)
plt.show()
fig.savefig("illustrasjon.pdf")

fig2 = plt.figure(1)
plt.plot(*np.array(pl.markings).T, 'kx', markersize=2)

plt.show()'''