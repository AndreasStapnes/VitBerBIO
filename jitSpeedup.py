
from numba import jit
import random
from numba.typed import List

from scenarioVariables import c, xLim, yLim, zLim, timestep, dissipation_density




@jit(nopython=True)
def jitTilHit(planesCoordinates, planesDirection, initialCoordinates, initialDirection, discrete=True):
    global c, xLim, yLim, zLim, timestep, dissipation_density
    probabilisticSurvivalValue = 1.0
    position = initialCoordinates;
    direction = initialDirection;


    def dotProd(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def D3Difference(a, b):
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    def enclosed(position):
        isEnclosed = xLim[0] <= position[0] <= xLim[1]
        isEnclosed &= yLim[0] <= position[1] <= yLim[1]
        isEnclosed &= zLim[0] <= position[2] <= zLim[1]
        return isEnclosed

    initialRelativeZSigns = [dotProd(D3Difference(position, planePosition), planeDirection) > 0 for
                             planePosition, planeDirection in zip(planesCoordinates, planesDirection)]
    planesAmt = len(planesCoordinates)

    def planePhotonCollision(index):
        finalRelativePos = D3Difference(position, planesCoordinates[index])
        finalZSign = dotProd(finalRelativePos, planesDirection[index]) > 0
        return initialRelativeZSigns[index] != finalZSign




    while(True):

        dissipation_value = dissipation_density(position[0], position[1], position[2])
        if(discrete):
            if random.random() < dissipation_value:
                return (False, position, 0.0)
        else:
            probabilisticSurvivalValue *= (1-dissipation_value)

        for index in range(planesAmt):
            if(planePhotonCollision(index)):
                return (True, position, probabilisticSurvivalValue)

        if not enclosed(position):
            return (False, position, probabilisticSurvivalValue)

        position[0] += direction[0] * c * timestep
        position[1] += direction[1] * c * timestep
        position[2] += direction[2] * c * timestep

