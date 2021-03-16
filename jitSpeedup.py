
from numba import jit
import math
import numpy as np
import random
from scenarioVariables import xLim, yLim, zLim, c, timestep, pixelSize

class speedup: #En klasse kun for å innkapsle alle jit-funksjoner
    #Den inneholder to funksjoner (jitTilHit og dissipation) hvilket opprinnelig er definert som None.


    jitTilHit = None  #Dette er hvor den 'ferdigkompilerte' jit-til-hit-funksjonen lagres
    dissipation = None #Tilsvarende er dette hvor den kompilerte dissipation-funksjonen lagres
    #Begge disse to vil aksesseres fra utenfor klassen

    # Dette blir cls.jitTilHit i klassefunksjonene definert i speedup. cls vider kun til klassen

    @classmethod
    def reloadDissipation(cls, box):
        # Når man kjører reloadDissipation med et spesifisert system (f.eks obj1_20kev) som box, vil dissipation
        #variabelen i speedup bli en jit-akselerert funksjon som kan kalles videre, for å finne dissiperingsverdien i et gitt punkt.
        #Alt denne funksjonen (reloadDIssipation) gjør er å re-kjøre kompilasjonen (som man ville gjort når man laster inn et nytt system)
        #Denne funksjonen er ikke brukt andre steder enn i reloadJit.

        global pixelSize, c, timestep #Henter aktuelle globale variabler
        #pixelSize er lengden på en 'pixel' eller 'index' i systemet box.
        #Dersom box er 128x128x13 og pixelSize = 1/128, vil boksen ha dimensjoner 1mx1mx(13/128)m

        xlen, ylen, zlen = np.shape(box) #Henter ut dimensjonene til boksen

        @jit(nopython=True)
        def attenuation(x, y, z):
            xind = int(math.floor(x / pixelSize)) + xlen // 2   #Beregner her en (integer) x,y og z-index for et (float) x,y,z-koordinat
            yind = int(math.floor(y / pixelSize)) + ylen // 2   #Dette gjøres slik at koordinatene (0,0,0) ligger får index i system-boksens midtpunkt (xlen//2, ylen//2, zlen//2)
            zind = int(math.floor(z / pixelSize)) + zlen // 2
            if (0 <= xind < xlen and 0 <= yind < ylen and 0 <= zind < zlen):
                return box[xind, yind, zind]
            return 0

        @jit(nopython=True)
        def dissipation(x, y, z):                       #Vi omtaler dissipasjon som sannsynlighet for å opphøre i ett enkelt tidssteg.
            return attenuation(x, y, z) * c * timestep  #Dissipasjonen er dermed gitt med attenuasjonsverdien ganger fotonets forflytning
        cls.dissipation = dissipation
        #cls refererer til class, så her setter vi dissipation-variabelen i speedup til å være dissipation-funksjonen
        #,hvilket for et foton i koordinater (x,y,z) angir sannsynligheten for å opphøre ila. neste tidssteg.


    @classmethod
    def reloadJit(cls, box):
        #Denne funkjsonen kjører man når man vil laste inn et nytt system (en ny numpy-matrise av attenuasjonskoeffisienter)
        #Dersom den aktuelle numpy-matrisen sendes inn som box, vil jitTilHit i speedup redefineres til å simulere ett
        #foton som forflyttes i det nevnte systemet. (Fotonet henter da dissipasjonsverdier fra det nye systemet)

        cls.reloadDissipation(box)      #Definerer først de nye dissipasjonsverdiene
        dissipation = cls.dissipation   #Og henter disse inn i en lokal dissipation-variabel
        global c, xLim, yLim, zLim, timestep #Henter ut scenario-verdier fra globalt skop

        @jit(nopython=True) #Redefinerer jitTilHit, som baserer seg på den nye dissipasjonsvariabelen
        def jitTilHit(planesCoordinates, planesDirection, initialCoordinates, initialDirection, discrete=True):
            '''
            :param planesCoordinates: En liste (ett element for hvert plan) av 3-tupler som består av (x,y,z)-koordinater som det aktuelle plan krysser
            :param planesDirection: En liste (ett element for hver plan) av 3-typler som består av (x,y,z)-koordinater for enhets-normalvektor til det aktuelle planet
            :param initialCoordinates: Opprinnelig koordinat (x,y,z) for et foton før det 'avfyres'
            :param initialDirection: Retning til et foton når det 'avfyres'
            :param discrete: discrete=true evaluerer for hvert tidssteg om fotonet vil opphøre, og stopper kjøring når dette er tilfellet.
            discrete=false fjerner ikke fotonet før det treffer et plan, men inneholder en probabilisticSurvivalValue som angir sannsynlighet for at fotonet når planet
            :return: returnerer verdier (bool, position, probabilisticSurvivalValue) der bool angir om fotonet traff et plan, posisjon angir fotonets site posisjon, og
            probabilisticSurvivalValue angir fotones sannsynlighet for å nå dit det gjorte (kun aktuelt for discrete=False)
            '''
            #jitTilHit skyter ut et foton fra sin startposisjon til det enten treffer et plan (og blir absorbert) eller til det havner utenfor
            #systemets grenser (angitt av xLim, yLim og zLim)

            global c, xLim, yLim, zLim, timestep #Henter globale verdier
            probabilisticSurvivalValue = 1.0 #Sannsynligheten for at fotonet når sin startposisjon er 1
            position = initialCoordinates; #Setter posisjon og retningsvariabler
            direction = initialDirection;


            def dotProd(a, b):                      #Definerer prikkprodukt mellom to vektorer
                return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

            def D3Difference(a, b):                 #Definerer differanse mellom to vektorer
                return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

            def enclosed(position):                 #Sjekkefunksjon for å verifisere at fotonet er innenfor 'lovlige grenser' angitt i globals
                isEnclosed = xLim[0] <= position[0] <= xLim[1]
                isEnclosed &= yLim[0] <= position[1] <= yLim[1]
                isEnclosed &= zLim[0] <= position[2] <= zLim[1]
                return isEnclosed

            initialRelativeZSigns = [dotProd(D3Difference(position, planePosition), planeDirection) > 0 for
                                     planePosition, planeDirection in zip(planesCoordinates, planesDirection)]
            #Lager en liste (ett element for hver plan) som beskriver om et punkt er 'over' eller 'under' et plan
            #Dvs, om punktet ligger på samme side av planet som planets normalvektor peker.
            #True angir at punkt og normalvektor er på samme side. False betyr motsatt
            #Dersom dette endrer seg i løpet av et tids-steg for et foton vet man at fotonet krysset planet

            planesAmt = len(planesCoordinates) #Angir antall plan man handler med

            def planePhotonCollision(index):
                finalRelativePos = D3Difference(position, planesCoordinates[index])
                finalZSign = dotProd(finalRelativePos, planesDirection[index]) > 0
                return initialRelativeZSigns[index] != finalZSign
            #Denne funksjonen tar inn en index (plan-index i planesCoordinates) og beregner så hvilket side av planet fotonet
            # ligger på. Dersom dette endret seg siden samme beregning ble gjort ved starten (initialRelativeZSigns) vet man
            # at fotonet krysset planet med index 'index'



            while(True):                                                                #Avfyrer her fotonet
                dissipation_value = dissipation(position[0], position[1], position[2])  #Beregner dissipasjonsverdi i fotonets posisjon
                if(discrete):                                                       #Dersom diskret, sjekker man om random.random() er under dissipasjonsverdien,
                                                                                    # i hvilket tilfelle anser man fotonet som absorbert
                    if random.random() < dissipation_value:
                        return (False, position, 0.0)                               #Ved absorpsjon avsluttes jitTilHit
                else:                           #Dersom ikke diskret, endrer man vare probabilisticSurvivalValue til seg selv ganget sannsynligheten for å overleve neste tidssteg
                    probabilisticSurvivalValue *= (1-dissipation_value)

                for index in range(planesAmt):                                      #Sjekker for hvert plan om fotonet nettop skar gjennom det
                    if(planePhotonCollision(index)):
                        return (True, position, probabilisticSurvivalValue)         #Dersom fotonet skjærer gjennom et plan, avsluttes jitTilHit

                if not enclosed(position):                                          #Dersom fotonet forlater de lovlige grensene, avsluttes jitTilHit
                    return (False, position, probabilisticSurvivalValue)

                #Returverdi-betydning er angitt i jitTilHit sin beskrivelse

                position[0] += direction[0] * c * timestep
                position[1] += direction[1] * c * timestep
                position[2] += direction[2] * c * timestep
                #Dersom fotonet enda ikke har skjært gjennom noen plan eller forflyttet seg ut i ulovlige områder, avanserer fotonet

        cls.jitTilHit = jitTilHit
        #den nylig kompilerte jitTilHit-funksjonen lagres nå i speedup-klassen.