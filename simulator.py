# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:04:37 2021

@author: astap
"""


import numpy as np

from jitSpeedup import speedup
from scenarioVariables import xLim, yLim, zLim, c, timestep

from numba.typed import List

isInLim = lambda D1Limits, D1Position: D1Limits[0]<D1Position<D1Limits[1]       #Sjekker om en 1D posisjon D1Pisition er innenfor en 'lovlig' grense D1Limits=[min, max]
enclosed = lambda D3Position: isInLim(xLim, D3Position[0]) & isInLim(yLim, D3Position[1]) & isInLim(zLim, D3Position[2]) #Sjekker om en posisjon er innenfor de globale lovlige grensene
crossProd = lambda a, b: np.array(((a[1]*b[2]-a[2]*b[1]), (a[2]*b[0]-a[0]*b[2]), (a[0]*b[1]-a[1]*b[0]))) #cross(a,b)  Beregner kryssproduktet axb
dotProd = lambda a,b: a[0]*b[0] + a[1]*b[1] + a[2]*b[2]                         #dotProd(a,b) Beregner prikkproduktet a*b
norm = lambda a: np.sqrt(dotProd(a,a))                                          #Beregner lengden på en vektor a
orthoProjection = lambda a, b: a - b*dotProd(a,b)/(norm(b)**2)                  #Beregner den ortogonale projeksjonen av a på b
normalize = lambda a: a/norm(a);                                                #Beregner enhetsvektoren parallell med a

def generateUnitBasis(normal):                          #Finner en ortogonal basis (zeta, xi, eta) hvor zeta er den innsendte normal
    zetaHat = normalize(normal)
    iHat = np.array([1,0,0]); jHat = np.array([0,1,0])
    if(abs(dotProd(zetaHat, iHat)) < 1/np.sqrt(2)):     #Dersom zeta-aksen og iHat-aksen avviker mer fra hverandre enn 45 grader brukes den ortogonale
        xiHat = orthoProjection(iHat, zetaHat)          #projeksjonen av iHat på zeta som en arm i basisen.
    else:
        xiHat = orthoProjection(jHat, zetaHat)          #Hvis forrige if ikke holder, vet man at zeta og jHat er mer enn 45 grader fra hverandre. Da brukes jHat sin projeksjon
    #Årsaken til denne sjekkingen er for å unngå tilfeller der iHat og zeta er parallelle, i hvilket tilfelle er det umulig å danne en basis med disse to.

    xiHat /= norm(xiHat)
    etaHat = crossProd(zetaHat, xiHat)                  #Det siste elementet i basisen beregnes fra kryssproduktet av de to ortogonal basiselementene xi og zeta
    return np.array((zetaHat, xiHat, etaHat))           #Returnerer så basisen



class plane:
    def __init__(self, location, direction, **kwargs):
        '''
       Plan-klasse
       Kan opprette plan-instanser der en ønsker å absorbere og/eller markere passerende fotoner.
        :param location: posisjonen til planet. Markerer også punktet (0,0) i planets egne (todimensjonale) koordinatsystem
        :param direction: Normalvektoren til planet (kan settes til None dersom man oppgir egen basis)

        :param kwargs: Valgfrie spesifikasjoner {
            basis <- (ferdiggenerert ortogonal enhetsbasis på form (xi,eta,zeta)). xi og eta spenner planet, og zeta er planets normalvektor
            opacity <- ("transparent" eller "opaque"). Angir om planet tillater at fotoner krysser uten å bli absorbert. Opaque absorberer fotoner.
            marker <- (True eller False). Angir om planet lagrer en liste skjæringspunkter (angitt i (xi,eta)-koordinater) for hvert skjærende foton
        }
        '''
        normal = normalize(direction)
        if("basis" in kwargs):      #Bruker valgfri basis dersom dette er spesifisert. Hvis ikke, generer en basert på normalvektoren
            self.xi, self.eta, self.zeta = kwargs["basis"];
        else:
            self.zeta, self.xi, self.eta = generateUnitBasis(normal)

        self.transparent = {"opaque" : 0, "transparent" : 1}[kwargs.get("opacity", "opaque")]
        #Defaultverdi for opacity er "opaque" == absorberende plan
        #Indekserer en dictionary slik at "transparent" gir self.transparent=1 og "opaque" gir self.transparent=0

        self.marker = kwargs.get("marker", True) #Angir om planet skal markere skjærende fotoner
        self.location = location #Planets origo-posisjon
        self.markings = [] #Alle markerte fotoner


    #Det samme som operator<=, slik at plan.__le__(foton) er det samme som plan<=foton
    def __le__(self, photon):       #Sjekker om fotonet nettop skar gjennom et plan (forrige timestep)
        relativeLocationFinal = photon.location - self.location
        zetaCoordinateFinal = dotProd(relativeLocationFinal, self.zeta)             #Beregner zeta-koordinat i planets egen basis etter antatt skjæring
        relativeLocationPrevious = photon.propagate(-c*timestep) - self.location    #Går ett tidssteg tilbake
        zetaCoordinatePrevious = dotProd(relativeLocationPrevious, self.zeta)       #Beregner zeta-koordinat i planets egen basis før antatt skjæring
        return np.sign(zetaCoordinatePrevious) != np.sign(zetaCoordinateFinal)      #Sjekker om zeta-koordinatet endret fortegn
        #Returnerer sann dersom fortegnet til zeta-koordinatet endret
        #seg forrige timestep. Da vil fotonet ha skjært gjennom planet forrige timestep.
        # Nå vil altså 'plan <= foton' == 'True'

    #Det samme som operator<, slik at plan.__lt__(foton) er det samme som plan<foton
    def __lt__(self, photon): #Dette er ikke en testfunksjon, men heller en funksjon som faktisk projiserer et foton på et plan
        #,og dersom spesifisert, markerer fotonet i planets markings-liste (slik angitt i plan __init__)
        # Returverdien til __lt__ angir om fotonet blir absorbert (planet er opaque) eller ikke

        #Følgende kodeblokk brukes i grunnen ikke i våre løsningsmetoder etter som vi i stor grad baserer oss på ikke-diskrete fotoner
        #men dersom man ønsker å markere (se marker i __init__) kryssende fotoner i et definert plan, er det neste kodeblokk som fikser dette
        #Da denne kodeblokken ikke er brukt, er kommentarene mer vage.
        if(self.marker):
            relativeLocationInitial = photon.location - self.location
            planePhotonDistance = dotProd(relativeLocationInitial, self.zeta)                   # Med vektor-algebra-betraktninger finner man her
            requiredTravelDistance = planePhotonDistance/dotProd(photon.direction, self.zeta)   # hvor langt fotonet må flytte seg for å nå planet.
            inPlanePosition = photon.propagate(distance=-requiredTravelDistance);               # Her finner man posisjonen i planet dette svarer til
            relativeLocationFinal = inPlanePosition - self.location
            self.markings.append(np.array((dotProd(relativeLocationFinal, self.xi), dotProd(relativeLocationFinal, self.eta))))
            # for så å finne de tilsvarende xi og eta-koordinatene. Dette legges til i planets 'markings'
            # en kan bruke mye tekst på å beskrive alle operasjonene, men fremgangsmåten svarer til vektor-algebraens skjæring mellom akse og plan, slik behandlet i R2-matte

        return not self.transparent
        #Sannhetsverdien til plan<foton angir om fotonet absorberes. plan < foton legger til fotonets projeksjon i det aktuelle planet sine 'markings'.


class photonSource():
    def __init__(self, location, direction, **kwargs):
        '''
            foton kilde-klasse
                Skaper plan som emmiterer fotoner i retning av planets normalvektor.
                    Kan opprette fotonkilde-instanser basert på et punkt og en basis. Tilsvarende plan-klassen viser basisen (xi,eta,zeta) sin zeta-koordinat
                    til en normalvektor for et plan med origo i det nevnte punktet.
                    Instansen vil kunne emmitere fotoner fra et areal i planet utspent av xi og eta.
                    Det foton-emmiterende arealet er angitt av alle (xi,eta) slik at xi ∈ xiActiveArea og eta ∈ etaActiveArea. Alle fotoner er rettet langs zeta.

            :param location: Angir origo for det foton-emmiterende planet
            :param direction: Angir normalvektoren og foton-retningen for det emmiterende planet. (Kan settes til none dersom en basis er spesifisert i kwargs)
            :param kwargs: Valgfrie parametre {
            basis <- (ferdiggenerert ortogonal enhetsbasis på form (xi,eta,zeta)).
                        xi og eta spenner ut det emmiterende planet. zeta er planets normalvektor, og angir retningen til emmiterte fotoner
            xiActiveArea <- (2-tuppel (xi_minimum, xi_maximum)) som angir xi-koordinatene hvor det skal emmiteres fotoner. default=[-1,1]
            etaActiveArea <- (2-tuppel (eta_minimum, eta_maximum)) som angir eta-koordinatene hvor det skal emmiteres fotoner. default=[-1,1]
            }

            '''
        normal = normalize(direction)
        self.location = location
        if("basis" in kwargs): #Genererer basis som før
            self.xi, self.eta, self.zeta = kwargs["basis"]
        else:
            self.zeta, self.xi, self.eta = generateUnitBasis(normal)
        self.xiActiveArea = kwargs.get("xiActiveArea", [-1,1]) #Henter ut xiActiveArea
        self.etaActiveArea = kwargs.get("etaActiveArea", [-1,1])

    def generatePhotons(self, **kwargs):
        '''
        Genererer fotoner som angitt i __init__ og avfyrer disse i zeta-retning

        :param kwargs: Valgfrie parametre {
        random <- (True eller False) Angir om foton-emisjonene skal skje tilfeldig i planets emmiterende område (for hvert foton velges en tilfeldig posisjon)
                        Hvis False genereres ekvidistante fotoner i emmisjonsplanet utspent av xi og eta-vektorene i basisen
                        Fordelingen skjer rektangulært (som piksler på en skjerm) med nabo-foton-emisjonspunkt-avstand 1/dpm (se dpm)

            discreteFotons <- (True eller False). Kun aktuelt dersom random=False
                        Angir om de genererte fotonene er diskrete (vil enten absorberes eller ikke i hvert tidssteg) eller
                        baserer seg på en statistisk overlevelses-rate-variabel (kan kun absorberes ved plan).
                        Dersom False vil generatePhotons returnere en 2d-matrise med overlevelses-ratene
                        for hvert foton ved deres forste plan-kollisjon / ved forlatelse av lovlig område. default=True

            dpm <- (float) Kun aktuelt dersom random=False.
                        dotsPerMeter angir antall fotoner per meter, slik at 1/dpm er avstanden mellom fotonemmisjons-punkter. default=128

            amount <- (int) Kun aktuelt dersom random=True .
                        Angir antall fotoner som skal emmiteres totalt
        }

        :return: (None) dersom discreteFotons = True
                (2d numpy-array av floats) som beskriver de statistiske overlevelses-ratene for hvert foton ved plan-absorpsjon
                dersom discreteFotons=False
        '''
        random = kwargs.get("random", True)                     #Henter random fra valgfrie variabler
        xiLen = self.xiActiveArea[1]-self.xiActiveArea[0]       #Henter xi-lengde for emmisjonsintervall
        xiNought = self.xiActiveArea[0]                         #Henter minste tillatte xi-koordinat i emmisjonsintervall
        etaLen = self.etaActiveArea[1]-self.etaActiveArea[0]    #Henter eta-lengde for emmisjonsintervall
        etaNought = self.etaActiveArea[0]                       #Henter minste tillatte eta-koordinat i emmisjonsintervall

        if (random):                                                    #Dersom tilfeldig fordeling
            amount = kwargs.get("amount")                               #Antall fotoner må være spesifisert dersom tilfeldig fordeling
            for i in range(amount): #I random må fotonene være diskrete (de kan enten fortsette med survivalRate 1 eller umiddelbart opphøre med survivalRate 0 i hver tidssteg)
                xiRand = np.random.random()*xiLen + xiNought            #velger et tilfeldig koordinat i det utspente photonGenerator-planet
                etaRand = np.random.random()*etaLen + etaNought         #^^ bare for eta

                pos = self.location + self.xi * xiRand + self.eta * etaRand #Danner det tilfeldige punktet
                ph = photon(pos, self.zeta) #Skaper fotonet i dette punktet, rettet normalt med planet
                ph.jitPrimer() #Avfyrer fotonet
                return None #Ingenting aktuelt å returnere

        else: #Dersom jevn fordeling av fotoner
            discrete = kwargs.get("discretePhotons", True)
            dpm = kwargs.get("dpm", 128)
            def approxDiscretize(start, end, dpm):  #Diskretiserer et intervall slik du ville med piksler, dvs, linspace bare der det er et halvt step fra hver side til første/siste punkt
                length = end - start                # Årsaken til dette er at man ønsker hver hver 'pixel' skal være representert av et foton gjennom sitt senter.
                totPts = round(dpm * length)        # Dvs for xi-intervall [0,2] og dpm=1 vil det genereres fotoner i 0.5 & 1.5, som farger piksler med utstrekning [0,1] & [1,2].
                stepLen = length/totPts
                return np.linspace(start+stepLen/2, end-stepLen/2, totPts)  #Koden her er relativt selvforklarende

                         #kort om unpack: *self.xiActiveArea === self.xiActiveArea[0], self.xiActiveArea[1]
            xiDiscretePoints = approxDiscretize(*self.xiActiveArea, dpm)    #Finner xi foton-avfyrings-koordinater
            etaDiscretePoints = approxDiscretize(*self.etaActiveArea, dpm)  #Finner eta foton-avfyrings-koordinater
            xiAmt = len(xiDiscretePoints); etaAmt = len(etaDiscretePoints)  #Henter antall koordinater i xi og eta-retning
            probabilisticSurvivalRates = np.zeros((xiAmt, etaAmt))          #Skaper en numpy-matrise en ønsker å lagre alle overlevelses-ratene i
            for xiIndex in range(xiAmt):
                for etaIndex in range(etaAmt):
                    xiCrd = xiDiscretePoints[xiIndex]                           #Finner xi-koordinat for et gitt foton
                    etaCrd = etaDiscretePoints[etaIndex]                        #Finner eta-koordinat fot et gitt foton
                    pos = self.location + self.xi * xiCrd + self.eta * etaCrd   #Finner (x,y,z) posisjonen til fotonet med xi-eta-indexer xiIndex & etaIndex.
                    ph = photon(pos, self.zeta, discrete)                       #Skaper fotonet
                    throwawayLocation, probSurvivalRate = ph.jitPrimer()        #Avfyrer fotonet og henter ut probSurvivalRate.
                                                                                #throwawayLocation må være tilstede for å hente verdier, men brukes ikke.

                    probabilisticSurvivalRates[xiIndex,etaIndex] = probSurvivalRate #Lagrer overlevelsesraten

            return probabilisticSurvivalRates                                   #Returnerer probsurvrates






class photon:
    planes = [];                        #Alle aktuelle plan hvilket fotonet kan krysse og interagere med
    planesCoordinates = List([])        #List (fra numba) er en jit-optimalisert liste. plan-koordinatene lagres her ved kjøring av updatePlanes for bruk i jitPrimer (i photon)
    planesDirections = List([])         #^^ Bare for plan-normalvektorer

    @classmethod
    def updatePlanes(cls):              #Klassefunksjon man kjører når man ønsker å forberede planene for de jit-akselererte funksjonene
        cls.planesCoordinates = List([List(plane.location) for plane in photon.planes])
        cls.planesDirections = List([List(plane.zeta) for plane in photon.planes])
                                        #Fyller photon-klassens planesCoordinates og planesDirections med en liste av 3-Lister (x,y,z)
                                        #bestående av hhv. planenes origo-koordinater og normalvektorer


    def __init__(self, location, direction, discreteHits=True):
        '''
        foton-klasse
        Instansiering skaper et foton i posisjon location, rettet langs direction.
        :param location: (3-tuppel (x,y,z)) Angir startposisjonen til fotonet
        :param direction: (3-tuppel (x,y,z)) Angir retningen til fotonet
        :param discreteHits: (True eller False) Angir om fotonet er diskret (vil enten overleve eller opphøre ved hvert timestep)
                eller probabilistisk (har en intern variabel som beskriver sannsynligheten fotonet har for å nå et absorberende plan / utenfor lovlig grense)
        '''
        self.location = location                    #Henter foton-posisjon
        self.direction = normalize(direction)       #Henter foton-retning
        self.discrete = discreteHits                #Angir hvorvidt fotonet er diskret

    def propagate(self, distance=c*timestep):                   #Funksjon for å forflytte fotonet en gitt avstand. Dersom ingen avstand
        return self.location + self.direction * distance        #spesifiseres, forflyttes det avstand tilsvarende lyshastighet i timestep tid

    def jitPrimer(self):                                        #Funksjon som forbereder fotonet for transmisjon, og deretter 'avfyrer' det.
        hitPlane, position, probabilisticSurvival = speedup.jitTilHit(photon.planesCoordinates, photon.planesDirections,
                          self.location, self.direction, self.discrete)
                                                                #Henter bool hitPlane, (x,y,z) position og float probabilisticSurvival fra jitTilHit i speedup-klassen
                                                                #Merk at jitTilHit kun kan returnere dersom fotonet
                                                        #(1): treffer et plan       -> hitPlane = True & Det finnes et plan som fotonet krysset i forrige timestep
                                                        #(2): kommer utenfor        -> hitPlane = False
                                                        #(3): er diskret og blir absorbert i mediet -> hitPlane = False

        if not hitPlane:                                #Her håndterer man (2) & (3)
            del self                                    #Sletter fotonet
            return position, 0                          #returnerer posisjon og survival-Probability=0

        self.location = position                        #Sett fotones posisjon til returverdien av jitTilHit
        for plane in photon.planes:                         #Sjekker gjennom alle plan for å finne det fotonet traff
            if(plane <= self):                              #Sjekker om fotonet traff (se plane.__lt__)
                if(plane < self):                           #Marker treff-punktet i planet fotonet traff. plane<self returnerer sann dersom plane sin opacity er opaque
                    del self                                #Dersom plan opaque, slett fotonet (det blir absorbert)
                else:                                       #Hvis ikke, avfyr fotonet igjen (rekursivt) til det omsider havner utenfor lovlig område eller treffer opaque plan
                    position, probabilisticSurvival = self.jitPrimer()
                return position, probabilisticSurvival

        #Funksjonen burde aldri komme her. Dersom det hadde vært tilfellet, hadde jit påstått det finnes et plan som fotonet skjærer,
        #mens python påstår det motsatte
        raise Exception("Disagreement of hit between jit-nopython and python")



''' Gammel kode for å skape uniformt fordelte retninger fra et punkt
def unitSphericalDistribution():
    latDist = np.random.random()*2 - 1
    sgn = np.sign(latDist); latDist *= sgn
    latitude = np.pi*(np.sqrt(latDist)/2 if sgn>0 else 1-np.sqrt(latDist)/2)
    longitude = 2*np.pi*np.random.random()
    x,y,z = np.sin(latitude)*np.cos(longitude), np.sin(latitude)*np.sin(longitude), np.cos(latitude)
    return x,y,z
    '''