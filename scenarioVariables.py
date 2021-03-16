xLim = (-2.0, 2.0);                         #Definerer her grensene for hvor
yLim = (-2.0, 2.0);                         #fotoner tillates å eksistere
zLim = (-2.0, 2.0);                         #Det er ikke hensiktsmessig å regne på fotoner utenfor dette
timestep = 1e-11;                           #Definerer timestep
c = 3e8                                     #Definerer lyshastighet




obj1_20kev = obj1_50kev = obj1_100kev = obj2_25kev = obj2_50kev = obj2_75kev = test_array = None
#deklarerer alle objekt-array-ene

defaultElementSize = 1 / 128;             #angir default-verdien i meter til lengden på ett element i hvert objekt

loadlist = {"obj1_20kev":"object1_20keV.npy",
            "obj1_50kev":"object1_50keV.npy",
            "obj1_100kev":"object1_100keV.npy",
            "obj2_25kev":"object2_25keV.npy",
            "obj2_50kev":"object2_50keV.npy",
            "obj2_75kev":"object2_75keV.npy",
            "test_array":"test_array.npy"}
import numpy as np
for object, objectFile in loadlist.items():
    with open("data/" + objectFile, "rb") as file:
        globals()[object] = np.load(file) * 100
                        #Ganger med 100 for å konvertere fra cm^-1 til m^-1
                        #indeksering av string i globals aksesserer variabelen med string-en som navn
                        #f.eks globals()["tall"] = 2; er det samme som tall=2
                        #Vi laster da inn obj1_20kev som np.load("data/object1_20kev.npt") osv.







