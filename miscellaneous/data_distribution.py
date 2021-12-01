import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from config import Config

if __name__ == '__main__':

    filepath = Config.config["data"]["output_paths"]

    files = os.listdir(filepath)
    files.sort(key=lambda x: int(x.strip(".mat")))

    permittivities = []
    eIeR = []
    both_lower = []
    both_higher = []
    both_middle = []
    others = []
    for index, file in enumerate(files):
        filename = os.path.join(filepath, file)
        a = loadmat(filename)["scatterer_params"]
        sp1 = a[0][0]
        sp2 = a[0][1]
        p1 = round(sp1[0][0][4][0][0], 2)
        p2 = round(sp2[0][0][4][0][0], 2)
        eIeR1 = np.imag(p1)/np.sqrt(np.real(p1))
        eIeR2 = np.imag(p2)/np.sqrt(np.real(p2))
        permittivities.append(p1)
        permittivities.append(p2)
        # eIeR.append(eIeR1)
        # eIeR.append(eIeR2)
        if eIeR1 < 0.2 and eIeR2 < 0.2:
            both_lower.append(index)
        elif eIeR1 > 0.5 and eIeR2 > 0.5:
            both_higher.append(index)
        elif 0.5 > eIeR1 > 0.2 and 0.5 > eIeR2 > 0.2:
            both_middle.append(index)
        else:
            others.append(index)
            eIeR.append(eIeR1)
            eIeR.append(eIeR2)

    print(f"Mean value: {np.mean(eIeR)}")

    plt.hist(eIeR, bins=100)
