import os
import numpy as np
from scipy.io import loadmat


class Load:

    @staticmethod
    def get_files(filepath):
        files = os.listdir(filepath)
        files.sort(key=lambda x: int(x.strip(".mat")))
        return files

    @staticmethod
    def get_input_data(filepath):
        real_data = []
        imag_data = []
        files = Load.get_files(filepath)
        for file in files:
            filename = os.path.join(filepath, file)
            guess = loadmat(filename)["guess"]
            real_data.append(guess[0][0][0])
            imag_data.append(guess[0][0][1])
        return real_data, imag_data

    @staticmethod
    def get_output_data(filepath):
        scatterers = []
        files = Load.get_files(filepath)
        for file in files:
            filename = os.path.join(filepath, file)
            scatterer = loadmat(filename)["scatterer"]
            scatterers.append(np.real(scatterer))
        return scatterers
