import os
import numpy as np
from scipy.io import loadmat

from config import Config


class Load:

    num_samples = Config.config["data"]['num_samples']

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
        num_files = Load.num_samples if Load.num_samples <= len(files) else len(files)
        for file in files[:num_files]:
            filename = os.path.join(filepath, file)
            guess = loadmat(filename)["guess"]
            real_data.append(guess[0][0][0])
            imag_data.append(guess[0][0][1])
        return real_data, imag_data

    @staticmethod
    def get_output_data(filepath):
        scatterers = []
        files = Load.get_files(filepath)
        num_files = Load.num_samples if Load.num_samples <= len(files) else len(files)
        for file in files[:num_files]:
            filename = os.path.join(filepath, file)
            scatterer = loadmat(filename)["scatterer"]
            scatterers.append(np.real(scatterer))
        return scatterers
