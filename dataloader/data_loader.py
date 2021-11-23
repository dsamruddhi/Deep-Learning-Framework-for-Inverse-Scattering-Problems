import random
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from dataloader.load_data import Load
from dataloader.process_data import Process

from utils.plot_utils import PlotUtils


class DataLoader:

    def __init__(self):

        self.input_path = Config.config["data"]["input_paths"]
        self.output_path = Config.config["data"]["output_paths"]
        self.test_size = Config.config["data"]["test_size"]

    @staticmethod
    def check_data(train_input, train_output):

        plot_cmap = PlotUtils.get_cmap()
        plot_extent = PlotUtils.get_doi_extent()

        for i in random.sample(range(0, train_input.shape[0]), 5):
            print(i)
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

            original = ax1.imshow(train_output[i, :, :], cmap=plot_cmap, extent=plot_extent)
            fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
            ax1.title.set_text("Output: ground truth")

            guess_real = ax2.imshow(train_input[i, :, :, 0], cmap=plot_cmap, extent=plot_extent)
            fig.colorbar(guess_real, ax=ax2, fraction=0.046, pad=0.04)
            ax2.title.set_text("Initial guess: real")

            guess_imag = ax3.imshow(train_input[i, :, :, 1], cmap=plot_cmap, extent=plot_extent)
            fig.colorbar(guess_imag, ax=ax3, fraction=0.046, pad=0.04)
            ax3.title.set_text("Initial guess: imaginary")

            plt.show()

    def main(self, show_data):

        X_real, X_imag = Load.get_input_data(self.input_path)
        X = np.asarray([X_real, X_imag])
        X = np.moveaxis(X, 0, -1)

        Y = Load.get_output_data(self.output_path)
        Y = np.asarray(Y)

        Process.check_data_sanctity(X, Y)

        train_input, train_output, test_input, test_output = Process.split_data(X, Y, self.test_size)

        print(f"Train input: {train_input.shape}, Train output: {train_output.shape} "
              f"Test input: {test_input.shape}, Test output: {test_output.shape}")

        if show_data:
            DataLoader.check_data(train_input, train_output)

        return train_input, train_output, test_input, test_output
