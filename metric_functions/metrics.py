import numpy as np
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity


class Metrics:

    @staticmethod
    def mse(ground_truth, prediction):
        return mean_squared_error(ground_truth, prediction)

    @staticmethod
    def psnr(ground_truth, prediction):
        # data_range = np.max(ground_truth)
        data_range = 1
        psnr = peak_signal_noise_ratio(ground_truth, prediction, data_range=data_range)
        return psnr

    @staticmethod
    def ssim(ground_truth, prediction):
        return structural_similarity(ground_truth, prediction)

    @staticmethod
    def relative_error(ground_truth, prediction):
        error = np.sum(np.abs(ground_truth - prediction)) / np.sum(np.abs(ground_truth))
        return error * 100
