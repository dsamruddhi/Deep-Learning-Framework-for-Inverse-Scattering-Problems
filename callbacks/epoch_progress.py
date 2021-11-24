import os
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from config import Config
from utils.plot_utils import PlotUtils


class EpochProgressCallback(Callback):

    def __init__(self, file_writer):
        super().__init__()
        self.file_writer = file_writer

    def on_epoch_end(self, epoch, logs=None):

        data_path = Config.config["data"]["standard_path"]
        list_dir = os.listdir(data_path)
        list_dir.sort(key=lambda x: int(x.strip("test")))

        for index, dir in enumerate(list_dir):
            real_rec = loadmat(os.path.join(data_path, dir, "real_rec.mat"))["real_rec"]
            imag_rec = loadmat(os.path.join(data_path, dir, "imag_rec.mat"))["imag_rec"]

            params = loadmat(os.path.join(data_path, dir, "params.mat"))["params"]

            test_input = np.asarray([real_rec, imag_rec])
            test_input = np.moveaxis(test_input, 0, -1)
            test_input = test_input[np.newaxis, ...]

            y_pred = self.model.predict(test_input)
            y_pred = y_pred[0, :, :, :]

            title = f"Test {index}: {round(params[0][0][0][0][4][0][0], 2)}"
            plot_buf = PlotUtils.plot_output(y_pred)
            image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            with self.file_writer.as_default():
                tf.summary.image(title, image, step=epoch)
            del image
