import os
import pickle
import datetime
import numpy as np
import tensorflow as tf
from scipy.io import loadmat

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, UpSampling2D, BatchNormalization,\
                                    Activation, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


from config import Config
from model.base_model import BaseModel
from dataloader.data_loader import DataLoader
from utils.plot_utils import PlotUtils

from loss_functions.reverse_huber import ReverseHuber
from callbacks.epoch_progress import EpochProgressCallback

from metric_functions.metrics import Metrics


class UNet(BaseModel):

    def __init__(self):

        super().__init__()

        # Data and its attributes
        self.train_input = None
        self.train_output = None
        self.test_input = None
        self.test_output = None
        self.pred_output = None
        self.data_generator = ImageDataGenerator(validation_split=Config.config["train"]["validation_split"])

        # Model and its attributes
        self.model_path = Config.config["model"]["model_path"]
        self.experiment_name = Config.config["model"]["experiment_name"]
        self.model = None
        self.checkpoint_callback = None
        self.tensorboard_callback = None
        self.epoch_callback = None

        # Training
        self.epochs = Config.config["train"]["epochs"]
        self.train_batch_size = Config.config["train"]["train_batch_size"]
        self.val_batch_size = Config.config["train"]["val_batch_size"]

        # Test
        self.test_mse = None
        self.test_psnr = None

        # Logging
        self.file_writer = None

    def load_data(self, show_data=False):
        self.train_input, self.train_output, self.test_input, self.test_output = DataLoader().main(show_data)

    def build(self):

        def _one_cnn_layer(input, num_filters, kernel_size, padding):
            layer = Conv2D(num_filters, kernel_size=kernel_size, padding=padding)(input)
            layer = BatchNormalization()(layer)
            layer = Activation("relu")(layer)
            return layer

        def _create_model():
            input_layer = Input(shape=(50, 50, 2))

            """ Down-sampling """

            conv1 = _one_cnn_layer(input_layer, 64, 3, "VALID")
            conv1 = _one_cnn_layer(conv1, 64, 3, "SAME")
            conv1 = _one_cnn_layer(conv1, 64, 3, "SAME")
            pool1 = MaxPooling2D(pool_size=2)(conv1)  # 24 x 24

            conv2 = _one_cnn_layer(pool1, 128, 3, "SAME")
            conv2 = _one_cnn_layer(conv2, 128, 3, "SAME")
            conv2 = _one_cnn_layer(conv2, 128, 3, "SAME")
            pool2 = MaxPooling2D(pool_size=2)(conv2)  # 12 x 12

            conv3 = _one_cnn_layer(pool2, 256, 3, "SAME")
            conv3 = _one_cnn_layer(conv3, 256, 3, "SAME")
            conv3 = _one_cnn_layer(conv3, 256, 3, "SAME")
            pool3 = MaxPooling2D(pool_size=2)(conv3)  # 6 x 6

            conv4 = _one_cnn_layer(pool3, 512, 3, "SAME")
            conv4 = _one_cnn_layer(conv4, 512, 3, "SAME")
            conv4 = _one_cnn_layer(conv4, 512, 3, "SAME")

            """ Up-sampling """

            up5 = (UpSampling2D(size=(2, 2))(conv4))  # 12 x 12
            merge5 = Concatenate()([conv3, up5])

            conv5 = _one_cnn_layer(merge5, 256, 2, "SAME")
            conv5 = _one_cnn_layer(conv5, 256, 3, "SAME")
            conv5 = _one_cnn_layer(conv5, 256, 3, "SAME")

            up6 = (UpSampling2D(size=(2, 2))(conv5))  # 24 x 24
            merge6 = Concatenate()([conv2, up6])

            conv6 = _one_cnn_layer(merge6, 128, 2, "SAME")
            conv6 = _one_cnn_layer(conv6, 128, 3, "SAME")
            conv6 = _one_cnn_layer(conv6, 128, 3, "SAME")

            up7 = (UpSampling2D(size=(2, 2))(conv6))  # 48 x 48
            merge7 = Concatenate()([conv1, up7])

            conv7 = _one_cnn_layer(merge7, 64, 2, "SAME")
            conv7 = _one_cnn_layer(conv7, 64, 3, "SAME")
            conv7 = _one_cnn_layer(conv7, 64, 3, "SAME")

            conv8 = Conv2DTranspose(1, kernel_size=3, padding="VALID")(conv7)  # 50 x 50
            merge9 = Concatenate()([input_layer, conv8])

            """ Final layer """
            conv10 = Conv2D(1, kernel_size=1)(merge9)
            conv10 = Activation("relu")(conv10)

            model = Model(inputs=input_layer, outputs=conv10)
            return model

        self.model = _create_model()
        lr_schedule = ExponentialDecay(Config.config["train"]["initial_learning_rate"],
                                       decay_steps=Config.config["train"]["decay_steps"],
                                       decay_rate=Config.config["train"]["decay_rate"])
        self.model.compile(optimizer=Adam(learning_rate=lr_schedule),
                           loss=ReverseHuber(slope=3, delta=0.1),
                           metrics=Config.config["train"]["metrics"])

    def log(self):
        with open(os.path.join(self.model_path, self.experiment_name, "config.pkl"), "wb") as f:
            pickle.dump(Config.config, f)

    def callbacks(self):

        # Checkpoint callback
        checkpoint_path = os.path.join(self.model_path,
                                       self.experiment_name,
                                       "checkpoints",
                                       "weights-{epoch:02d}-{loss:.4f}-{val_loss:.4f}")
        self.checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                                   monitor='val_loss',
                                                   verbose=0,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   save_freq="epoch",
                                                   period=1)

        # Tensorboard callback
        log_dir = os.path.join(self.model_path,
                               "logs",
                               self.experiment_name,
                               datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.file_writer.set_as_default()
        self.tensorboard_callback = TensorBoard(log_dir=log_dir,
                                                histogram_freq=1)

        # Epoch progress callback
        self.epoch_callback = EpochProgressCallback(self.file_writer)

    def train(self):
        model_history = self.model.fit(self.data_generator.flow(self.train_input,
                                                                self.train_output,
                                                                batch_size=self.train_batch_size,
                                                                subset='training'),
                                       validation_data=self.data_generator.flow(self.train_input,
                                                                                self.train_output,
                                                                                batch_size=self.val_batch_size,
                                                                                subset='validation'),
                                       steps_per_epoch=len(self.train_input) / (self.train_batch_size + self.val_batch_size),
                                       shuffle=True,
                                       epochs=self.epochs,
                                       callbacks=[self.checkpoint_callback,
                                                  self.tensorboard_callback,
                                                  self.epoch_callback])

        with open(os.path.join(self.model_path, self.experiment_name, "model_history.pkl"), "wb") as f:
            pickle.dump(model_history.history, f)

    def get_best_model(self):
        """ Pick model with lowest validation error after training is done """
        model_dir = os.path.join(self.model_path, self.experiment_name, "checkpoints")
        best_model_path = max([os.path.join(model_dir, d) for d in os.listdir(model_dir)], key=os.path.getmtime)
        # TODO: load_model throws error when it does not find custom loss functions during inference.
        #  Adding custom_objects={"ReverseHuber": ReverseHuber(slope=1, delta=0.1).call} throws error since call method
        #  does not take reduction that is passed to it. Setting compile=False for now to avoid this issue since
        #  inference is still possible. FIX this later if needed. Problem only with custom loss functions, not needed
        #  with predefined ones.
        self.model = load_model(best_model_path, compile=False)

    def evaluate(self):
        self.pred_output = self.model.predict(self.test_input)
        self.pred_output = self.pred_output[:, :, :, 0]

    def see_random_results(self):
        PlotUtils.plot_random_results(self.test_output, self.test_input, self.pred_output)

    def get_metrics(self):
        self.test_mse = [Metrics.mse(self.test_output[i, :, :], self.pred_output[i, :, :])
                         for i in range(0, self.test_output.shape[0])]
        self.test_psnr = [Metrics.psnr(self.test_output[i, :, :], self.pred_output[i, :, :])
                          for i in range(0, self.test_output.shape[0])]

    def log_extreme_outputs(self, num=5, show=False):

        error = tf.math.reduce_sum(tf.math.square(self.pred_output - self.test_output), axis=(1, 2))
        sorted_index = sorted(range(len(error)), key=lambda k: error[k])

        title = "Worst Reconstructions"
        for i, index in enumerate(sorted_index[-num-1:-1]):
            plot_buf = PlotUtils.plot_errors(self.test_output[index, :, :], self.pred_output[index, :, :], title, show)
            image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            with self.file_writer.as_default():
                tf.summary.image("Worst Reconstructions", image, step=i)
            del image

        title = "Best Reconstructions"
        for i, index in enumerate(sorted_index[0:num]):
            plot_buf = PlotUtils.plot_errors(self.test_output[index, :, :], self.pred_output[index, :, :], title, show)
            image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            with self.file_writer.as_default():
                tf.summary.image("Best Reconstructions", image, step=i)
            del image

    # TODO Keeping this function here for the lack of a better place right now. Figure out where this goes later.
    def model_compare(self):
        """ Generate and plot results for a set of fixed indices everytime the function is run.
            The model used could be different and is set from the config file.
            Intent is to be able to compare results obtained by different models on the same samples
        """
        indices = {10, 20, 40, 60, 100}
        for index in indices:
            PlotUtils.generate_sim_result(self.test_output[index, :, :],
                                          self.test_input[index, :, :, 0],
                                          self.test_input[index, :, :, 1],
                                          self.pred_output[index, :, :])
            print(f"Index: {index}, MSE: {self.test_mse[index]}, PSNR: {self.test_psnr[index]}")

    def predict(self):
        """ Function used to predict output for out-of-sample simulation data or experiment data.
            Read input and expected output files from disk everytime
        """
        real_rec = loadmat(Config.config["test"]["chi_real_path"])["real_rec"]
        imag_rec = loadmat(Config.config["test"]["chi_imag_path"])["imag_rec"]

        if Config.config["test"]["output_path"]:
            ground_truth = loadmat(Config.config["test"]["output_path"])["scatterer"]
        else:
            ground_truth = None

        test_input = np.asarray([real_rec, imag_rec])
        test_input = np.moveaxis(test_input, 0, -1)
        test_input = test_input[np.newaxis, ...]

        self.y_pred = self.model.predict(test_input)
        self.y_pred = self.y_pred[0, :, :, 0]

        if ground_truth:
            PlotUtils.generate_sim_result(ground_truth, real_rec, imag_rec, self.y_pred)
        else:
            PlotUtils.generate_exp_result(real_rec, imag_rec, self.y_pred)
