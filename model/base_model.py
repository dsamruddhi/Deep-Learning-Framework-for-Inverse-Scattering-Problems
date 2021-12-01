from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self):
        pass

    @abstractmethod
    def load_data(self):
        """ Load input and output data as training and test datasets"""
        pass

    @abstractmethod
    def build(self):
        """ Create the model, set loss metrics and optimization algorithm and parameters"""
        pass

    @abstractmethod
    def log(self):
        """ Log model and training config and other things needed """
        pass

    @abstractmethod
    def callbacks(self):
        """ Create callbacks needed during training """
        pass

    @abstractmethod
    def train(self):
        """ Train model based on training parameters specified in config file """
        pass

    @abstractmethod
    def evaluate(self):
        """ Method to obtain model output on test dataset"""
        pass

    @abstractmethod
    def predict(self):
        """ Method to predict model output for out-of-sample simulation and/or experiment data """
        pass
