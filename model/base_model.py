from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""
    def __init__(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
