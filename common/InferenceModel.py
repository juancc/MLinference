"""
Abstract class for inference Machine Learning models
Explicit model protocol of lite models only for prediction
 
JCA
Vaico
"""
import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

class InferenceModel(ABC):
    """Abstract class for inference models"""
    
    def __init__(self, filepath, *args, **kwargs):
        """
        :arg filepath: (Str) path of self contained model
        """
        super().__init__() 

    @abstractmethod
    def predict(self, x, *args, **kwargs):
        """Predict function. Return model prediction type Geometries
        :x input
        :return predicted geometries
        """
        pass

    @classmethod
    def load(cls, filepath, *args, **kwargs):
        """Return model loaded from file. Support with ABCmodels"""
        log.info('Loading model from: {}'.format(filepath))
        model = cls(filepath, *args, **kwargs)
        return model
