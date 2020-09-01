"""
Abstract class for Machine Learning models
Explicit model protocol
 
JCA
Vaico
"""
import pickle
import reprlib
import logging
from abc import ABC, abstractmethod

from common.exceptions import ConfigParamError

log = logging.getLogger(__name__)


class AbcModel(ABC):
    """Abstract class for classification and detection Architecture Models"""

    @property
    @abstractmethod
    def _defaults(self):
        """Dict with default model configuration.
        The parameters in this dict are copied to __dict__
        Thus are available as instance variables
        Used instead of __dict__ for avoiding mix with other
        instance variables used in the model
        """
        pass

    def __init__(self, *args, **kwargs):
        # Update _defaults values and **Over-write Nested Dicts** in defaults"""
        self._defaults = {**self._defaults, **kwargs}  # Update defaults
        self._defaults['architecture'] = type(self).__name__
        self.__dict__.update(self._defaults)  # set class variables
        self.model = self.load_architecture()  # set a prediction model
        super().__init__()


    @abstractmethod
    def predict(self, x, *args, **kwargs):
        """Predict function. Return model prediction type Geometries
        :x input
        :return predicted geometries
        """
        pass

    @abstractmethod
    def train(self, dataset, *args, **kargs):
        pass

    @abstractmethod
    def load_architecture(self):
        """Return a compiled model or dict of models"""
        pass

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({})'.format(class_name, reprlib.repr(self.__dict__))

    def get_config(self):
        """Get current values of _default variables"""
        config = {}
        for key in list(self._defaults):
            config[key] = getattr(self, key)
        return config

    def get_weights(self):
        """Return required raw data for saving a self-included model"""
        log.info('Returning weights from self model')
        return self.model.get_weights()

    def set_weights(self, _weights):
        log.info('Loading weights from object')
        self.model.set_weights(_weights)

    def save(self, path, include=None):
        """
        Save full model in backend and metadata of model,
         all the model parameters in signature are stored
        :param model: model instance to be saved
        :param path: path to store model
        :return: None
        """
        filename = path + '.ml'
        log.info('Saving model in: {}'.format(filename))

        # Try to save weights and model metadata
        full_model = {}
        try:
            # Get Keras model weights
            log.info('Getting model weights')
            full_model['weights'] = self.get_weights()
        except Exception:
            log.critical('Unable to get model weights.')
            raise ConfigParamError('Unable to get model weights.')

        # Read parameters of the model stored in _defaults
        config = self.get_config()
        full_model['conf'] = config

        # Change weights=='imagenet' to avoid download and reload from weights
        if 'weights' in full_model['conf']:
            if full_model['conf']['weights'] == 'imagenet':
                full_model['conf']['weights'] = None

        # Save model and metadata
        with open(filename, 'wb') as handle:
            pickle.dump(full_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        log.info('Model Saved!')

    @classmethod
    def load(cls, filepath, *args, **kwargs):
        """Decode file and create a model with the data and configuration"""
        log.info('Loading model from: {}'.format(filepath))

        with open(filepath, 'rb') as handle:
            model_data = pickle.load(handle)
        model = cls(conf=model_data['conf'], *args, **kwargs)
        model.set_weights(model_data['weights'])

        return model
