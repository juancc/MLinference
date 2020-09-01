"""
Multi Strategy
General model that load a models configuration 
Run multiple models that are independednt one for another

Example of node configuration
conf={
    'models': [AbcModel1, AbcModel2, Cascade,...],
}

JCA
Vaico
"""
import json
import logging

class Multi:
    """
    Run multiple models that are not related simultaneously
    """
    def __init__(self, conf):
        """
        :param conf: dict with loaded models and crop image configuration
        """
        self.logger = logging.getLogger(__name__)
        self.list_models = conf['models']
        self.logger.info('Instantiating Multi strategy with {} models'.format(len(self.list_models)))


    def predict(self, frame):
        # Run detector
        self.logger.info('Predicting with Multi strategy')
        predictions = []
        # Queue prediction: one prediction run after the other
        self.logger.info('Predicting with {} models'.format(len(self.list_models)))
        for model in self.list_models:
            predictions.append(model.predict(frame))
        self.logger.info('Returning prediction of {} objects'.format(len(predictions)))
        
        return predictions
