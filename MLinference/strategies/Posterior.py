"""
Posterior Strategy
Run models with input of previous predictions

Parameters:
    - models: (list) models to run one after another over the predictions.
    Each time new elements could be added to predictions (Order matters)

JCA
Vaico
"""
import logging


class Posterior():
    def __init__(self, models, *args, **kwargs):
        """
        :param models: (list)  models to run one after another over the predictions.
        Each time new elements could be added to predictions (Order matters)
        """
        self.logger = logging.getLogger(__name__)
        self.models = models
        self.logger.info('Posterior strategy with {} models'.format(len(self.models)))

    def predict(self, im, *args, **kwargs):
        predictions=[]
        for model in self.models:
            predictions = model.predict(im, predictions=predictions)

        return predictions