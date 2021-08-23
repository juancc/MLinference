"""
Multi Strategy
General model that load a models configuration 
Run multiple models that are independednt one for another

PARAMETERS
    models = [AbcModel1, AbcModel2, Cascade,...]

JCA
Vaico
"""
import json
import logging

class Multi:
    """
    Run multiple models that are not related one after another
    """
    def __init__(self, models, *args, **kwargs):
        """
        :param models: [AbcModel1, AbcModel2, Cascade,...]
        """
        self.logger = logging.getLogger(__name__)
        self.list_models = models
        self.logger.info(f'Instantiating Multi strategy with {len(self.list_models)} models')


    def predict(self, frame, *args, **kwargs):
        # Run detector
        self.logger.info('Predicting with Multi strategy')
        predictions = []
        # Queue prediction: one prediction run after the other
        # If model running on same instance there is not improvement on running models ion parallel
        for model in self.list_models:
            predictions += model.predict(frame)
        
        self.logger.info(f'Returning prediction of {len(predictions)} objects')
        
        return predictions

# from concurrent import futures
# class Multi:
#     """
#     Run multiple models that are not related concurrently
#     """
#     def __init__(self, models, *args, **kwargs):
#         """
#         :param models: [AbcModel1, AbcModel2, Cascade,...]
#         """
#         self.logger = logging.getLogger(__name__)
#         self.list_models = models
#         self.max_workers = 5 # max number of models to trigger at same time
#         self.logger.info(f'Instantiating Multi strategy: {len(self.list_models)} models. Max workers {self.max_workers }')


#     def predict(self, frame, *args, **kwargs):
#         # Run detector
#         self.logger.info('Predicting with Multi strategy')
#         predictions = []
#         # Concurrent predictions
#         workers = min(self.max_workers, len(self.list_models))
#         self.logger.info(f'Running predictions with {workers} workers')
        
#         with futures.ThreadPoolExecutor(max_workers=workers) as executor:
#             to_do_map = {}
#             i=0
#             for model in self.list_models:
#                 future = executor.submit(model.predict, frame)
#                 to_do_map[future] = i
#                 i+=1
#             done_iter = futures.as_completed(to_do_map)
#             for future in done_iter:
#                 predictions += future.result()
            
#             self.logger.info(f'Returning prediction of {len(predictions)} objects')
#         return predictions

