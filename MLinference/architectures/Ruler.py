"""
Model that return a evaluation of an image based the existent predictions of Vision models
And a set of rules. If some class or element is present on the prediction a value is added.

The model then assign the evaluation as a prediction of the image.

EVAL FUNCTION
The eval function is used to return a dict with the components of the evaluation and most 
have a 'total' key with the total of the evaluation

The eval function recieves the predictions as a dict
An example of eval_function is found on examples/ruler_model_example.py


JCA
Vaico
"""
import logging
from MLgeometry import Object
from MLcommon import InferenceModel

class Ruler(InferenceModel):
    def __init__(self, filepath, eval_function=None, label=None, *args, **kwargs) -> None:
        """
        :param eval_function: (function) function that return: date, frame_evaluation, ids_evaluation
            - frame evaluation is a dict containing the components of the evaluation and the 'total'
            - ids_evaluation return a dict containing the components of the evaluation and the total by ID found on image
        """
        self.logger = logging.getLogger(__name__)
        if eval_function:
            self.logger.info(f'Instantiated Ruler model with function {eval_function}')
            self.eval_function = eval_function
        else:
            err= 'eval_function not provided as class parameter'
            self.logger.error(err)
            raise TypeError(err)
        self.label = label if label else 'Ruler'
        
    def predict(self, x, predictions=None, *args, **kwargs):
        if predictions:
            # TODO: Support multiple image predictions
            if not isinstance(predictions[0], dict):
                # Pass predictions as dict
                self.logger.info('Casting predictions as dict')
                predictions = [Object._asdict(p) for p in predictions]
            
            self.logger.info('Passing predictions to eval function...')
            date, feval, id_eval = self.eval_function(predictions)

            self.logger.info('Adding evaluation to frame predictions')
            res = {
                'label': self.label,
                'score': 1,
                'properties': feval
            }
            predictions.append(res)

            self.logger.info('Casting predictions to MLgeometry')
            predictions = Object.from_dict(predictions)
        return predictions

    # Auxiliar functions for building eval_function
    @staticmethod
    def label_exists(label, list_objs, id_identifier='id:'):
        """Check if label is in list of objects. If label == 'id'
        Check if label with ID exists on list of objects. and return the ID"""
        if list_objs:
            for l in list_objs:
                obj_label = l['label'].lower()
                if label.lower() == 'id' and obj_label.startswith(id_identifier):
                    _id = obj_label.split(':')[-1]
                    return _id
                elif obj_label == label.lower():
                    return True
        return False

    @staticmethod
    def add_risk_person(_id, risk_label, points ,hist, defaults=None):
        """Add the risk of an _id to hist"""
        if _id:
            if _id not in hist:
                hist[_id] = dict(defaults) if defaults else {}
            hist[_id][risk_label] = points
            if 'total' in hist[_id]:
                hist[_id]['total'] += points
            else:
                hist[_id]['total'] = points
    @staticmethod
    def midpoint(ptA, ptB):
            return (ptA[0] + ptB[0] * 0.5), (ptB[1])
        
