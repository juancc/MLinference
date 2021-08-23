"""
Class for combine multiple prediction nodes based on the output of a main detector model.
The regions output of the detector are passed on other nodes (Classification or detection)
based some rules.

Cascade Prediction Node
 - Get predictions from a main detector
 - Results are passed to specialized classifiers/detector

All the predictions are obtained from other nodes
Classifiers are run in a concurrent requests
Inputs are:
    - Detector node direction
    - Subregion to be cropped from detector result
    - Subregions dict (detector-output : classifier-node)

PARAMETERS
    main_model = {'model': AbcModel},

    sub_models ={
            'person':

                [
                    {
                        'model': AbcModel,
                        'weights': (0,0,1,1),
                        'conditions':['square_h']

                    },
                ],
            ...
        }

Use *all* for pass all the subregions in a classifier

ROI
    The roi inside the located class by the detector is defined by weights
    Each weight from 0-1 of the values: (xi, yi, w, h)
    Where 0 -> xi | 1 -> xi+W
    Example:
        "roi_weights": (0.1, 1, 0.5, 1),
        "roi_conditions": ["center_x", ],

    - Conditions:
        * center_x: center x direction
        * center_y: center y direction
        * square_w: square to width (after weighted)
        * square_h: square to height (after weighted)


JCA
Vaico
"""
import logging
from collections import defaultdict

from MLinference.strategies.auxfunc.cropper import crop_rect


class Cascade():
    """
    Cascade strategy of passing the input for several models to build a final prediction.
    Usually general prediction is returned by a Detector and specialized Classifiers
    run on the output of the detector.
    Instantiate outside lambda_function.
    """
    def __init__(self, main_model, sub_models, *args, **kwargs):
        """
            main_model = {'model': AbcModel},
            sub_models ={
                    'person':

                        [
                            {
                                'model': AbcModel,
                                'weights': (0,0,1,1),
                                'conditions':['square_h']

                            },
                        ],
                    ...
                }
        """
        self.logger = logging.getLogger(__name__)
        self.main_model = main_model
        self.sub_models = sub_models
        self.logger.info('Cascade strategy with {} main model and {} sub-models'.format(self.main_model,  len(self.sub_models)))

    def predict(self, frame, *args, **kwargs):
        # Run detector
        self.logger.info(f'Cascade with main model {self.main_model}. Number of images: {len(frame)}')
        frame_preds = self.main_model['model'].predict(frame) # Return a list of prediction for image on frame

        # Create a list of image crops to be predicted with the same model
        # Track where do belong each image crop (to which object on the main predictions) 
        # by storing the image index and the object index
        # image index is the id of the list of frame_preds and the object index is the id of the list of areas
        # After add all image crops predict with each model the list of crops
        to_predict = defaultdict(lambda: {'im': [], 'idx': []})# { model: {'im': [list of images], 'idx':  [area_id, roi_id] } }
        area_id = 0
        self.logger.info(f'Adding areas to predict...')
        for areas in frame_preds: # for prediction of each image
            # Queue prediction: one prediction run after the other
            self.logger.info(f'Predicting {len(areas)} areas with {len(self.sub_models)} sub-models')
            roi_id = 0
            for roi in areas:
                if roi.label in self.sub_models:
                    area = {
                            'xmin': roi.geometry.xmin,
                            'ymin': roi.geometry.ymin,
                            'xmax': roi.geometry.xmax,
                            'ymax': roi.geometry.ymax,
                        }
                    for model in self.sub_models[roi.label]: 
                        im_crop = crop_rect(
                            frame, 
                            area,
                            model['weights'] if 'weights' in model else[0,0,1,1],
                            model['conditions'] if 'conditions' in model else []
                        )
                        to_predict[model['model']]['im'].append(im_crop)
                        to_predict[model['model']]['idx'].append((area_id, roi_id))
                roi_id += 1
            area_id += 1
        
        self.logger.info(f'Predicting list of image crops with {len(to_predict)} models')
        for model, im_data in to_predict.items():
            crop_preds = model.predict(im_data['im'])
            for i in im_data['idx']:
                newobj = crop_preds[i[1]]
                try:
                    frame_preds[i[0]][i[1]].subobject.append(newobj)
                except AttributeError:
                    frame_preds[i[0]][i[1]].subobject = [newobj]
        return frame_preds


