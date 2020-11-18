"""
MaskDistance (Classifier)
    - Posterior model
    - Define the probability of a class based the distance of a ROI and a mask
    - Take the prediction of previous model and define if certain classes belong to a class
"""
import logging

import cv2 as cv
import numpy as np
from numpy import linalg as LA

from MLcommon import InferenceModel
from MLgeometry import Object
from MLgeometry.geometries import Line

class MaskDistance(InferenceModel):
    def __init__(self, filepath, interest_labels=None, mask_label=None,
    thrs_mask=0.45, labels=None, debug=False, def_obj_scale=1.1, def_pred_threshold=1, *args, **kwargs):
        """
        :param interest_labels: (dict) labels to be analyzed for classification 
            with its corresponding obj_scale: (float) relative scale interest_labels, its pred_threshold
            and scale_reference (optional) if not present same object will be take as scale reference
            {
                'label':{
                    'obj_scale': 1.1,
                    'pred_threshold': 1,
                    'scale_reference: 'label-reference-object'
                }
            }
            ** If not specified. defaults will be used. 
        :param threshold: (float) min value to be considered to belong to class 0
        :param labels: (dict) {idx: label} 0: away from edge
        :param mask_label: (str) name of the label of the mask object
        :param thrs_mask: (float) mask preprocessing thresh
        :param obj_scale: (float) relative scale of interest_labels
        """
        self.logger = logging.getLogger(__name__)
        self.interest_labels = interest_labels
        self.mask_label = mask_label
        self.thrs_mask = thrs_mask
        self.labels = labels
        self.debug = debug

        # Default values
        self.def_obj_scale = def_obj_scale
        self.def_pred_threshold = def_pred_threshold

        self.logger.info('Instantiated MaskDistance for {}'.format(list(self.interest_labels)))

    def find_mask(self, predictions):
        """Find mask object and handle if mask should be kept"""
        for p in predictions:
            if p.label.lower() == self.mask_label.lower():
                return p
        return None

    def midpoint(self, ptA, ptB):
	    return (int((ptA[0] + ptB[0]) * 0.5), int((ptB[1])))

    def get_reference_distance(self, predictions, reference):
        """Get the mean lower border of a rederence object"""
        dist = []
        for p in predictions:
            if p.label.lower() == reference.lower():
                roi = p.geometry
                dist.append(roi.xmax-roi.xmin)
        if dist:
            return np.average(dist)
        else:
            return None



    def predict(self, x, predictions=None, custom_labels=None, debug=False, keep_mask=True,*args, **kwargs):
        """
        :param keep_mask: (bool) this parameter will be passed to the geometry of the mask object. 
            False: mask as matrix will not be stored only indexs
        :param debug: (bool) x will be drawn"""
        debug = debug if debug else self.debug
        # For custom labels
        if self.labels:
            labels = dict(self.labels)
            if custom_labels:
                labels.update(custom_labels)
                self.logger.info('Using custom labels for prediction')
        elif not self.labels and custom_labels:
            labels = custom_labels
        else:
            labels = None

        if predictions:
            if isinstance(predictions[0], dict):
                predictions = Object.from_dict(predictions)
            
            if self.labels:
                labels = dict(self.labels)
                if custom_labels:
                    labels.update(custom_labels)
                    self.logger.info('Using custom labels for prediction')
            elif not self.labels and custom_labels:
                labels = custom_labels
            else:
                labels = None
            
            # Get mask
            mask_obj = self.find_mask(predictions)
            if mask_obj:
                # Preprocess mask
                mask = mask_obj.geometry.mask

                # Change if mask should me kept
                mask_obj.geometry.keep_mask = keep_mask

                ths = cv.threshold(mask, self.thrs_mask*mask.max(), 1,cv.THRESH_TOZERO)[1]
                ths = cv.adaptiveThreshold(np.uint8(ths*255), 0.8, 
                    cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 5)
                
                # Get coords of pixels>0
                mask_coords = np.argwhere(ths.transpose()>0)
                for p in predictions:
                    # Distance of the lower points of a reference ROI to calculate the object scale
                    # If not provided the same object ROI will be used
                    reference_dist = None
                    if p.label in self.interest_labels:
                        # Get threshold, scale for each class and reference object
                        if 'pred_threshold' not in self.interest_labels[p.label]:
                            self.logger.info(f'Using default value: {self.def_pred_threshold} of "pred_threshold" for {p.label}')
                            pred_threshold = self.def_pred_threshold
                        else:
                            pred_threshold = self.interest_labels[p.label]['pred_threshold']
                        if 'obj_scale' not in self.interest_labels[p.label]:
                            self.logger.info(f'Using default value: {self.def_obj_scale} of "obj_scale" for {p.label}')
                            obj_scale = self.def_obj_scale
                        else:
                            obj_scale = self.interest_labels[p.label]['obj_scale']
                        if 'scale_reference' not in self.interest_labels[p.label]:
                            # Object label to be taken as scale reference 
                            self.logger.info(f'Using same object as scale reference')
                        else:
                            scale_reference = self.interest_labels[p.label]['scale_reference']
                            reference_dist = self.get_reference_distance(predictions, scale_reference)
                            if not reference_dist:
                                continue
                        
                        roi = p.geometry
                        # Calculate midpoint in the lower part of the object
                        c1 = [roi.xmin, roi.ymax]
                        c2 = [roi.xmax, roi.ymax]
                        midp = self.midpoint(c1, c2)

                        #Calculate pixel/scale ratio
                        reference_dist =  reference_dist if reference_dist else (c2[0]-c1[0])
                        D = reference_dist / (obj_scale)

                        # Center arround the midpoint
                        coords_t = mask_coords - midp

                        # Calculate the norm of each row
                        dist = LA.norm(coords_t, axis=1)
                        min_dist_idx = np.argmin(dist)

                        min_dist = dist[min_dist_idx] / D # Transform to the real scale 

                        # For adding Line geometry to prediction
                        near_coord = mask_coords[min_dist_idx] 
                        start_point = (int(midp[0]), int(midp[1]))
                        end_point = (int(near_coord[0]), int(near_coord[1]))

                        # Testing
                        if debug:
                            if not x is None:
                                x[mask > 0] = (0,255,0)
                                
                                pt1 = (int(roi.xmin), int(roi.ymin))
                                pt2 = (int(roi.xmax), int(roi.ymax))
                                cv.rectangle(x, pt1, pt2, (0,0,255), 2) 

                                cv.line(x, start_point, end_point, (255,0,0), 2)
                        
                        # Add subobject to prediction
                        idx = 1 if min_dist>pred_threshold else 0
                        try:
                            lbl = str(labels[idx]) if labels else str(idx)
                        except KeyError:
                            self.logger.error('Custom labels not provide name for {}. Using default'.format(idx))
                            lbl = str(idx)
                        
                        # Testing
                        if debug:
                            if not x is None:
                                x[mask > 0] = (0,255,0)
                                
                                pt1 = (int(roi.xmin), int(roi.ymin))
                                pt2 = (int(roi.xmax), int(roi.ymax))
                                cv.rectangle(x, pt1, pt2, (0,0,255), 2) 

                                cv.line(x, start_point, end_point, (255,0,0), 2)

                                cv.putText(x, lbl, pt2, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA) 


                        
                        new_obj = Object(
                                    label=lbl,
                                    properties={
                                        'distance': min_dist
                                    },
                                    geometry=Line(start_point, end_point)
                                )
                        if p.subobject:
                            p.subobject.append(new_obj)
                        else:
                            p.subobject = [new_obj]

        return predictions

def show_im(x):
    cv.imshow('window_name', x)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    import json

    logging.basicConfig(level=logging.DEBUG)
    
    im = cv.imread('test/data/channel3_2020-08-11_14_03_15')

    with open('test/data/preds-channel3_2020-08-11_14_03_15', 'r') as f:
        preds = json.load(f)

    interest_labels = {
        'persona':{
            'obj_scale': 1.12,
            'pred_threshold': 1
        },'balde':{
            'obj_scale': 1.12,
            'pred_threshold': 1,
            'scale_reference': 'persona'
        },
    }

    model = MaskDistance(None, interest_labels=interest_labels,  mask_label='borde',
                   labels={0:'cerca de borde', 1:'lejos de borde'})
    # print(model)
    res = model.predict(im, preds, debug=True, keep_mask=False)
    # print(res)
    # for_save = [r._asdict() for r in res]
    # print(for_save)
    # Save res 
    # with open('subobject-properties.json', 'w') as f:
    #     json.dump(for_save, f)


    show_im(im)
