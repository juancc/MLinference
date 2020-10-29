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
from MLgeometry import creator

class MaskDistance(InferenceModel):
    def __init__(self, filepath, interest_labels=None, mask_label=None, threshold=1,
    thrs_mask=0.45, obj_scale=1.12, labels=None, debug=False, *args, **kwargs):
        """
        :param interest_labels: (list) labels to be analyzed for classification
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
        self.threshold = threshold 
        self.labels = labels
        self.obj_scale = obj_scale
        self.debug = debug

        self.logger.info('Instantiated MaskDistance for {}'.format(self.interest_labels))

    def find_mask(self, predictions):
        for p in predictions:
            if p.label.lower() == self.mask_label.lower():
                return p
        return None

    def midpoint(self, ptA, ptB):
	    return (int((ptA[0] + ptB[0]) * 0.5), int((ptB[1])))


    def predict(self, x, predictions=None, custom_labels=None, debug=False, *args, **kwargs):
        """ If debug x will be drawn"""
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
                predictions = creator.from_dict(predictions)
            
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
                ths = cv.threshold(mask, self.thrs_mask*mask.max(), 1,cv.THRESH_TOZERO)[1]
                ths = cv.adaptiveThreshold(np.uint8(ths*255), 0.8, 
                    cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 5)
                
                # Get coords of pixels>0
                mask_coords = np.argwhere(ths.transpose()>0)
                for p in predictions:
                    if p.label in self.interest_labels:
                        roi = p.geometry
                        # Calculate midpoint in the lower part of the object
                        c1 = [roi.xmin, roi.ymax]
                        c2 = [roi.xmax, roi.ymax]
                        midp = self.midpoint(c1, c2)

                        #Calculate pixel/scale ratio
                        D = (c2[0]-c1[0]) / (self.obj_scale)

                        # Center arround the midpoint
                        coords_t = mask_coords - midp

                        # Calculate the norm of each row
                        dist = LA.norm(coords_t, axis=1)
                        min_dist = np.min(dist) / D # Transform to the real scale 

                        # Testing
                        if debug:
                            min_dist_idx = np.argmin(dist)
                            near_coord = mask_coords[min_dist_idx] # transform back to xage coords
                            start_point = (int(midp[0]), int(midp[1]))
                            end_point = (int(near_coord[0]), int(near_coord[1]))
                            if not x is None:
                                x[mask > 0] = (0,255,0)
                                
                                pt1 = (int(roi.xmin), int(roi.ymin))
                                pt2 = (int(roi.xmax), int(roi.ymax))
                                cv.rectangle(x, pt1, pt2, (0,0,255), 2) 

                                cv.line(x, start_point, end_point, (255,0,0), 2)
                        
                        # Add subobject to prediction
                        idx= 1 if min_dist>self.threshold else 0
                        try:
                            lbl = str(labels[idx]) if labels else str(idx)
                        except KeyError:
                            self.logger.error('Custom labels not provide name for {}. Using default'.format(idx))
                            lbl = str(idx)
                        new_obj = Object(
                                    geometry=None,
                                    label=lbl,
                                    subobject= Object(
                                        geometry=None,
                                        label=min_dist,
                                        subobject=None,
                                        score=None
                                    ),
                                    score=None
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

    im = cv.imread('test/data/ch1_2020-03-17_07:29:01')

    with open('test/data/mask_predictions', 'r') as f:
        preds = json.load(f)

    model = MaskDistance(None, interest_labels=['persona'],  mask_label='borde',
                   labels={0:'lejos de borde', 1:'cerca de borde'})
    # print(model)
    res = model.predict(im, preds, debug=True)
    print(res)
    show_im(im)
