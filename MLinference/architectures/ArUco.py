"""
ArUco Class
Works as a classifier where given a Boungox it define the ID if a marker is found
Given a ROI with Aruco markers return the an MLobject with ID as label of the marker.
When multiple markers are detected, return the most common.
Uses OpenCV implementation of ARUCO
https://www.docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html

JCA
Vaico
"""
import logging

import numpy as np
import cv2 as cv

from MLgeometry import Object
from MLcommon import InferenceModel


class ArUco(InferenceModel):
    def __init__(self, filepath, labels=None, ar_dict=None ,*args, **kwargs):
        """
        :param labels: (dict) {marker_id : object_id}
        :param ar_dict: (int) dictionary ID of opencv Aruco markers
        """
        self.logger = logging.getLogger(__name__)
        self.labels = labels

        ar_dict = ar_dict if ar_dict else cv.aruco.DICT_6X6_250
        self.logger.info('Using {} AR dictionary'.format({ar_dict}))

        # Load the dictionary that was used to generate the markers.
        self.dictionary = cv.aruco.Dictionary_get(ar_dict)

        # Initialize the detector parameters using default values
        self.parameters = cv.aruco.DetectorParameters_create()
        self.logger.info('Loaded model with: {} Ids'.format('Custom' if labels else 'Default' ))


    def predict(self, im, custom_labels=None, *args, **kwargs):
        labels = custom_labels if custom_labels else self.labels
        if custom_labels: self.logger.info('Using custom labels for prediction')

        marker_corners, marker_ids, rejected_candidates = cv.aruco.detectMarkers(im, self.dictionary, parameters=self.parameters)

        res = None
        if marker_ids:
            # Find the most common marker in ROI (Mode)
            (_, idx, counts) = np.unique(marker_ids, return_index=True, return_counts=True)
            index = idx[np.argmax(counts)]
            m_id = marker_ids[index][0]
            try:
                _id = labels[m_id] if labels else m_id
            except KeyError:
                self.logger.error('Custom labels not provide name for {}. Using default'.format(m_id))
                _id = m_id
            label = 'ID:{}'.format(_id)
            res = [Object(
                    geometry=None,
                    label= label,
                    subobject=None,
                    score=None)]
        return res



if __name__ == '__main__':
    im = cv.imread('test/data/marker_1.png')
    ar_dict = 10
    model = ArUco(None, ar_dict=ar_dict, labels={1:'juan'})
    res = model.predict(im, custom_labels={33:'carlos'})
    print(res)
