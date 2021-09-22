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
from MLgeometry.geometries import Polygon
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

    @staticmethod
    def _split(im, nrows, ncols):
        """Split a matrix into sub-matrices."""
        imgheight=im.shape[0]
        imgwidth=im.shape[1]

        y1 = 0
        M = imgheight//nrows
        N = imgwidth//ncols

        tiles = []
        for y in range(0,imgheight,M):
            row = []
            for x in range(0, imgwidth, N):
                y1 = y + M
                x1 = x + N
                row.append(im[y:y+M,x:x+N])
            tiles.append(row)
        return tiles

    def predict(self, im, custom_labels=None, as_detector=False, split=None, *args, **kwargs):
        """Find all the markers in the image if run as detector each marker will be taken as an individual object.
        Otherwise only one marker will be defined as in the image
        :param split: (tuple) of (number rows, number columns)
            Each image part is passed by the model and predictions are assembled
        """
        labels = custom_labels if custom_labels else self.labels
        if custom_labels: self.logger.info('Using custom labels for prediction')

        if as_detector and split:
            self.logger.info(f'Splitting image on {split} parts...')
            
            # Split images
            nrows= split[0]
            ncols = split[1]
            imgheight=im.shape[0]
            imgwidth=im.shape[1]

            y1 = 0
            M = imgheight//nrows
            N = imgwidth//ncols
            marker_corners = []
            marker_ids = []
            for y in range(0,imgheight,M):# rows
                for x in range(0, imgwidth, N):# cols
                    y1 = y + M
                    x1 = x + N
                    tile = im[y:y+M,x:x+N]

                    tile_marker_corners, tile_marker_ids, _ = cv.aruco.detectMarkers(tile, self.dictionary, parameters=self.parameters)
                    for i in range(len(tile_marker_corners)):
                        mc = tile_marker_corners[i]
                        offset = np.array([[x,y]])
                        marker_corners.append(mc+offset)
                        marker_ids.append(tile_marker_ids[i])
        else:
            marker_corners, marker_ids, rejected_candidates = cv.aruco.detectMarkers(im, self.dictionary, parameters=self.parameters)

        res = []
        if marker_ids is not None:
            if as_detector:
                self.logger.info(f'Running as detector. Found {len(marker_ids)} markers')
                for i in range(len(marker_ids)):
                    m_id = marker_ids[i][0]
                    try:
                        _id = labels[m_id] if labels else m_id
                    except KeyError:
                        self.logger.info('Custom labels not provide name for {}. Using default'.format(m_id))
                        _id = m_id
                    label = 'ID:{}'.format(_id)
                    res.append(
                        Object(
                            geometry=Polygon(marker_corners[i][0]),
                            label = label
                        )
                    )
            else:
                # Find the most common marker in ROI (Mode)
                (_, idx, counts) = np.unique(marker_ids, return_index=True, return_counts=True)
                index = idx[np.argmax(counts)]
                m_id = marker_ids[index][0]
                try:
                    _id = labels[m_id] if labels else m_id
                except KeyError:
                    self.logger.info('Custom labels not provide name for {}. Using default'.format(m_id))
                    _id = m_id
                label = 'ID:{}'.format(_id)
                res = [Object(
                        geometry=None,
                        label= label,
                        subobject=None,
                        score=None)]
        return res


if __name__ == '__main__':
    import json

    logging.basicConfig(level=logging.DEBUG)

    # im = cv.imread('test/data/multiple_markers.jpg')
    im = cv.imread('test/data/marker_10x10px.jpg')

    ar_dict = 10
    model = ArUco(None, ar_dict=ar_dict, labels={1:'juan'})
    res = model.predict(im, custom_labels={33:'carlos'}, as_detector=True, split=(2,2))
    print(res)

    from MLdrawer.drawer import draw
    draw(res, im)

    cv.imshow("window_name", im)
    cv.waitKey(0) 
    cv.destroyAllWindows() 



    # for_save = [r._asdict() for r in res]
    
    # with open('markers-prediction.json', 'w') as f:
    #     json.dump(for_save, f)