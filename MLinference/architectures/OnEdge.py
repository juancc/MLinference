"""
On Edge (Classifier)
    - Posterior model
    - Define the probability of being near an edge based the output of probability of edge and a ROI.
    - Take the prediction of previous model and define if certain classes are near to an edge
"""
import logging

from MLcommon import InferenceModel
from MLgeometry import Object
from MLgeometry import creator

class OnEdge(InferenceModel):
    def __init__(self, filepath, interest_labels=None, mask_label=None, threshold=0.2, labels=None, *args, **kwargs):
        """
        :param interest_labels: (list) labels to be analyzed for classification
        :param threshold: (float) min value to be considered on an edge
        :param labels: (dict) {idx: label} 0: away from edge
        """
        self.logger = logging.getLogger(__name__)
        self.interest_labels = interest_labels
        self.mask_label = mask_label
        self.threshold = threshold
        self.labels = labels

        self.logger.info('Instantiated OnEdge for {}. Threshold: {}'.format(self.interest_labels, threshold))

    def find_mask(self, predictions):
        for p in predictions:
            if p.label.lower() == self.mask_label.lower():
                return p
        return None


    def predict(self, x, predictions=None, custom_labels=None, *args, **kwargs):
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
                for p in predictions:
                    if p.label in self.interest_labels:
                        roi = p.geometry
                        crop = mask_obj.geometry.mask[int(roi.ymin): int(roi.ymax), int(roi.xmin): int(roi.xmax)]
                        score = crop[crop>0].mean()
                        idx= 1 if score>self.threshold else 0

                        try:
                            lbl = str(labels[idx]) if labels else str(idx)
                        except KeyError:
                            self.logger.error('Custom labels not provide name for {}. Using default'.format(idx))
                            lbl = str(idx)
                        new_obj = Object(
                                    geometry=None,
                                    label=lbl,
                                    subobject=None,
                                    score=None
                                )
                        if p.subobject:
                            p.subobject.append(new_obj)
                        else:
                            p.subobject = [new_obj]

        return predictions

if __name__ == '__main__':
    model = OnEdge(None)
    print(model)