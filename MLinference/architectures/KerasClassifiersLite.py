"""
KerasClassifiersLite
Model used for inference.
Extend MLcommon inference abstract class and return MLgeometry objects
Exported keras classifier as tf-lite model

JCA
Vaico
"""
import numpy as np
import cv2 as cv
import sys
import logging

try:
    import tensorflow as tf
except Exception:
    import tflite_runtime.interpreter as tflite

from MLgeometry import Object
from MLgeometry import BoundBox
from MLcommon import InferenceModel

class KerasClassifiersLite(InferenceModel):
    def __init__(self, filepath, labels=None, score_threshold=0.5, backend='tf', *args, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.score_threshold = score_threshold
        self.labels = {int(idx):label for idx,label in labels.items()} if labels else None# fix string idx to int
        self.backend = backend
        if 'tensorflow' in sys.modules.keys() and backend=='tf':
            try:
                self.interpreter = tf.lite.Interpreter(model_path=filepath)
            except AttributeError as e:
                self.logger.error('Cant not use tf backend. Try to update Tensorflow>2.2. Error: {}'.format(e))
                import tflite_runtime.interpreter as tflite
                self.interpreter = tflite.Interpreter(model_path=filepath)
                self.backend = 'tflite'
        else:
            try:
                import tflite_runtime.interpreter as tflite
                self.interpreter = tflite.Interpreter(model_path=filepath)
                self.backend = 'tflite'
            except Exception as e:
                self.logger.error('Cant not use tflite backend. Error: {}'.format(e))
                self.interpreter = tf.lite.Interpreter(model_path=filepath)
                self.backend = 'tf'
        self.logger.info('Using {} backend'.format(self.backend))
        # if self.backend == 'tf': self.predict = self.tf_predict

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.logger.info('Loaded model from: {}'.format(filepath))
    
    @classmethod
    def read_class_names(cls, class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    def predict(self, im, custom_labels=None, *args, **kwargs):
        """Prediction using tflite interpreter"""
        if self.labels:
            labels = dict(self.labels)
            if custom_labels:
                labels.update(custom_labels)
                self.logger.info('Using custom labels for prediction')
        elif not self.labels and custom_labels:
            labels = custom_labels
        else:
            labels = None

        input_size = tuple(self.input_details[0]['shape'][1:3])
        image_data = cv.resize(im, input_size)
        image_data = np.array(image_data, dtype = np.float32)
        image_data = np.expand_dims(image_data, 0)

        self.interpreter.set_tensor(self.input_details[0]['index'], image_data)
        self.interpreter.invoke()

        pred = self.interpreter.get_tensor(self.output_details[0]['index'])

        resp = []
        score =  pred.max()
        if score > self.score_threshold:
            resp.append(Object(
                label=labels[pred.argmax()],
                score=score
            ))
        return resp


if __name__ == '__main__':
    im_path = 'test/chev_aveo.jpeg'
    im = cv.imread(im_path)
    labels =KerasClassifiersLite.read_class_names('car_labels.txt')

    model_path = 'clasificador.tflite'
    model = KerasClassifiersLite(model_path, labels=labels)

    res = model.predict(im)
    print(res)