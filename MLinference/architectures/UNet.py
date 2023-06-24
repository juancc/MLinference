"""
U-Net Architecture for segmentation
Based on: "U-Net: Convolutional Networks for Biomedical Image Segmentation".
By Ronneberger, Olaf; Fischer, Philipp; Brox, Thomas (2015).

Parameters
- labels: {idx:'label'}

"""
import sys
import logging

import numpy as np
import cv2 as cv
try:
    import tensorflow as tf
except Exception:
    import tflite_runtime.interpreter as tflite

from MLgeometry import Object
from MLgeometry import Mask
from MLcommon import InferenceModel


class UNet(InferenceModel):
    def __init__(self, filepath, labels=None, model_size='big', mask_threshold= 0.2, *args, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.model_size = model_size
        self.labels = labels if labels else {0: 'unknown'}
        self.mask_threshold = mask_threshold

        if 'tensorflow' in sys.modules.keys():
            try:
                self.interpreter = tf.lite.Interpreter(model_path=filepath)
            except AttributeError as e:
                print('Try to update Tensorflow. Error: {}'.format(e))
                import tflite_runtime.interpreter as tflite
                self.interpreter = tflite.Interpreter(model_path=filepath)
        else:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=filepath)

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # NxHxWxC, H:1, W:2
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.logger.info('Loaded model from: {}'.format(filepath))


    def predict(self, im, custom_labels=None, *args, **kwargs):
        labels = dict(self.labels)
        if custom_labels:
            labels.update(custom_labels)
            self.logger.info('Using custom labels for prediction')

        original_image = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        if self.model_size == "big":
            image_data = cv.resize(original_image, (512, 512))
        else:
            image_data = cv.resize(original_image, (256, 256))
        image_data = image_data / 255.
        images_data = []
        for i in range(1):
            images_data.append(image_data)

        images_data = np.asarray(images_data).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], images_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        intensity_map = np.squeeze(output_data)
        intensity_map = cv.threshold(intensity_map,  self.mask_threshold*intensity_map.max(), 1, cv.THRESH_TOZERO)[1]
        intensity_map = cv.resize(intensity_map, (original_image.shape[1], original_image.shape[0]))

        try:
            lbl = labels[0]
        except KeyError:
            self.logger.error('Custom labels not provide name')
            lbl = 'unknown'

        return [Object(Mask(intensity_map, [], keep_mask=True), lbl, float(np.mean(intensity_map)))]



if __name__ == '__main__':
    import sys
    sys.path.append('/misdoc/vaico/mldrawer/')
    from MLdrawer.drawer import draw


    im = cv.imread('test/data/im.png')
    model = UNet('/home/juanc/Downloads/bordes_Unet-20200901.tflite')
    res = model.predict(im)
    draw(res, im)

    cv.imshow('Prediction',im)

    cv.waitKey(0) # waits until a key is pressed
    cv.destroyAllWindows() # destroys the window showing image

