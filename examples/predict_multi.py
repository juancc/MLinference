"""
Example of prediction with multiple models
"""
import cv2 as cv

from MLinference.architectures import Yolo4
from MLinference.architectures import UNet
from MLinference.strategies import Multi

im = cv.imread('test/data/im.png')

# List of models to run on prediction
models = [
    Yolo4('/path/to/model.tflite',
          labels=Yolo4.read_class_names('test/data/coco.names')),
    UNet('/path/to/model.tflite')
]

model = Multi(models)
res = model.predict(im)
