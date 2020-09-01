"""Example for prediction using model weights"""
from MLinference.architectures.Yolo4 import Yolo4
import cv2 as cv


im = cv.imread('test/data/im.png')
labels =Yolo4.read_class_names('test/data/coco.nombres')
# Each architecture may need different parameters
# All architectures inherit the .load method
model = Yolo4.load('path/to/model.tflite',
                       labels=labels,
                       input_size=416)
# Result is a list of objects type MLinference.geometry.Objects
res = model.predict(im)
