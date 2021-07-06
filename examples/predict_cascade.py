import cv2 as cv
from time import time
import logging

logging.basicConfig(level=logging.INFO)

from MLinference.architectures import Yolo4
from MLinference.architectures import KerasClassifiers
from MLinference.strategies import Cascade


model_classifier = KerasClassifiers.load('/home/juanc/Downloads/resnet_casco_23oct2.ml')
labels =Yolo4.read_class_names('/misdoc/vaico/MLinference/test/data/coco.names')
model_main = Yolo4.load('/misdoc/vaico/architectures/yolov4_tflite/checkpoints/yolov4_sota.tflite',
                       labels=labels, 
                       input_size=416)

model = Cascade(main_model={'model': model_main}, sub_models={'person': [{'model': model_classifier}]})

im = cv.imread('/misdoc/vaico/MLinference/test/data/channel3_2020-08-11_14_03_15')
start = time()
res = model.predict(im)
print(res)
print(f'Elapsed time: {time()-start} s')

# Draw test
try:
    from MLdrawer.drawer import draw
    draw(res[0], im)
    cv.imshow('image', im)
    cv.waitKey(0)
except Exception as e:
    print('MLdrawer not installed')
