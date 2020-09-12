import cv2 as cv

from MLinference.architectures import Yolo4
from MLinference.architectures import KerasClassifiers
from MLinference.strategies import Cascade


model_classifier = KerasClassifiers.load('/home/juanc/Downloads/cascos_resnet50-20200801.ml')
labels =Yolo4.read_class_names('/misdoc/vaico/MLinference/test/data/coco.names')
model_main = Yolo4.load('/misdoc/vaico/architectures/yolov4_tflite/checkpoints/yolov4_sota.tflite',
                       labels=labels, 
                       input_size=416)

model = Cascade(main_model={'model': model_main}, sub_models={'person': [{'model': model_classifier}]})


im = cv.imread('/misdoc/vaico/MLinference/test/data/im.png')
res = model.predict(im)
print(res)
