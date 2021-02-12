from MLinference.architectures import KerasClassifiersLite
import cv2 as cv

model_path = '/misdoc/vaico/models/Classifiers/car_classifier.tflite'
labels = KerasClassifiersLite.read_class_names('/misdoc/vaico/models/Classifiers/car_labels.txt')
model = KerasClassifiersLite(model_path, labels)

im = cv.imread('/home/juanc/tmp/aveo.jpeg')
res = model.predict(im)
print(res)