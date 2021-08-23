"""
Example of prediction with multiple models
"""

import cv2 as cv
import logging
logging.basicConfig(level=logging.INFO)

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

from MLinference.architectures import KerasClassifiers
from MLinference.strategies import Multi

im = cv.imread('test/data/im.png')


# List of models to run on prediction
models = [
    KerasClassifiers.load('/misdoc/vaico/server-SURA2020/app/models/cascos.ml'),
    KerasClassifiers.load('/misdoc/vaico/server-SURA2020/app/models/arnes.ml'),
]

model = Multi(models)
res = model.predict(im)
print(res)
