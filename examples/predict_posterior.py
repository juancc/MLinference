import cv2 as cv

from MLinference.architectures import Yolo4
from MLinference.architectures import UNet
from MLinference.architectures import OnEdge
from MLinference.strategies import Posterior, Multi


worker_detector = Yolo4.load('/misdoc/vaico/architectures/yolov4_tflite/checkpoints/yolov4_custom_v2.tflite',
                      labels={0:'persona'}, input_size=608)

edges_mask = UNet.load('/home/juanc/Downloads/Unet_bordes_sep1.tflite', labels={0:'borde'})
on_edge = OnEdge(None, interest_labels=['persona'], mask_label='borde', labels={0:'lejos', 1:'cerca'})
main_models = Multi(models=[worker_detector, edges_mask])

model = Posterior(models=[main_models, on_edge])

# im = cv.imread('test/data/im.png')
im = cv.imread('/misdoc/datasets/baluarte/00025/imgs/Edificio valuarte_ch3_20200723125753_20200723131711_19.jpg')
res = model.predict(im)
print(res)


import sys
sys.path.append('/misdoc/vaico/mldrawer/')
from MLdrawer.drawer import draw


draw(res, im)
cv.imshow('Prediction',im)

cv.waitKey(0) # waits until a key is pressed
cv.destroyAllWindows() # destroys the window showing image