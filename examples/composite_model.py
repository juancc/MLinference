import cv2 as cv

from MLdrawer.drawer import draw

from MLinference.architectures import Yolo4, UNet, OnEdge, KerasClassifiers, ArUco
from MLinference.strategies import Posterior, Multi, Cascade

im = cv.imread('/home/juanc/untitled.png')


worker_detector = Yolo4.load('/misdoc/vaico/architectures/yolov4_tflite/checkpoints/yolov4_sota.tflite',
                      labels={0:'persona'}, input_size=416)

helmet_model = KerasClassifiers.load('/home/juanc/Downloads/resnet_casco_sep.ml')
harness_model = KerasClassifiers.load('/home/juanc/Downloads/resnet_arnes_sep.ml')
harness_model.labels = ['sin arnes', 'con arnes']
id_model = ArUco(None, ar_dict=10)


worker_cascade = Cascade(
    main_model={'model':worker_detector},
    sub_models={'persona': [
        {'model': helmet_model},
        {'model': harness_model},
        {'model': id_model},
        
        ]}
)

edges_mask = UNet.load('/home/juanc/Downloads/bordes_Unet-20200901.tflite', labels={0:'borde'})
on_edge = OnEdge(None, interest_labels=['persona'], mask_label='borde', labels={0:'lejos de borde', 1:'cerca de borde'})

main_models = Multi(models=[worker_cascade, edges_mask])

model = Posterior(models=[main_models, on_edge])

res = model.predict(im)
print(res)


draw(res, im)
cv.imshow('Prediction',im)

cv.waitKey(0) # waits until a key is pressed
cv.destroyAllWindows() # destroys the window showing image