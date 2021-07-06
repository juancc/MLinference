"""
Model for group Keras available classifiers for automatic selection during training.
Main intention: Perform an architecture search using Training - Hyper-parameters optimization

Models for image classification with weights trained on ImageNet:
    ------------------------------------------------------------------------------
    |Model               Size 	Top-1 Accuracy 	Top-5 Accuracy 	Parameters 	Depth |
    |------------------------------------------------------------------------------
    |
     Xception           88 MB 	        0.790 	    0.945 	    22,910,480 	126
    |VGG16 	            528 MB 	        0.713 	    0.901 	    138,357,544 23
    |VGG19 	            549 MB 	        0.713 	    0.900 	    143,667,240 26
    |ResNet50 	        98 MB 	        0.749 	    0.921 	    25,636,712 	-
    |Resnet50_imageAI   ""              ""          ""              ""
    |ResNet101 	        171 MB 	        0.764 	    0.928 	    44,707,176 	-
    |ResNet152 	        232 MB 	        0.766 	    0.931 	    60,419,944 	-
    |ResNet50V2 	    98 MB 	        0.760 	    0.930 	    25,613,800 	-
    |ResNet101V2        171 MB 	        0.772 	    0.938 	    44,675,560 	-
    |ResNet152V2        232 MB 	        0.780 	    0.942 	    60,380,648 	-
    |ResNeXt50 	        96 MB 	        0.777 	    0.938 	    25,097,128 	-
    |ResNeXt101 	    170 MB 	        0.787 	    0.943 	    44,315,560 	-
    |InceptionV3        92 MB 	        0.779 	    0.937 	    23,851,784 	159
    |InceptionResNetV2 	215 MB 	        0.803 	    0.953 	    55,873,736 	572
    |MobileNet 	        16 MB 	        0.704 	    0.895 	    4,253,864 	88
    |MobileNetV2 	    14 MB 	        0.713 	    0.901 	    3,538,984 	88
    |DenseNet121 	    33 MB 	        0.750 	    0.923 	    8,062,504 	121
    |DenseNet169 	    57 MB 	        0.762 	    0.932 	    14,307,880 	169
    |DenseNet201 	    80 MB 	        0.773 	    0.936 	    20,242,984 	201
    |NASNetMobile 	    23 MB 	        0.744 	    0.919 	    5,326,716 	-
    |NASNetLarge 	    343 MB 	        0.825 	    0.960 	    88,949,818 	-
    ------------------------------------------------------------------------------

TRAIN OPTIMIZERS
Name and parameters in: https://keras.io/optimizers/

CALLBACK FUNCTIONS
Name and parameters in https://keras.io/callbacks/
* Also availabsle


JCA
Vaico
"""
import logging

import numpy as np
import cv2 as cv
from keras.preprocessing.image import img_to_array

from MLcommon import AbcModel
from MLgeometry.Object import Object

from MLinference.architectures.kerasClassifiers import interface

LOGGER = logging.getLogger(__name__)


class KerasClassifiers(AbcModel):
    """Integrated classifieras available in Keras
    Methods for load_weights, set_weights, get_weights... can be used through self.model.*
    """
    _defaults = {
        # Common in all feature extractors
        "input_shape": (224, 224, 3),  # multiple of 32, hw
        'include_top': True,
        'pooling': None,
        'labels': 'imagenet',
        'weights': 'imagenet',  #imagenet or None -> random init
        'feature_extractor': {
            'model': 'mobilenet',
            'alpha': 1.0,
            'depth_multiplier': 1,
            'dropout': 1e-3
        },
        'train_metrics': ['accuracy', ],
        'loss_function': 'categorical_crossentropy',
        'train_optimizer': {
            "function": 'Adam',
            'lr': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-08,
            'decay': 0.0
        },
        'train_callbacks': None,  # Dict: {'EarlyStopping':{'monitor='val_loss''}, 'ModelCheckpoint', 'TensorBoard'}
        'train_finetune_layers': None,  # integer counting from last layer, number of layers to be trained
        'pred_threshold': 0, # Minimum score value of prediction to return object
    }

    def __init__(self, conf, *args, **kwargs):
        # Keras models interfaces defined in auxfunc.interface
        conf.update(kwargs)
        self.interface = getattr(interface, conf.get('feature_extractor', self._defaults['feature_extractor'])['model'])
        super().__init__(**conf)

    def predict(self, img, pred_threshold=None, custom_labels=None, *args, **kwrags):
        labels = None
        if custom_labels: 
            LOGGER.info('Using custom labels for prediction')
            labels = dict(self.labels).update(custom_labels)
        threshold = pred_threshold if pred_threshold else self.pred_threshold

        # Preprocess and pack images data
        img = img if isinstance(img, list) else [img]
        imdata = []
        for i in img:
            if not isinstance(i, np.ndarray):  # Loaded with keras -> image.load_image
                x = img_to_array(i)
            else:  # Loaded with OpenCV
                x = cv.resize(i, (self.input_shape[1], self.input_shape[0]))  # resize is width, height
                x = x[..., ::-1].astype(np.float32)  # BGR -> RGB

            # x = np.expand_dims(x, axis=0)  # expands dimensions as it expects to have [1, None]
            x = self.interface.preprocess_input(x)
            imdata.append(x)
        imdata = np.asarray(imdata).astype(np.float32)

        preds = self.model.predict(imdata)
        resp = []
        for j in range(len(img)):
            if self.labels == 'imagenet':
                decoded_preds = self.interface.decode_predictions(preds, top=3)[j]
                object_name = decoded_preds[0][1]
                score = decoded_preds[0][2]
            else:
                decoded_preds = self.decode_predictions(preds, top=3, custom_labels=labels)[j]
                object_name = decoded_preds[0]
                score = decoded_preds[1]
            if score > threshold:
                resp.append(Object(
                    label=object_name,
                    score=score
                ))
        return resp

    def train(self, dataset, *args, **kargs):
        pass

    def load_architecture(self):
        """Return the weights (if TF compiled model) or dict of models"""
        LOGGER.info('Loading architecture: {}'.format(self.feature_extractor['model']))

        if self.weights == 'imagenet' and self.labels != 'imagenet':
            # Load imagenet weights with different number of classes
            # Imagenet model without top and copy weights to target model
            # Target model
            args = {
                **self.feature_extractor,
                "input_shape": self.input_shape,
                'include_top': self.include_top,
                'pooling': self.pooling,
                'classes': len(self.labels),
                'weights': None
            }
            del args['model']  # not an arg in model function in keras
            model = self.interface.model(**args)

            # Imagenet model
            _args = {
                **self.feature_extractor,
                "input_shape": self.input_shape,
                'include_top': False,
                'pooling': self.pooling,
                'classes': 1000,
                'weights': 'imagenet'
            }
            del _args['model']  # not an arg in model function in keras
            model_imag = self.interface.model(**_args)
            LOGGER.info('Copying weights from imagenet to custom model')
            for layer in model_imag.layers:
                try:
                    model.get_layer(name=layer.name).set_weights(layer.get_weights())
                    LOGGER.debug("Copy weights from layer {}".format(layer.name))
                except:
                    LOGGER.debug("Could not transfer weights for layer {}".format(layer.name))
        else:
            args = {
                **self.feature_extractor,
                "input_shape": self.input_shape,
                'include_top': self.include_top,
                'pooling': self.pooling,
                'classes': 1000 if self.labels == 'imagenet' else len(self.labels),
                'weights': self.weights
            }
            del args['model']

            model = self.interface.model(**args)

        return model

    def decode_predictions(self, preds, top=5, custom_labels=None):
        labels = custom_labels if custom_labels else self.labels
        results = []
        
        for pred in preds:
            idx = pred.argmax()
            results.append([labels[idx], pred[idx]])# label, score
            # top_indices = pred.argsort()[-top:][::-1]
            # for i in top_indices:
            #     each_result = []
            #     each_result.append(labels[i])
            #     each_result.append(pred[i])
            #     results.append(each_result)

        return results


if __name__ == '__main__':
    model = KerasClassifiers.load('/home/juanc/Downloads/resnet_casco_23oct2.ml')
    img_path = '/misdoc/datasets/baluarte/02:42:ac:11:00:02/2020-09-03_22:01:48'
    im = []
    for i in range(3):
        im.append(cv.imread(img_path))
    res = model.predict(im)
    print(res)
    print(len(res))