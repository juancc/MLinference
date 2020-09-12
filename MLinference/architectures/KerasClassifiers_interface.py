from keras import applications

class mobilenet:
    model = applications.mobilenet.MobileNet
    decode_predictions = applications.mobilenet.decode_predictions
    preprocess_input = applications.mobilenet.preprocess_input
class resnet50:
    model = applications.resnet50.ResNet50
    decode_predictions = applications.resnet50.decode_predictions
    preprocess_input = applications.resnet50.preprocess_input

class resnet50_imageAI:
    """Resnet implementation of library imageAI. Model is the same of Keras
    https://github.com/OlafenwaMoses/ImageAI
    """
    model = applications.resnet50.ResNet50

    def preprocess_input(x):
        """Preprocesses a tensor encoding a batch of images.

        # Arguments
            x: input Numpy tensor, 4D.
            data_format: data format of the image tensor.

        # Returns
            Preprocessed tensor.
        """

        # 'RGB'->'BGR' **Required
        x *= (1. / 255)
        return x

class xception:
    model = applications.xception.Xception
    decode_predictions = applications.xception.decode_predictions
    preprocess_input = applications.xception.preprocess_input

class vgg16:
    model = applications.vgg16.VGG16
    decode_predictions = applications.vgg16.decode_predictions
    preprocess_input = applications.vgg16.preprocess_input

class vgg19:
    model = applications.vgg19.VGG19
    decode_predictions = applications.vgg19.decode_predictions
    preprocess_input = applications.vgg19.preprocess_input

class resnet101:
    model = applications.resnet.ResNet101
    decode_predictions = applications.resnet.decode_predictions
    preprocess_input = applications.resnet.preprocess_input

class resnet152:
    model = applications.resnet.ResNet152
    decode_predictions = applications.resnet.decode_predictions
    preprocess_input = applications.resnet.preprocess_input

class resnet50v2:
    model = applications.resnet_v2.ResNet50V2
    decode_predictions = applications.resnet_v2.decode_predictions
    preprocess_input = applications.resnet_v2.preprocess_input

class resnet101v2:
    model = applications.resnet_v2.ResNet101V2
    decode_predictions = applications.resnet_v2.decode_predictions
    preprocess_input = applications.resnet_v2.preprocess_input

class resnet152v2:
    model = applications.resnet_v2.ResNet152V2
    decode_predictions = applications.resnet_v2.decode_predictions
    preprocess_input = applications.resnet_v2.preprocess_input

# class resnext50:
#     model = applications.keras_applications.resnext.ResNeXt50
#     decode_predictions = applications.keras_applications.resnext.decode_predictions
#     preprocess_input = applications.keras_applications.resnext.preprocess_input

# class resnext101:
#     model = applications.keras_applications.resnext.ResNeXt101
#     decode_predictions = applications.keras_applications.resnext.decode_predictions
#     preprocess_input = applications.keras_applications.resnext.preprocess_input

class inceptionv3:
    model = applications.inception_v3.InceptionV3
    decode_predictions = applications.inception_v3.decode_predictions
    preprocess_input = applications.inception_v3.preprocess_input

class inceptionresnetv2:
    model = applications.inception_resnet_v2.InceptionResNetV2
    decode_predictions = applications.inception_resnet_v2.decode_predictions
    preprocess_input = applications.inception_resnet_v2.preprocess_input

class densenet121:
    model = applications.densenet.DenseNet121
    decode_predictions = applications.densenet.decode_predictions
    preprocess_input = applications.densenet.preprocess_input

class densenet169:
    model = applications.densenet.DenseNet169
    decode_predictions = applications.densenet.decode_predictions
    preprocess_input = applications.densenet.preprocess_input

class densenet201:
    model = applications.densenet.DenseNet201
    decode_predictions = applications.densenet.decode_predictions
    preprocess_input = applications.densenet.preprocess_input

class nasnetlarge:
    model = applications.nasnet.NASNetLarge
    decode_predictions = applications.nasnet.decode_predictions
    preprocess_input = applications.nasnet.preprocess_input

class nasnetmobile:
    model = applications.nasnet.NASNetMobile
    decode_predictions = applications.nasnet.decode_predictions
    preprocess_input = applications.nasnet.preprocess_input

class mobilenetv2:
    model = applications.mobilenet_v2.MobileNetV2
    decode_predictions = applications.mobilenet_v2.decode_predictions
    preprocess_input = applications.mobilenet_v2.preprocess_input





