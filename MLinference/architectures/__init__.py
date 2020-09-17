try:
    # Tensorflow architectures
    from MLinference.architectures.Yolo4 import Yolo4
    from MLinference.architectures.UNet import UNet
except Exception as e:
    print('Error: "{}" trying to import Yolo4 and UNet'.format(e))
try:
    # Keras architectures
    from MLinference.architectures.KerasClassifiers import KerasClassifiers
except Exception as e:
    print('Error: "{}" trying to import KerasClassifier'.format(e))
try:
    # Opencv architectures
    from MLinference.architectures.ArUco import ArUco
except Exception as e:
    print('Error: "{}" trying to import ArUco'.format(e))

# No especial dependencies
from MLinference.architectures.OnEdge import OnEdge