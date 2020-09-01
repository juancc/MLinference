# Vaico - Visual Artificial Intelligence
This package contains Abstract class for Machine Learning models with explicit model protocol.
Instantiate AbcModel for create a new model and ensure better server compatibility. Also contain predicted geometries and model architectures


## AbcModel methods

Implement required methods and set a dict of the default model parameters (Intended for the user to modify the model behaviour) in a variable **_defaults**. 

All the keywords parameters are loaded as instance variables and also stored in _defaults variable

```python
from MLcommon import AbcModel

class NewModel(AbcModel):
    _defaults = {
        'labels': None,
        'train_epsilon': 0.15
        'any_other': None
    }               
  
```

# Zip architectures
For share a new architecture compress it in zip format under the folder python for AWS lambda compatibility. Install the requirements of each architecture
## Models
Each model is loaded from two files:
- zip_path: path to architecture file (.zip)
- model_path: path to model file (.ml)

### Loading models example for Cascade
Loading models example for generating a Cascade configuration:
```python
conf['main_model']['model'] = load_zip_model(
    conf['main_model']['model_path'],
    conf['main_model']['zip_path'])
for roi, models in conf['sub_models'].items():
    for model_conf in models:
        model_conf['model'] = load_zip_model(model_conf['model_path'], model_conf['zip_path'])
```
# Prediction strategies:
## Cascade
Class for combine multiple prediction models based on the output of a main detector model.
The regions output of the detector are passed on other nodes (Classification or detection)
based some rules.

### Cascade Prediction Strartegy
 - Get predictions from a main detector
 - Results are passed to specialized classifiers/detector

All the predictions are obtained from other nodes
Classifiers are run in a concurrent requests
Inputs are:
    - Detector node direction
    - Subregion to be cropped from detector result
    - Subregions dict (detector-output : classifier-node)

#### Example
```python
conf: {
    'max_concur_req': 10,
    'main_model':{
        'model':AbcModel,
    }
     'sub_models':
        {
            'person':

                [
                    {
                        'model': AbcModel,
                        'weights': (0,0,1,1),
                        'conditions':['square_h']

                    },
                ],
            ...
        }
}
```
Use *all* for pass all the subregions in a classifier

## Multi
General model that load a models configuration 
Run multiple models that are independednt one for another

Example of node configuration
```python
conf={
    'models': [AbcModel1, AbcModel2, Cascade,...],
}
```

#### Roi
The roi inside the located class by the detector is defined by weights
Each weight from 0-1 of the values: (xi, yi, w, h)
Where 0 -> xi | 1 -> xi+W
Example:
    "roi_weights": (0.1, 1, 0.5, 1),
    "roi_conditions": ["center_x", ],

- Conditions:
    * center_x: center x direction
    * center_y: center y direction
    * square_w: square to width (after weighted)
    * square_h: square to height (after weighted)

# Geometry
Package for representing data as Objects with common geometries. Used as an interface for prediction elements in Computer Vision problems.
Each predicted object is represented as an Object. Which have geometry, label, score and subobject

Implemented:
- BoundBox
- Point
- Circle
- Polygon
- Polyline

## Importing
For use geometries:
```python
from MLgeometry import Point
P = Point(1,2,z=13)
```
Create from dict:
```python
from MLgeometry import creator
o_dict = {
    'geometry': 'boundbox',
    'label': 'myBB',
    'score': 1,
    'subobject': None
}

new_obj = creator.create_geometry(o_dict)

```
## Methods
Geometry special methods: Geometry._asdict() and Geometry._fromdict(dict)


# Architectures

## Yolo v4 Lite
Yolov4 port for inference with tensorflow lite

# Instalation of TF-lite
pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_x86_64.whl

# Zip
Compressed file for deploy on lambda layer. Include MLcommon module 








# License
All rights of this library belongs to Vaico Visión Artificial and no one may distribute, reproduce, or create derivative works.

Vaico
