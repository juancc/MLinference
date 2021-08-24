"""
Example of using Ruler model for predict the level of hazzards using eval_function

JCA
Vaico
"""
import numpy as np
import json
import logging

from MLinference.architectures import Ruler
from MLgeometry import Object

logging.basicConfig(level=logging.INFO)

# Load predictions
with open("test/data/preds-risks-baluarte.json", 'r') as f:
    frame_prediction = json.load(f)

frame_prediction = Object.from_dict(frame_prediction)

# -- Example of eval function --
# General variables
person_scale = 1.12 # Pixel - Real life Object scale
min_dist = 2 # Min distance between persons to be considered near - in meters

# EVALUATION METRICS
# Persons related risks
person = 0.2
no_helmet = 3
person_edge = 2
no_helmet_edge = 6
no_harness_edge = 6

# Obstacles related
obstacle_labels = ['balde', 'carretilla', 'tubo', 'tabla']
obstacle = 0.1
obstacle_edge = 3


def eval_function(frame_prediction):
    """Example of eval function"""

    date = frame_prediction['date'] if 'frame' in frame_prediction else None
    # Frame evaluation
    feval = {
        'total':0,
        'sin casco':0,
        'personas en borde': 0,
        'sin arnes': 0,
        'personas': 0,
        'balde en borde': 0,
        'carretilla en borde': 0,
        'tubo en borde': 0,
        'tabla en borde': 0,
        'obstáculos': 0,
        # 'obstáculos en borde': 0,
    }

    # ID evaluation
    id_feval = {
        'total':0,
        'sin casco':0,
        'personas en borde': 0,
        'sin arnes': 0,
    }
    ids = {}
    position_ids = {} # persons with ID in frame 'id': (x,y) 
    pixel_ratio = [] # pixel/scale ratio for distance between persons - average

    for obj in frame_prediction:
        if obj['label'] == 'persona':
            feval['total'] += person
            feval['personas'] += person

            subobjs = obj['subobject']

            _id = Ruler.label_exists('id', subobjs)
            if _id: # Person position as middle point of lower edge 
                geo = obj['boundbox']
                c1 = [geo['xmin'], geo['ymax']]
                c2 = [geo['xmax'], geo['ymax']]
                pos = Ruler.midpoint(c1, c2)
                position_ids[_id] = pos
                if not pixel_ratio: # Calculate pixel/scale ratio
                    pixel_ratio.append((c2[0]-c1[0]) / person_scale)

            on_edge = Ruler.label_exists('cerca de borde', subobjs)
            without_helmet = Ruler.label_exists('sin casco', subobjs)
            without_harness = Ruler.label_exists('sin arnes', subobjs)

            if not on_edge and without_helmet:
                feval['total'] += no_helmet
                feval['sin casco'] += no_helmet
                Ruler.add_risk_person(_id, 'sin casco', no_helmet, ids, defaults=id_feval)
                
            if on_edge:
                feval['total'] += person_edge
                feval['personas en borde'] += person_edge

                if without_helmet:
                    feval['total'] += no_helmet_edge
                    feval['sin casco'] += no_helmet_edge
                    Ruler.add_risk_person(_id, 'sin casco', no_helmet_edge, ids, defaults=id_feval)

                if without_harness:
                    feval['total'] += no_harness_edge
                    feval['sin arnes'] += no_harness_edge
                    Ruler.add_risk_person(_id, 'sin arnes', no_harness_edge, ids, defaults=id_feval)

        elif obj['label'] in obstacle_labels:
            feval['total'] += obstacle
            feval['obstáculos'] += obstacle

            if 'subobject' in obj:
                subobjs = obj['subobject']
                on_edge = Ruler.label_exists('cerca de borde', subobjs)
                if on_edge:
                    feval['total'] += obstacle_edge
                    feval[f'{obj["label"]} en borde'] += obstacle_edge
    # Add personas-cerca to each id
    for i in ids:
        ids[i]['personas-cerca'] = []
    
    
    # Calculate distance between persons with ID
    if position_ids:
        ratio = np.average(pixel_ratio)
        for _id, pos in position_ids.items():
            for next_id, next_pos in position_ids.items():
                if _id != next_id:
                    # Calculate distance between persons
                    p1 = np.array(pos)
                    p2 = np.array(next_pos)
                    dist = np.linalg.norm(p1 - p2) / ratio
                    if dist < min_dist:
                        ids[_id]['personas-cerca'].append(next_id)
    return date, feval, ids


model = Ruler(None, eval_function=eval_function, label='risk-assestment')

preds = model.predict(None, frame_prediction)

print(preds)