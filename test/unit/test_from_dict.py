import unittest
from geometry import creator
from geometry import Object
from geometry import BoundBox
from geometry import Point
from geometry import Circle
from geometry import Polygon


class test_from_dict(unittest.TestCase):
    def test_create_from_dict(self):
        d = {'label': 'tie', 'score': '0.42', 'boundbox': {'xmin': 248, 'ymin': 221, 'xmax': 260, 'ymax': 269}}
        obj_true = Object(
            geometry=BoundBox(xmin=d['boundbox']['xmin'],
                              ymin=d['boundbox']['ymin'],
                              xmax=d['boundbox']['xmax'],
                              ymax=d['boundbox']['ymax']),
            label=d['label'],
            score=d['score'],
            subobject= None)
        obj = creator.from_dict(d)
        self.assertEqual(obj,obj_true)

    def test_create_from_dict_point(self):
        d = {"score": "1.00", "point": {"x": 139, "y": 25}, "label": "varilla"}

        obj_true = Object(
            geometry=Point(
                x=d['point']['x'],
                y=d['point']['y']),
            label=d['label'],
            score=d['score'],
            subobject= None)
        obj = creator.from_dict(d)
        self.assertEqual(obj,obj_true)

    def test_create_from_dict_polygon(self):
        d = {"score": "1.00", "polygon":{'points': [[1, 2, 3], [1, 3, 3],[2, 4, 3]]} , "label": "mypol"}

        obj_true = Object(
            geometry=Polygon(d['polygon']['points']),
            label=d['label'],
            score=d['score'],
            subobject= None)
        obj = creator.from_dict(d)

        self.assertEqual(obj,obj_true)

    def test_create_from_dict_circle(self):
        d = {"score": "1.00", "circle": {"x": 139, "y": 25, 'r':100}, "label": "circulo"}

        obj_true = Object(
            geometry=Circle(
                x=d['circle']['x'],
                y=d['circle']['y'],
                r=d['circle']['r']),
            label=d['label'],
            score=d['score'],
            subobject= None)
        obj = creator.from_dict(d)
        self.assertEqual(obj,obj_true)

    def test_not_geometry(self):
        d = {'label': 'tie', 'score': '0.42', 'boundbox': {'xmin': 248, 'ymin': 221, 'xmax': 260, 'ymax': 269}, 'subobject': {'label': 'supertie', 'score': 0.8}}
        obj_true = Object(
            geometry=BoundBox(xmin=d['boundbox']['xmin'],
                              ymin=d['boundbox']['ymin'],
                              xmax=d['boundbox']['xmax'],
                              ymax=d['boundbox']['ymax']),
            label=d['label'],
            score=d['score'],
            subobject=Object(
                geometry=None,
                label=d['subobject']['label'],
                score=d['subobject']['score'],
                subobject=None
            ))
        obj = creator.from_dict(d)
        print(obj)
        self.assertEqual(obj, obj_true)

if __name__ == '__main__':
    unittest.main()
