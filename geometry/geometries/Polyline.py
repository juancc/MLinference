"""Geometry Type Polylines
Instantiated with a list of coordinates in x, and a list of coordinates in y
The cordinates are corresponding by its index

To convert in arrays use:
import numpy as np
all_x = np.array(Polyline.all_x).astype(float).astype(int),
all_y = np.array(Polyline.all_y).astype(float).astype(int)

Edited (31-3-2020)
JCA
"""
import reprlib

from geometry.geometries.Geometry import Geometry


class Polyline(Geometry):
    __slots__ = ('all_x', 'all_y')

    def __init__(self, all_x, all_y):
        self.all_x = list(all_x)
        self.all_y = list(all_y)

        if len(self.all_y) != len(self.all_x):
            raise ValueError('Coordinates most have the same number of values')

    def centroid(self):
        return sum(self.all_x)/len(self), sum(self.all_y)/len(self)

    def _asdict(self):
        return {
            'all_x': self.all_x,
            'all_y': self.all_y
        }

    @classmethod
    def _fromdict(cls, info_dict):
        return cls(
            info_dict['all_x'],
            info_dict['all_y']
        )

    def __len__(self):
        return len(self.all_x)

    def __iter__(self):
        """Return pair of coordinates"""
        return zip(self.all_x, self.all_y)

    def __repr__(self):
        class_name = type(self).__name__
        args = {
            'all_x': self.all_x,
            'all_y': self.all_y
        }
        return '{}({})'.format(class_name, reprlib.repr(args))


if __name__ == '__main__':
    p1 = Polyline([1,2,3], [3,4,9])
    p2 = Polyline([1,2,3], [3,4,9])
    print(p1)
    print(len(p1))
    for i in p1:
        print(i)
    print(p1==p2)