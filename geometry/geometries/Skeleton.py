import reprlib
from geometry.geometries.Geometry import Geometry


class Skeleton(Geometry):
    __slots__ = ('points', 'connections', 'names')

    def __init__(self, points, connections, names):
        self.points = list(points)
        self.connections = list(connections)
        self.names = list(names)

    def centroid(self):
        pass

    def _asdict(self):
        return {
            'points': self.points,
            'connections': self.connections,
            'names': self.names
        }

    @classmethod
    def _fromdict(cls, info_dict):
        return cls(
            info_dict['points'],
            info_dict['connections'],
            info_dict['names'],
        )

    def __iter__(self):
        return zip(self.points, self.connections, self.names)

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({})'.format(class_name, reprlib.repr([i for i in self]))