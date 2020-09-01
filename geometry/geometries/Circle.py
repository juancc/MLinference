from geometry.geometries.Geometry import Geometry

class Circle(Geometry):

    __slots__ = ('x', 'y', 'r')

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def centroid(self):
        return (self.x, self.y)

    def _asdict(self):
        return {
            'x' : self.x,
            'y' : self.y,
            'r' : self.r
        }

    @classmethod
    def _fromdict(cls, info_dict):
        return cls(
            float(info_dict['x']),
            float(info_dict['y']),
            float(info_dict['r'])
        )


    def __iter__(self):
        return (i for i in (self.x, self.y, self.r))

