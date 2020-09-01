"""
Polygon Geometry Class
Instantiate with any iterable each element most contain the same number of dimensions

SUPPORT
 - len
 - equals

JCA
Vaico
"""

from geometry.geometries.Geometry import Geometry

class Polygon(Geometry):

    __slots__ = ('points')

    def __init__(self, points):
        """
        Individual points most be indexable
        :arg points: iterable
        """
        self.points = list(points)

    def __iter__(self):
        return iter(self.points)

    def centroid(self):
        n_dim = len(self.points[0])
        c = []
        for n in range(n_dim):
            _sum = 0
            for point in self.points:
                _sum += point[n]
            c.append(_sum/len(self.points))
        return c

    def _asdict(self):
        return {
            'points': self.points,
        }

    def __len__(self):
        return len(self.points)

    @classmethod
    def _fromdict(cls, info_dict):
        return cls(info_dict['points'])

    def __eq__(self, other):
        return (len(self) == len(other) and
                all(a == b for a, b in zip(self, other)))


if __name__ == '__main__':
    cords = [[1,2], [2,3]]
    p1 = Polygon(cords)
    print(p1._asdict())

    d = {'points': [[1, 2], [2, 3]]}
    pd = Polygon._fromdict(d)
    print(pd)