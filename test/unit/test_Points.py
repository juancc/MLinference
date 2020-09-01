import unittest

from geometry.geometries.Point import Point


class PointTests(unittest.TestCase):

    def test_point_centroid(self):
        p = Point(1, 2, z=3)
        self.assertEqual(p.centroid(), (1,2,3))

    def test_point_as_dict(self):
        p = Point(1, 2, z=3)
        self.assertEqual(p._asdict(), {'x': 1, 'y': 2, 'z': 3})

    def test_point_indexing(self):
        p = Point(1, 2, z=3)
        self.assertEqual(p[2], 3)

    def test_point_len(self):
        p = Point(1, 2, z=3, w=9)
        self.assertEqual(len(p), 4)

    def test_point_from_dict(self):
        d = {'x': 1, 'y': 2, 'z': 3, 't': 4}
        p = Point._fromdict(d)
        self.assertEqual(p, Point(1,2,z=3,t=4))




if __name__ == '__main__':
    unittest.main()
