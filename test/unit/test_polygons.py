import unittest

from geometry.geometries.Polygon import Polygon

class PolygonTests(unittest.TestCase):
    def test_equal(self):
        cords1 = [[1, 2], [2, 3]]
        cords2 = [[1, 2], [2, 3]]
        p1 = Polygon(cords1)
        p2 = Polygon(cords2)
        self.assertEqual(p1, p2)

    def test_not_equal(self):
        cords1 = [[1, 2], [1, 3]]
        cords2 = [[1, 2], [2, 3]]
        p1 = Polygon(cords1)
        p2 = Polygon(cords2)
        self.assertNotEqual(p1, p2)

    def test_centroid(self):
        cords = [[1, 2, 3], [1, 3, 3],[2, 4, 3]]
        p = Polygon(cords)
        self.assertEqual(p.centroid(), [1.3333333333333333, 3, 3])

    def test_asdict(self):
        cords = [[1, 2], [2, 3]]
        p1 = Polygon(cords)
        self.assertEqual(p1._asdict(), {'points': [[1, 2], [2, 3]]})


    def test_fromdict(self):
        cords = [[1, 2], [2, 3]]
        p1 = Polygon(cords)

        d = {'points': [[1, 2], [2, 3]]}
        pd = Polygon._fromdict(d)
        self.assertEqual(p1, pd)


if __name__ == '__main__':
    unittest.main()
