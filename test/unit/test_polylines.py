# import unittest
#
# from MLgeometry.geometries.Polyline import Polyline
#
# class PolylineTests(unittest.TestCase):
#     def test_equal(self):
#         p1 = Polyline([1,2,3,4], [1,2,3,4])
#         p2 = Polyline([1,2,3,4], [1,2,3,4])
#         self.assertEqual(p1, p2)
#
#     def test_not_equal(self):
#         p1 = Polyline([1, 2, 1, 3], [1, 2, 1, 3])
#         p2 = Polyline([1, 2, 2, 3], [1, 2, 2, 3])
#         self.assertNotEqual(p1, p2)
#
#     def test_centroid(self):
#         p = Polyline([1, 2, 3], [1, 3, 3])
#         self.assertEqual(p.centroid(), (2, 2.3333333333333335))
#
#     def test_asdict(self):
#         p1 = Polyline([1, 2], [2, 3])
#         self.assertEqual(p1._asdict(), {'all_x': [1, 2], 'all_y': [2, 3]})
#
#
#     def test_fromdict(self):
#         p1 = Polyline([1, 2], [2, 3])
#
#         d = {'all_x': [1, 2], 'all_y': [2, 3]}
#         pd = Polyline._fromdict(d)
#         self.assertEqual(p1, pd)
#
#
# if __name__ == '__main__':
#     unittest.main()
