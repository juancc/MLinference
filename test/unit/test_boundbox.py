import unittest

from MLinference.geometry import BoundBox


class BBTests(unittest.TestCase):

    def test_bb_iou_equal(self):
        a = BoundBox(1, 2, 3, 4)
        b = BoundBox(1, 2, 3, 4)
        self.assertEqual(1.0, a.iou(b))

    def test_bb_iou_half(self):
        a = BoundBox(0, 0, 10, 10)
        b = BoundBox(5, 0, 15, 10)
        self.assertEqual(0.375, a.iou(b))



if __name__ == '__main__':
    unittest.main()
