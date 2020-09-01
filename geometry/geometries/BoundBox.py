from geometry.geometries.Geometry import Geometry

class BoundBox(Geometry):

    __slots__ = ('xmin', 'ymin', 'xmax', 'ymax')

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def centroid(self):
        return (self.xmin + (self.xmax - self.xmin)/2,
                self.ymin + (self.ymax - self.ymin)/2)

    def _asdict(self):
        return {
            'xmin' : self.xmin,
            'xmax' : self.xmax,
            'ymin' : self.ymin,
            'ymax' : self.ymax,
        }

    @classmethod
    def _fromdict(cls, info_dict):
        return cls(
            float(info_dict['xmin']),
            float(info_dict['ymin']),
            float(info_dict['xmax']),
            float(info_dict['ymax']),
        )

    def __iter__(self):
        return (i for i in (self.xmin,self.ymin,self.xmax,self.ymax))

    def iou(self, other):
        """Return the Interception Over Union
        Details in https://en.wikipedia.org/wiki/Jaccard_index"""
        xA = max(self.xmin, other.xmin)
        yA = max(self.ymin, other.ymin)
        xB = min(self.xmax, other.xmax)
        yB = min(self.ymax, other.ymax)

        # Area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # Union: sum of the two areas
        boxAArea = (self.xmax - self.xmin + 1) * (self.ymax - self.ymin + 1)
        boxBArea =  (other.xmax - other.xmin + 1) * (other.ymax - other.ymin + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou



if __name__ == '__main__':
    a = BoundBox(1,2,3,4)
    b = BoundBox(1,2,3,4)
    print(a.iou(b))
