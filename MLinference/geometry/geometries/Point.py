"""
Multiple Dimensions Point Geometry Class
By default it takes 2 arguments Point(x,y)
But support multiple named dimensions i.e. Point(1,2,z=3,t=4)

SUPPORT
 - Iteration,
 - Get by index,
 - Len
 - Get value by dimension name Point.x

JCA
"""

from MLinference.geometry.geometries.Geometry import Geometry

class Point(Geometry):

    __slots__ = ('x', 'y')

    def __init__(self, x, y, **named_dimensions):
        self.x = x
        self.y = y
        for dim, value in named_dimensions.items():
            setattr(self, dim, value)
            self.__slots__ = self.__slots__ + (dim,)

    def centroid(self):
        return tuple(self.__iter__())

    def _asdict(self):
        return {i: getattr(self,i) for i in self.__slots__ }

    def __getitem__(self, index):
        return getattr(self, self.__slots__[index])

    @classmethod
    def _fromdict(cls, info_dict):

        # Extra dimensions
        extra_dim = {}
        for key, value in info_dict.items():
            if key != 'x' and key != 'y':
                extra_dim[key] = float(value)

        return cls(
            float(info_dict['x']),
            float(info_dict['y']),
            **extra_dim
        )

    def __iter__(self):
        return (getattr(self,i) for i in self.__slots__)

    def __len__(self):
        return len(self.__slots__)

    def __getattr__(self, name):
        if name in self.__slots__:
            return getattr(self, name)
        else:
            raise AttributeError('Point does not have attribute {}'.format(name))
