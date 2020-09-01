"""Interface for different types of geometries that describe a predicted Object
JCA
Vaico
"""
import reprlib
from abc import ABC, abstractmethod


class Geometry(ABC):
    """Geometry protocol

    Implement __slots__ in subclasses
    By default, Python stores instance attributes in a per-instance dict named __dict__ .
    Dictionaries have a significant memory overhead because of the underlying hash table
    used to provide fast access. If you are dealing with millions of instances with
    few attributes, the __slots__ class attribute can save a lot of memory,
    by letting the interpreter store the instance attributes in a tuple instead of a dict .
    """
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def centroid(self):
        """Return the  the arithmetic mean position of all the points in the figure.
        Informally, it is the point at which a cutout of the shape could
        be perfectly balanced on the tip of a pin"""
        pass

    @abstractmethod
    def _asdict(self):
        # Return a dictionary version of the geometry
        pass

    @classmethod
    def _fromdict(cls, info_dict):
        raise  NotImplementedError('Class {} does not have implemented _fromdict method.'.format(cls.__name__))

    @abstractmethod
    def __iter__(self):
        pass

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({})'.format(class_name, reprlib.repr([i for i in self]))

