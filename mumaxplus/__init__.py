"""GPU accelerated micromagnetic simulator."""

import _mumaxpluscpp as _cpp

from .antiferromagnet import Antiferromagnet
from .dmitensor import DmiTensor
from .ferromagnet import Ferromagnet
from .fieldquantity import FieldQuantity
from .grid import Grid
from .parameter import Parameter
from .poissonsystem import PoissonSystem
from .scalarquantity import ScalarQuantity
from .strayfield import StrayField
from .timesolver import TimeSolver
from .variable import Variable
from .world import World
from . import util

# Import multi-GPU functionality
try:
    from _mumaxpluscpp import multigpu
except ImportError:
    # Multi-GPU support not available
    class DummyMultiGpu:
        @staticmethod
        def is_enabled():
            return False
        @staticmethod
        def initialize():
            pass
        @staticmethod
        def set_enabled(enabled):
            pass
    multigpu = DummyMultiGpu()

__all__ = [
    "_cpp",
    "Antiferromagnet",
    "DmiTensor",
    "Ferromagnet",
    "FieldQuantity",
    "Grid",
    "Parameter",
    "ScalarQuantity",
    "StrayField",
    "TimeSolver",
    "Variable",
    "World",
    "PoissonSystem",
    "util",
    "multigpu",
]
