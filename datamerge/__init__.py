from ._version import get_versions


__version__ = get_versions()["version"]
del get_versions

from . import _version

__version__ = _version.get_versions()["version"]

from . import dataclasses
from . import mergecore
from . import readersandwriters
from . import plotting

# from .plotting import plotFigure
# from .dataclasses import (
#     scatteringDataObj,
#     rangeConfigObj,
#     mergeConfigObj,
#     outputRangeObj,
# )
# from .findscaling import findScaling
# from .mergecore import mergeCore
# from .readersandwriters import (
#     scatteringDataObjFromNX,
#     outputToNX,
#     mergeConfigObjFromYaml,
# )