"""Aerospace Utility Library.

This package provides a collection of modules for performing aerospace-related
computations, including atmospheric modeling, coordinate transformations,
aerodynamic performance analysis, and flight trajectory tools (FTT).

Modules:
- atmos_1976   : U.S. Standard Atmosphere 1976 model.
- constants    : Physical and mathematical constants.
- das          : Data acquisition and signal processing utilities.
- geo          : Geospatial computations and utilities.
- transform    : Transform class.

FTT Modules:
- TFB          : Tower fly-by functions.
- aero_mod     : Aerodynamic modeling functions.
- FQ           : Flight quality metrics and evaluation.
- performance  : Aircraft performance calculations.

Version: 1.0.5
"""

from . import atmos_1976
from . import constants
from . import das
from . import geo
from . import transform


# modules for FTT
from . import aero_mod
from . import flying_qualities
from . import performance
from . import tower_fly_by

VERSION = '1.0.5'

__all__ = [
    "atmos_1976", "constants", "das", "geo", "transform",
    "performance", "aero_mod", "flying_qualities", "tower_fly_by"
]
