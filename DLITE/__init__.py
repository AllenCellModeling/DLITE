"""Top-level package for DLITE."""

__author__ = 'Ritvik Vasan'
__email__ = 'rvasan@eng.ucsd.edu'
__version__ = '0.1.0'


def get_module_version():
    return __version__

import matplotlib as mpl
mpl.use('TkAgg')

from . import cell_describe, ManualTracing, ManualTracingMultiple, SurfaceEvolver, AICS_data
