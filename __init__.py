"""Private package of J. Xu"""

# Date of update: 10.23.2019
# Author: Jiachen Xu


import numpy as np
from .basiccmd import *
from .data import *
from .onlinestream import *
from .hardware import *
from .signalprocessing import *


__version__ = '0.0.1'

# have to import verbose first since it's needed by many things
"""from .utils import (set_log_level, set_log_file, verbose, set_config,
                    get_config, get_config_path, set_cache_dir,
                    set_memmap_min_size, grand_average, sys_info, open_docs)
"""
