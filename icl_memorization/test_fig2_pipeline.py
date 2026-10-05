import os
import sys
import glob
import math
import pickle
from collections import defaultdict

fmri_root = os.path.dirname(os.path.dirname(sys.executable))
fmri_bin = os.path.join(fmri_root, "bin")
fmri_lib = os.path.join(fmri_root, "lib")
os.environ["PATH"] = fmri_bin + os.pathsep + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = fmri_lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import seaborn as sns
import utils

print("Imports successful")
