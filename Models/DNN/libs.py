# ### Uncomment cell to install h2o and it dependencies

# # ** Dependencies **
# !pip3 install requests
# !pip3 install tabulate
# !pip3 install future

# ### The following command removes the H2O module for Python.
# !pip3 uninstall h2o

# ### Next, use pip to install this version of the H2O Python module.
# !pip3 install https://h2o-release.s3.amazonaws.com/h2o/rel-zorn/4/Python/h2o-3.36.0.4-py2.py3-none-any.whl

#
#### IMPORT LIBRARIES
#
#

import os
import h2o
from h2o.estimators import H2ODeepLearningEstimator

from IPython.display import display
from collections import Counter
import numpy as np
import pandas as pd

## Visualization
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.figure_factory as ff


# Set ploting defaults
px.defaults.height                  = 550
pio.templates.default               = "plotly_white" # pio.templates
px.defaults.template                = "plotly_white"
px.defaults.color_continuous_scale  = px.colors.sequential.Teal_r
px.defaults.color_discrete_sequence = px.colors.sequential.Teal_r

plotly_config = {
  "staticPlot": True,
  "scrollZoom": True,
  "displayModeBar": False,
  "editable": True,
}

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


h2o.init ()