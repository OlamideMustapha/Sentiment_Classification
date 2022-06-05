# import os.path as osp
import os
import re


from tqdm import tqdm 
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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


import torch
from torch_geometric.data import Data, Dataset
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, EdgeConv, DynamicEdgeConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader