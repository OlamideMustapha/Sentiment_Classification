import os

import functools
from collections import Counter
from IPython.display import display
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

import nltk

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam


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


warnings.filterwarnings ('ignore')