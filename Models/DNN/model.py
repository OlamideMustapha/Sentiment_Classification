from libs import *
from data import *

### Feedforward artificial neural network (ANN) 
dl = H2ODeepLearningEstimator (
  model_id='FNN_tfidf',
  distribution="multinomial",
  hidden=[100, 100],
  epochs=50,
  train_samples_per_iteration=-1,
  reproducible=True,
  activation='rectifier',
  single_node_mode=False,
  balance_classes=True,
  force_load_balance=True,
  seed=23123,
  stopping_rounds=0,
  # stopping_metric='AUC',
  auc_type='weighted_ovr',
  # verbose=True,
)