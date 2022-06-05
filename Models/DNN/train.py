from libs import *
from data import *
from model import *


model = dl.train (
  x=predictors,
  y=response,
  training_frame=train,
  validation_frame=valid
)


history = model.score_history ()


history_fig = make_subplots (rows=1, cols=2, subplot_titles=['Loss','AUC'])

history_fig.add_trace (go.Scatter (y=history['training_logloss'], name='Loss', legendgroup='Loss', legendgrouptitle=dict (text='Model Loss')), 1, 1)
history_fig.add_trace (go.Scatter (y=history['validation_logloss'], name='Validation loss', legendgroup='Loss'), 1,1 )
history_fig.add_trace (go.Scatter (y=history['training_auc'], name='AUC', legendgroup='AUC', legendgrouptitle=dict (text='Model AUC')), 1, 2)
history_fig.add_trace (go.Scatter (y=history['validation_auc'], name='Validation AUC', legendgroup='AUC'),1, 2)

history_fig.update_layout (height=520, width=1500, colorway=px.colors.sequential.Tealgrn_r)


predictions = dl.predict (test)



# y_true = test_split ['Sentiment'].values
y_true = y_test
y_pred = predictions ['predict'].as_data_frame ().values

# y_true = y_true.flatten ().tolist ()
y_pred = y_pred.flatten ().tolist ()

cm = metrics.confusion_matrix (y_pred=y_pred, y_true=y_true, labels=['Positive', 'Negative', 'Neutral'])

y = ['Neutral', 'Negative', 'Positive']
x = ['Positive', 'Negative', 'Neutral']

if not os.path.exists ("./results"):
  os.mkdir ("./results")

history_fig.write_image ('./results/ffnn_train_history.png')

cm_fig = ff.create_annotated_heatmap (np.flipud (cm.T), y=y, x=x, reversescale=True, colorscale='Teal_r')
cm_fig.update_layout (width=600)
cm_fig.write_image ('./results/ffnn_confusion_matrix.png')

print (metrics.classification_report (y_true, y_pred))

variable_importance = model.varimp (use_pandas=True).head (10).sort_values (by='relative_importance')
var_importance_fig = px.bar (variable_importance, y="variable", x="relative_importance", color="relative_importance", template="plotly_white", orientation='h', title="Variable Importance", width=1500, height=600)
var_importance_fig.update_layout (showlegend=False)

var_importance_fig.write_image ('./results/ffnn_variable_importance.png')