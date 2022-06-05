
import model
from model import lstm
from libs import *
from data import *

with tf.device ('/cpu:0'):
  history = lstm.fit (x=X_train, y=y_train, batch_size=512, epochs=50, validation_split=0.25)


with tf.device ('/cpu:0'):
  scores = lstm.evaluate (X_test, y_test, batch_size=512, verbose=0)
  print ("Loss: %.2f\nAUC: %.2f%%\nAccuracy: %.2f%%" % (scores[0], scores[1] * 100, scores[2] * 100))


history_fig = make_subplots (rows=1, cols=3, subplot_titles=['Loss','Area Under The Curve', 'Accuracy'])

history_fig.add_trace (go.Scatter (y=history.history['loss'], name='Loss', legendgroup='Loss', legendgrouptitle=dict (text='Model Loss')), 1, 1)
history_fig.add_trace (go.Scatter (y=history.history['val_loss'], name='val loss', legendgroup='Loss'), 1,1 )
history_fig.add_trace (go.Scatter (y=history.history['auc'], name='AUC', legendgroup='AUC', legendgrouptitle=dict (text='Model AUC')), 1, 2)
history_fig.add_trace (go.Scatter (y=history.history['val_auc'], name='Val AUC', legendgroup='AUC'),1, 2)
history_fig.add_trace (go.Scatter (y=history.history['accuracy'], name='Accuracy', legendgroup='Accuracy', legendgrouptitle=dict (text='Model Accuracy')), 1, 3)
history_fig.add_trace (go.Scatter (y=history.history['val_accuracy'], name='Val Accuracy', legendgroup='Accuracy'),1, 3)

history_fig.update_layout (width=1500, height=520, colorway=px.colors.sequential.Tealgrn_r)

if not os.path.exists ("./results"):
  os.mkdir ("./results")

history_fig.write_image ('./results/lstm_train_history.png')


with tf.device ('/cpu:0'):
  y_pred = lstm.predict (X_test)
  y_pred = list (map (
    lambda x: keys[x],
    [np.argmax (i) for i in y_pred]
    ))


y_true = [np.argmax (i) for i in y_test.values]
y_true = list (map (lambda x: keys[x] , y_true))

print (Counter (y_true))
print (Counter (y_pred))

y = ['Neutral', 'Negative', 'Positive']
x = ['Positive', 'Negative', 'Neutral']

cm = confusion_matrix (y_pred=y_pred, y_true=y_true, labels=x)

cm_fig = ff.create_annotated_heatmap (np.flipud (cm.T), y=y, x=x, reversescale=True, colorscale='Teal_r')
cm_fig.update_layout (width=600)
cm_fig.write_image ('./results/lstm_confusion_matrix.png')

print (classification_report (y_true, y_pred))