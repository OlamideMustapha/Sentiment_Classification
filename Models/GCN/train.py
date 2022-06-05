from libs import *
from data import *
from model import GCN

gcn = GCN (hidden_channels=50)


optimizer = torch.optim.Adam (gcn.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss ()

def train (train_loader):
  gcn.train ()
  for data in train_loader:
    out = gcn (data.x, data.edge_index, data.batch)
    loss = criterion (out, data.y)
    loss.backward ()
    optimizer.step ()
    optimizer.zero_grad ()  

def test (loader):
  gcn.eval ()
  loss = 0
  correct = 0
  for data in loader:
    out = gcn (data.x, data.edge_index, data.batch)  
    pred = out.argmax (dim=1)
    correct += int ((pred == data.y).sum())
    loss = criterion (out, data.y)
    # loss.append (l_)
  return (correct / len (loader.dataset)), float (loss)


def fit (train_set, validation_size, batch_size, epoch):
  train_loader  = DataLoader (train_set [0:round (len (train_set) * validation_size)], batch_size=batch_size, shuffle=False)
  valid_loader  = DataLoader (train_set [round (len (train_set) * validation_size): ],  batch_size=batch_size, shuffle=False)

  accuracy        = []
  loss            = []
  valid_accuracy  = []
  valid_loss      = []
  
  epoch_ = list (range (1, epoch + 1))

  # for i in tqdm (epoch_, total=len (epoch_), colour='black'):
  for i in epoch_:
    train (train_loader)
    accuracy_, loss_             = test (train_loader)
    valid_accuracy_, valid_loss_ = test (valid_loader)

    print (f'Epoch: {i}/{epoch}, loss: loss {loss_:.4f} - accuracy: {accuracy_:.4f} - val_loss: {valid_loss_:.4f} - val_accuracy: {valid_accuracy_:.4f}')

    accuracy.append (accuracy_)
    loss.append (loss_)
    valid_accuracy.append (valid_accuracy_)
    valid_loss.append (valid_loss_)


  return pd.DataFrame ({'Epoch':epoch_, 'loss': loss, 'accuracy': accuracy, 'val_loss': valid_loss, 'val_accuracy': valid_accuracy})



history = fit (train_dataset, validation_size=.25, batch_size=512, epoch=170)

history_fig = make_subplots (rows=1, cols=2, subplot_titles=['loss', 'Accuracy'])

history_fig.add_trace (go.Scatter (y=history['loss'], name='Loss', legendgroup='Loss', legendgrouptitle=dict (text='Model Loss')), 1, 1)
history_fig.add_trace (go.Scatter (y=history['val_loss'], name='val loss', legendgroup='Loss'), 1,1 )
history_fig.add_trace (go.Scatter (y=history['accuracy'], name='Accuracy', legendgroup='Accuracy', legendgrouptitle=dict (text='Model Accuracy')), 1, 2)
history_fig.add_trace (go.Scatter (y=history['val_accuracy'], name='Test Accuracy', legendgroup='Accuracy'),1, 2)
history_fig.update_layout (width=1500, height=520, colorway=px.colors.sequential.Tealgrn_r)


if not os.path.exists ("./results"):
  os.mkdir ("./results")


history_fig.write_image ('./results/gcn_train_history.png')


test_loader = DataLoader (test_dataset, shuffle=False)

y_pred = []
y_true = []

for data in test_loader:
  out = gcn (data.x, data.edge_index, data.batch) 
  y_true.append (int (data.y))
  y_pred.append (int (out.argmax(dim=1)))



y = ['Neutral', 'Negative', 'Positive']
x = ['Positive', 'Negative', 'Neutral']


y_pred = list (map (lambda cat: x[cat] , y_pred))
y_true = list (map (lambda cat: x[cat] , y_true))

cm = confusion_matrix (y_pred=y_pred, y_true=y_true, labels=x)


cm_fig = ff.create_annotated_heatmap (np.flipud (cm.T), y=y, x=x, reversescale=True, colorscale='Teal_r')
cm_fig.update_layout (width=600)
cm_fig.write_image ('./results/gcn_confusion_matrix.png')

print (classification_report (y_true, y_pred))