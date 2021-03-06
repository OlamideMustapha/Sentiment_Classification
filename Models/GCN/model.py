from libs import *
from data import dataset



class GCN (torch.nn.Module):
  def __init__ (self, hidden_channels):
    super (GCN, self).__init__ ()
    torch.manual_seed (12345)
    self.conv1 = GraphConv (dataset.num_node_features, hidden_channels)
    self.conv2 = GraphConv (hidden_channels, hidden_channels, aggr='mean')
    self.conv3 = GraphConv (hidden_channels, hidden_channels, aggr='mean')
    self.conv4 = GraphConv (hidden_channels, hidden_channels, aggr='mean')
    self.conv5 = GraphConv (hidden_channels, hidden_channels, aggr='mean')
    self.lin   = Linear (hidden_channels, 3)


  def forward (self, x, edge_index, batch):
    # 1. Obtain node embeddings 
    x = self.conv1 (x, edge_index)
    x = x.relu ()
    x = self.conv2 (x, edge_index)
    x = x.relu ()
    x = self.conv3 (x, edge_index)
    x = x.relu ()
    x = self.conv4 (x, edge_index)
    x = x.relu ()
    x = self.conv5 (x, edge_index)

    # 2. Readout layer
    x = global_mean_pool (x, batch)  # [batch_size, hidden_channels]

    # 3. Apply a final classifier
    x = F.dropout (x, p=0.5, training=self.training)
    x = x.relu ()
    x = self.lin (x)
    # x = torch.softmax (x)
    
    return x