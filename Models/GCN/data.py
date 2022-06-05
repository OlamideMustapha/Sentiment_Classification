from libs import *

embedding_df = pd.read_csv ('../../Data/Preprocessed/NODE_EMBEDDING.csv', na_values=None, na_filter=None)

def get_embeddings (x):
  x = re.sub (r'\[|\]|', '', x)
  x = x.split (', ')
  return [float (i) for i in x]



class GraphDataset(Dataset):
  def __init__(self, root, raw_file, transform=None, pre_transform=None, pre_filter=None):
    """
    root: is a directory where the data set will be stored
    """
    self.raw_file = raw_file
    super().__init__(root, transform, pre_transform, pre_filter)

  @property
  def raw_file_names (self):
    return self.raw_file
  
  @property
  def processed_file_names (self):
    # not implemented
    """ 
    if files are found in the raw_dir, processing is skipped
    """
    return 'Not Implemented'
    
  def download (self):
    # not implemented
    pass

  def process(self):
    self.data = pd.read_excel (self.raw_paths[0])
    
    factorize = self.data.Sentiment.factorize ()
    self.data['Sentiment'] = factorize[0]
    keys = factorize[1]

    for index, tweet in tqdm (self.data.iterrows (), total=self.data.shape[0]):
      text = tweet ['cleaned_tweet']
      node_features         = self._get_node_features (text)
      edge_index, edge_attr = self._get_adjacency_info (text)
      graph_label           = tweet['Sentiment']

      torch.save(
        Data (x=node_features, edge_index=edge_index, y=graph_label),
        os.path.join(self.processed_dir, f'data_{index}.pt'))

  def _get_node_features (self, text):
    """ Maps all words in a text into their respective node embedding
        Return: a 2D array
    """
    x = [
      get_embeddings (embedding_df.embedding [embedding_df.name == word].values[0])
      for word in text.split (" ")
    ]

    return torch.as_tensor (x, dtype=torch.float)
    
  def _get_adjacency_info (self, text):
    text_seq  = text.split (" ")
    edges     = []
    edge_attr = []
    word_dict = {}

    for k, v in enumerate (text_seq):
      if v not in word_dict.keys ():
        word_dict [v] = k
      
    for idx, word in enumerate (text_seq):
      if idx != 0:
        previous_word = text_seq [idx - 1]
        forward  = [word_dict [previous_word], word_dict [word]]
        backward = [word_dict [word], word_dict [previous_word]]

        if forward in edges and backward in edges:
          for_index  = edges.index (forward)
          back_index = edges.index (backward)
          edge_attr [for_index] = [edge_attr [for_index] [0] + 1]
          # edge_attr [back_index] = [edge_attr [back_index] [0] + 1]
          
        else:
          edge_attr.append ([1])
          # edge_attr.append ([1])
          edges.append (forward)
          # edges.append (backward)


    return torch.tensor(edges, dtype=torch.long).t().contiguous(), edge_attr

  def len(self):
    return self.data.shape[0]


  def get(self, idx):
    data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
    return data



dataset = GraphDataset (
  root='../../Data/Preprocessed/data',
  raw_file='NG_ELECTION_TWEETS_CLEANED.xlsx'
)


dataset = dataset.shuffle()

train_dataset = dataset[:round (len(dataset) * 0.75)]
test_dataset =  dataset[round (len(dataset) * 0.75):]


train_loader = DataLoader (train_dataset, batch_size=512, shuffle=False)
test_loader  = DataLoader (test_dataset,  batch_size=512, shuffle=False)