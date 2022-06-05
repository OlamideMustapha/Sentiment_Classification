from data import *
from libs import *


lstm = Sequential ([
  Embedding (input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, mask_zero=True),
  SpatialDropout1D (0.4),
  LSTM (units=embedding_dim*2, dropout=0.2, recurrent_dropout=0.2, unroll=False, return_sequences=True),
  LSTM (units=64),
  Dense (units=3, activation='softmax')
])


lstm.compile (loss ='categorical_crossentropy', optimizer='adam', metrics = ['AUC', 'accuracy'])