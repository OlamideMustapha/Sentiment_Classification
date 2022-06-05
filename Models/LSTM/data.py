from libs import *


tweet_df = pd.read_excel ('../../Data/Preprocessed/NG_ELECTION_TWEETS_CLEANED.xlsx')

X = tweet_df.drop ('Sentiment', axis=1)
y = tweet_df [['Sentiment']]

vocab = set ()
X = tweet_df.drop ('Sentiment', axis=1)
for x, y in tqdm (X.iterrows (), total=X.shape[0]):
  vocab = vocab.union (set (y.values[0].split (' ')))

vocab_size    = len (vocab)
embedding_dim = 128
max_length    = len (tweet_df['cleaned_tweet'].max ())
# max_length    = 50
padding_type  = 'post'
trunc_type    = 'post'

X = tweet_df.drop ('Sentiment', axis=1)
y = tweet_df [['Sentiment']]


tokenizer = Tokenizer (num_words=vocab_size, oov_token= ' ')
tokenizer.fit_on_texts (X['cleaned_tweet'].values)

word_index = tokenizer.word_index
X = tokenizer.texts_to_sequences (X['cleaned_tweet'].values)
X = pad_sequences (X, padding='post', maxlen=max_length)

X_train, X_test, y_train, y_test = train_test_split (X, y, stratify=y, train_size=0.75, random_state=1234)


resampler = SMOTE (sampling_strategy='not majority', random_state=1234)

X_train, y_train = resampler.fit_resample (X_train, y_train)
X_test, y_test   = resampler.fit_resample (X_test, y_test)

y_train = pd.get_dummies (y_train, prefix='', prefix_sep='')
y_test  = pd.get_dummies (y_test, prefix='', prefix_sep='')

keys = y_train.keys ()