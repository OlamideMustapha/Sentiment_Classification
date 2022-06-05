from libs import *


#
#
#### READ VECTORIZED DOCUMENTS
#
#
tweet_df = pd.read_csv ('../../Data/preprocessed/NG_TWEETS_TFIDF.csv')


## SPLITTING DATA TO TRAIN AND TEST SETS, WITH A PROPORTION TO .75 TO .25
predictors = [col for col in tweet_df.columns if col != 'Sentiment']
response   = 'Sentiment'


resampler = SMOTE (sampling_strategy='not majority', random_state=1234)
X_resample, y_resample = resampler.fit_resample (tweet_df [predictors], tweet_df [response])


X_train, X_test, y_train, y_test = train_test_split (X_resample, y_resample, stratify=y_resample, train_size=0.75, random_state=1234)


train_frame  = h2o.H2OFrame (pd.concat ([y_train, X_train], axis=1))
train, valid = train_frame.split_frame (ratios = [.75], destination_frames=['train_tfidf_frame', 'valid_tfidf_frame'],seed = 1234)
test         = h2o.H2OFrame (X_test, destination_frame='test_frame')