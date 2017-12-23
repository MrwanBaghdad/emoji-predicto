import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from lib import construct_data_matrix, split_data
from lib.data_paths import *
from scorer_semeval18 import main

tokenized_tweets = pickle.load(open(TOK_TWEETS_PATH, 'rb'))
print('loaded tweets')

data_matrix = construct_data_matrix(tokenized_tweets)
print('constructed data matrix')
print('Dim:', data_matrix.shape)
print('Density:', np.count_nonzero(data_matrix) / np.size(data_matrix))

labels = np.asarray(open(CLEAN_LABELS_PATH).read().splitlines())
data_train, data_test, labels_train, labels_test = split_data(data_matrix, labels)
print('split data')

estimators = [10, 50, 100]
depths = [5, 10, 15, 20]

# for estimator in estimators:
# for depth in depths:
clf = RandomForestClassifier(max_depth=3)
clf.fit(data_train, labels_train)

score = clf.predict(data_test)

main(labels_test, score)
