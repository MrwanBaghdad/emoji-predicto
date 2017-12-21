import pickle

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from lib import construct_data_matrix, split_data

tokenized_tweets = pickle.load(open('data/tokenized_tweets.pic', 'rb'))
print('loaded tweets')

data_matrix = construct_data_matrix(tokenized_tweets)
print('constructed data matrix')
print('Dim:', data_matrix.shape)
print('Density:', np.count_nonzero(data_matrix) / np.size(data_matrix))

labels = np.asarray(open('data/cleaned_labels').read().splitlines())
data_train, data_test, labels_train, labels_test = split_data(data_matrix, labels)
print('split data')

clf = DecisionTreeClassifier(max_depth=10)
clf.fit(data_train, labels_train)
print('trained model')

score = clf.score(data_test, labels_test)
print(score)
