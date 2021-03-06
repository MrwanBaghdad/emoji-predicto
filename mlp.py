import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier

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

clf = MLPClassifier(max_iter=200, verbose=True, alpha=0.001)
clf.fit(data_matrix, labels)

score = clf.predict(data_matrix)

f = open('english.output.txt', 'w')
for s in score:
    f.write(s + '\n')
f.close()

main(labels, score)
