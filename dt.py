import pickle

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

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

clf = DecisionTreeClassifier(max_depth=10)
clf.fit(data_train, labels_train)
scores = cross_val_score(clf, data_matrix, labels, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print('trained decision tree')

print(clf.score(data_test, labels_test))

score = clf.predict(data_test)

# f = open('english.output.txt', 'w')
# for s in score:
#     f.write(s + '\n')
# f.close()

main(labels_test, score)
