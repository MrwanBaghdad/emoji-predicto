import pickle

import numpy as np
from lib import construct_data_matrix, split_data
from lib.data_paths import *
from scorer_semeval18 import main as eval
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

tokenized_tweets = pickle.load(open(TOK_TWEETS_PATH, 'rb'))
print('loaded tweets')

data_matrix = construct_data_matrix(tokenized_tweets)
print('constructed data matrix')
print('Dim:', data_matrix.shape)
print('Density:', np.count_nonzero(data_matrix) / np.size(data_matrix))

labels = np.asarray(open(CLEAN_LABELS_PATH).read().splitlines())
data_train, data_test, labels_train, labels_test = split_data(data_matrix, labels)
print('split data')

bern = BernoulliNB()
bern.fit(data_train, labels_train)
print("\nbern", bern.score(data_test, labels_test))
eval(labels_test, bern.predict(data_test))

multi = MultinomialNB()
multi.fit(data_train + abs(np.min(data_train)), labels_train)
print("\nmulti", multi.score(data_test + abs(np.min(data_test)), labels_test))
eval(labels_test, multi.predict(data_test))

tree = DecisionTreeClassifier(max_depth=10)
tree.fit(data_train, labels_train)
print("\ntree", tree.score(data_test, labels_test))
eval(labels_test, tree.predict(data_test))

clf = RandomForestClassifier(max_depth=3)
clf.fit(data_train, labels_train)
print("\nrandomforest depth=3", clf.score(data_test, labels_test))
eval(labels_test, clf.predict(data_test))

clf = RandomForestClassifier(max_depth=5)
clf.fit(data_train, labels_train)
print("\nrandomforest depth=3", clf.score(data_test, labels_test))
eval(labels_test, clf.predict(data_test))