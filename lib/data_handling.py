import pickle

import fasttext
import numpy as np
from sklearn.model_selection import train_test_split

# noinspection PyUnresolvedReferences
model = fasttext.load_model('data/model.bin')


# noinspection PyShadowingNames
def construct_data_matrix(tweets, max_word_count):
    data_matrix = []

    for idx, tweet in enumerate(tweets):
        tweet_vector = []
        for word in tweet:
            tweet_vector.append(model[word])

        tweet_vector = np.ravel(tweet_vector)
        padding_len = max_word_count * 100 - len(tweet_vector)  # length of any word vector = 100
        tweet_vector = np.concatenate((tweet_vector, np.zeros(padding_len)))

        data_matrix.append(tweet_vector)

    return data_matrix


def split_data(data, labels):
    return train_test_split(data, labels, test_size=0.3)


if __name__ == '__main__':
    tokenized_tweets = pickle.load(open('../data/tokenized_tweets.pic', 'rb'))
    tweet_of_max_length = max(tokenized_tweets, key=len)

    data_matrix = construct_data_matrix(tokenized_tweets, len(tweet_of_max_length))
    labels = open('../data/maro_labels').read().splitlines()

    data_train, data_test, labels_train, labels_test = split_data(data_matrix, labels)

    print(len(data_train), len(labels_train))
    print(len(data_test), len(labels_test))
