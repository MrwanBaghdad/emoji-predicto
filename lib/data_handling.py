import pickle
from os.path import abspath
import fasttext
import numpy as np
from sklearn.model_selection import train_test_split

# noinspection PyUnresolvedReferences
model = fasttext.load_model(abspath('../data/model.bin'))


# ref: https://stats.stackexchange.com/a/239071
# noinspection PyShadowingNames
def construct_data_matrix(tweets):
    data_matrix = []

    for tweet in tweets:
        tweet_matrix = np.ndarray([len(tweet), model.dim])

        for matrix_idx, word in zip(range(len(tweet)), tweet):
            tweet_matrix[matrix_idx] = model[word]

        # concatenate min and max of every feature in the tweet
        tweet_vector = np.concatenate((
            np.min(tweet_matrix, axis=0),
            np.max(tweet_matrix, axis=0)
        ))

        data_matrix.append(tweet_vector)

    return np.asmatrix(data_matrix)


def split_data(data, labels):
    return train_test_split(data, labels, test_size=0.3)


if __name__ == '__main__':
    tokenized_tweets = pickle.load(open('../data/tokenized_tweets.pic', 'rb'))
    tweet_of_max_length = max(tokenized_tweets, key=len)

    data_matrix = construct_data_matrix(tokenized_tweets)
    labels = open('../data/cleaned_labels').read().splitlines()

    data_train, data_test, labels_train, labels_test = split_data(data_matrix, labels)

    print(len(data_train), len(labels_train))
    print(len(data_test), len(labels_test))
