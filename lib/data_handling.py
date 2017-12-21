import fasttext
import numpy as np

from lib import process_tweets

# noinspection PyUnresolvedReferences
model = fasttext.load_model('../data/model.bin')

padding_vector = np.array([0 for _ in range(10)])


def pad_tweet(tweet, max_word_count):
    for i in range(len(tweet), max_word_count):
        tweet.append(padding_vector)

    return tweet


# noinspection PyShadowingNames
def construct_data_matrix(tweets, max_word_count):
    data_matrix = []

    for tweet in tweets:
        tweet_vector = []
        for word in tweet:
            word_vector = model[word]
            tweet_vector.append(word_vector)

        if len(tweet) < max_word_count:
            tweet_vector = pad_tweet(tweet_vector, max_word_count)

        data_matrix.append(
            np.ravel(tweet_vector)
        )

    return np.asmatrix(data_matrix)


if __name__ == '__main__':
    tokenized_tweets = process_tweets(open('../data/maro_text').read().splitlines())
    tweet_of_max_length = max(tokenized_tweets, key=len)

    data_matrix = construct_data_matrix(tokenized_tweets, len(tweet_of_max_length))
    print(data_matrix.shape)
