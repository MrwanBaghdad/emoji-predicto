from gensim.models import Word2Vec
import numpy as np
#load data
tweets  = [i[:-1] for i in open("../data/cleaned_tweets").readlines()[:300]]
tweets_tokenized = [t.lower().split(' ') for t in tweets]
print(tweets_tokenized[0])
model = Word2Vec(tweets_tokenized)

#create vectors min_max



def tweet_to_vector_iter():
    for one_tweet_tokenized in tweets_tokenized:
        temp_matrix = np.ndarray([len(one_tweet_tokenized), model.vector_size])
        for matrix_index, word in zip(range(len(one_tweet_tokenized)), one_tweet_tokenized):
            try:
                temp_matrix[matrix_index] = model[word]
            except KeyError as err  :
                pass
        assert len(one_tweet_tokenized) * model.vector_size == temp_matrix.size 
        yield temp_matrix

def min_max_rep(temp_matrix):
    return np.concatenate((
        np.min(temp_matrix,axis=0), np.max(temp_matrix,axis=0)
    ))

def flatten_rep_iter():
    #TODO padding
    for vec in tweet_to_vector_iter():
        yield vec.flatten()
    

if __name__ == "__main__":
    for i in tweet_to_vector_iter():
        assert i.size = model.vector_size * 2
    