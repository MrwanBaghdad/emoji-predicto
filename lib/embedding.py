import fasttext
try:
    from lib.data_paths import *
except ImportError as err:
    from data_paths import *


# model = fasttext.skipgram(CLEAN_TWEETS_PATH, WORD_EMB_MODEL_PATH)
# print('Vocab Size:', len(model.words))


# from gensim.models import Word2Vec
# sentences = open(CLEAN_TWEETS_PATH).readlines()
# sentences = [s[:-1].split(' ') for s in sentences]
# w2v_model = Word2Vec(sentences, size=100, window=10, min_count=3)
# w2v_model.save(WORD_EMB_MODEL_PATH) 


from nltk.corpus import twitter_samples 
import data_cleaning 

strings = twitter_samples.strings(twitter_samples.fields())

strings = map(
    strings,
    [
        data_cleaning.remove_url,
        data_cleaning.remove_non_english,
        data_cleaning.remove_stop_words
    ]
)

from gensim.models.word2vec import LineSentence

# train wikipedia corpus 
# model = Word2Vec.load_word2vec_format('')
#train using twitter + sentiement

EPOCH = 2

for i in range(EPOCH):
    for tweet, sentiment in zip(LineSentence(CLEAN_TWEETS_PATH), open(CLEAN_LABELS_PATH).readlines()):
        model.train(tweet.join( sentiment))
        # print(tweet[:-1] + " " + sentiment)

# model.save(WORD_EMB_MODEL_PATH)
