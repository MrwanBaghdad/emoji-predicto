import fasttext

from lib.data_paths import *

model = fasttext.skipgram(CLEAN_TWEETS_PATH, WORD_EMB_MODEL_PATH)
print('Vocab Size:', len(model.words))
