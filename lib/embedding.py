import fasttext
from os.path import abspath

model = fasttext.skipgram(abspath('../data/cleaned_tweets'), abspath('../data/model'))
print(len(model.words))
