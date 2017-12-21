import fasttext

model = fasttext.skipgram('../data/cleaned_tweets', '../data/model')
print(len(model.words))
