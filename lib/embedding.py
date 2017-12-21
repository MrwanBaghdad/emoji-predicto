import fasttext

from lib import clean_tweets

tweets = clean_tweets(open('../data/maro_text').read().splitlines())
cleaned_tweets_file = open('../data/cleaned_tweets', 'w')

for tweet in tweets:
    cleaned_tweets_file.write(tweet + '\n')

cleaned_tweets_file.close()

model = fasttext.skipgram('../data/cleaned_tweets', '../data/model', dim=10)
print(len(model.words))
