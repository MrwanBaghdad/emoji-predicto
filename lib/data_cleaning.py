# coding: utf-8

import re
import string
from time import time
from xml.sax.saxutils import unescape

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from lib.word_segmentation import segment

# Regex's
USER_regex = re.compile(r'@user')
NUMBERS_regex = re.compile(r'[\u0030-\u0039]+')
DOTS_regex = re.compile(r'((\s*)?\.(\s*)?){2,}')  # 2 dots or more
UNICODE_DOTS_regex = re.compile(r'…|️…|・・・')
SPACES_regex = re.compile(r'\s{2,}')  # 2 spaces or more
EXCLAMATION_regex = re.compile(r'!{2,}')
UNICODE_SPACES_regex = re.compile(r'️')
SYMBOLS_regex = re.compile(r'[' + string.punctuation + '’¿“”—•▃¯ツ]')
NONENGLISH_regex = re.compile(r'[^a-zA-Z\s.]')

# URL regex source: http://www.noah.org/wiki/RegEx_Python
URL_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

stopwords = set(stopwords.words('english')) - {'no', 'not'}
STOP_WORDS_regex = '\\b(' + '|'.join(stopwords) + ')\\b'
STOP_WORDS_regex = re.compile(STOP_WORDS_regex)

# Load NLTK classes
tokenizer = TweetTokenizer(reduce_len=True)
lemm = WordNetLemmatizer()
stemmer = SnowballStemmer('english')  # not used yet


def remove_user_keyword(original_text):
    return USER_regex.sub('', original_text)


def remove_stop_words(original_text):
    return STOP_WORDS_regex.sub('', original_text)


# noinspection SpellCheckingInspection
def process_hashtags(original_text):
    cleaned_text = original_text
    hashtags = set(re.findall(r"#(\w+)", original_text))
    for hashtag in hashtags:
        cleaned_text = cleaned_text.replace('#', ' #')
        cleaned_text = cleaned_text.replace(hashtag, ' '.join(segment(hashtag)))

    return cleaned_text


def remove_url(original_text):
    return URL_regex.sub(' ', original_text)


def remove_symbols(original_text):
    cleaned_text = EXCLAMATION_regex.sub('!', original_text)
    cleaned_text = UNICODE_DOTS_regex.sub('.', cleaned_text)
    cleaned_text = DOTS_regex.sub('.', cleaned_text)
    cleaned_text = SYMBOLS_regex.sub(' ', cleaned_text)
    cleaned_text = UNICODE_SPACES_regex.sub(' ', cleaned_text)

    return cleaned_text


def pad_dot(original_text):
    return original_text.replace('.', ' . ')


def remove_non_english(original_text):
    return NONENGLISH_regex.sub(' ', original_text)


# noinspection PyShadowingNames
def clean_tweet(tweet):
    tweet = unescape(tweet)
    tweet = remove_url(tweet)
    tweet = process_hashtags(tweet)
    tweet = remove_user_keyword(tweet)
    tweet = remove_stop_words(tweet)

    tweet = NUMBERS_regex.sub('', tweet)
    tweet = remove_symbols(tweet)
    tweet = remove_non_english(tweet)
    tweet = pad_dot(tweet)
    tweet = SPACES_regex.sub(' ', tweet)  # must be final
    return tweet.strip()


# noinspection PyShadowingNames
def tokenize_tweet(tweet):
    tweet = tokenizer.tokenize(tweet)

    tweet = [token for token in tweet
             if token.lower() not in stopwords]

    tweet = [lemm.lemmatize(word) for word in tweet]

    return tweet


# noinspection PyShadowingNames
def process_tweet(tweet):
    tweet = clean_tweet(tweet)
    tweet = tokenize_tweet(tweet)

    return tweet


# noinspection PyShadowingNames
def process_tweets(tweets):
    _tweets = list()
    for tweet in tweets:
        _tweets.append(process_tweet(tweet))

    return _tweets


def clean_tweets(tweets):
    _tweets = list()
    for tweet in tweets:
        _tweets.append(clean_tweet(tweet))
    return _tweets


if __name__ == '__main__':
    tweets = open('../data/maro_text').read().splitlines()

    start = time()
    tweets = process_tweets(tweets)

    print('Time: %.3f s' % (time() - start))
