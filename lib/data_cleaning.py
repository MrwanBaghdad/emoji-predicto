# coding: utf-8

import re
from xml.sax.saxutils import unescape

from nltk.corpus import stopwords

from lib.word_segmentation import segment

# Regex's
USER_regex = re.compile(r'@user')
NUMBERS_regex = re.compile(r'[\u0030-\u0039]+')
DOTS_regex = re.compile(r'\.{2,}')  # 2 dots or more
UNICODE_DOTS_regex = re.compile(r'…')
SPACES_regex = re.compile(r'\s{2,}')  # 2 spaces or more
UNICODE_SPACES_regex = re.compile(r'️')
SYMBOLS_regex = re.compile(r'[@\-ـ_:#/"\'%\[\]\n&]')

# URL regex source: http://www.noah.org/wiki/RegEx_Python
URL_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

stopwords = stopwords.words('english')
stopwords = ['\\b' + x.strip() + '\\b' for x in stopwords if x.strip() != '']
stopwords = '|'.join(stopwords)
STOP_WORDS_regex = re.compile(stopwords)


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
    cleaned_text = DOTS_regex.sub('.', original_text)
    cleaned_text = UNICODE_DOTS_regex.sub('.', cleaned_text)
    cleaned_text = SYMBOLS_regex.sub(' ', cleaned_text)
    cleaned_text = UNICODE_SPACES_regex.sub(' ', cleaned_text)

    return cleaned_text


# noinspection PyShadowingNames
def process_tweet(tweet):
    tweet = unescape(tweet)
    tweet = remove_url(tweet)
    tweet = process_hashtags(tweet)
    tweet = remove_user_keyword(tweet)
    tweet = remove_stop_words(tweet)

    tweet = NUMBERS_regex.sub('', tweet)
    tweet = remove_symbols(tweet)
    tweet = SPACES_regex.sub(' ', tweet)  # must be final
    return tweet.strip()


if __name__ == '__main__':
    for tweet in open('../data/text'):
        print(tweet.replace('\n', ''))
        print(process_tweet(tweet))
        print()
