# coding: utf-8

from functools import wraps
# import nltk
# WORDS = set(nltk.corpus.brown.words())
WORDS = set(open('../data/words.txt').read().splitlines())


def memoize(f):
    """
    A hacky memoize method decorator.
    source: https://coderwall.com/p/5frh7g/python-memoization-decorator
    """

    def _c(*args, **kwargs):
        if not hasattr(f, 'cache'):
            f.cache = dict()
        key = (args, tuple(kwargs))
        if key not in f.cache:
            f.cache[key] = f(*args, **kwargs)
        return f.cache[key]

    return wraps(f)(_c)


# @memoize
def segment(sentence):
    """
    Segment a string of chars using the brown nltk corpus.
    Keeps longest possible words found in `WORDS`.

    :param sentence: The sentence to segment.
    :return Sentence segmentation.
    """
    words = []
    sentence = sentence.lower()

    # skip non alphabetic words
    if not sentence.isalpha():
        return [sentence]

    temp_sentence = sentence
    while temp_sentence:
        # longest segments possible
        for i in range(len(temp_sentence), 0, -1):
            _segment = temp_sentence[:i]
            if _segment in WORDS:
                words.append(_segment)
                temp_sentence = temp_sentence[i:]
                break
        else:  # no matching segments were found
            # return as is
            return [sentence]
    # return segmentation
    return words


if __name__ == '__main__':
    print(segment('acquirecustomerdata'))
    print(segment('tomorethanicanbe'))
    print(segment('balletdtroitsummerintensive'))
    print(segment('livethoughit'))
    print(segment('VivaLasVegas'))
    print(segment('LiveYourLife'))
    print(segment('LifeisShortLife'))
