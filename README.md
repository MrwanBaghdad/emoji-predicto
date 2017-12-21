# Multilingual Emoji Prediction

## Installation
```
sudo pip3 install -r requirements.txt

sudo python3 -m nltk.downloader -d /usr/local/share/nltk_data stopwords wordnet
```

### NER
To use Named Entity Recognition, additional NLTK datasets need to be installed:
```
sudo python3 -m nltk.downloader -d /usr/local/share/nltk_data punkt averaged_perceptron_tagger maxent_ne_chunker
```

## Run
- Execute data cleaning script to generate cleaned tweets and labels files.
- Execute embedding script to generate fasttext model
- Create your classification model