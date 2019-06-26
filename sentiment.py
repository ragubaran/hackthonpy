import os
import json
import spacy
import pandas as pd
import string
import nltk
import datetime
from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#function
def spacy_preprocessor(sentence):
    sentence = sentence.strip().replace("\n", " ").replace("\r", " ")
    sentence = sentence.lower()
    text = parser(sentence)
    text = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in text ]
    text = [ word for word in text if word not in stopwords and word not in punctuations ]
    text = " ".join([i for i in text])
    return text

# objects
nlp = spacy.load('en')
stopwords = list(STOP_WORDS)
punctuations = string.punctuation
analyser = SentimentIntensityAnalyzer()
parser = English()

# data read
input_file = open ('data.json')
json_array = json.load(input_file)
sentences = []
docs = []

# print data
print("data")
for item in json_array:
    title = item["title"]
    date_time_obj = datetime.datetime.strptime(item["creation_date"], '%Y-%m-%d %H:%M:%S.%f %Z')
    print(date_time_obj.year)
    print(title)
    docs.append(spacy_preprocessor(title))
    sentences.append(title)

print("vectorize")
vectorizer = CountVectorizer(max_df=0.8,stop_words=stopwords, max_features=1000000, ngram_range=(1,3))
vectorizer.fit(docs)
bag_of_words = vectorizer.transform(docs)
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in
              vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1],
                    reverse=True)
print(words_freq)

print("sentences")
print(sentences)
scores =[]

print("scoring")
for sentence in sentences:
    score = analyser.polarity_scores(sentence)
    print(sentence)
    print(score)
    blob1 = TextBlob(sentence)
    print(format(blob1.sentiment))
    scores.append(score)
