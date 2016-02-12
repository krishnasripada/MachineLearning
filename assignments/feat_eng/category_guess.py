from csv import DictReader, DictWriter
from collections import defaultdict
from collections import Counter

import numpy as np
import argparse
import nltk
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.tokenize import WordPunctTokenizer

kTOKENIZER = WordPunctTokenizer()

def tokenize(text):
    d = defaultdict(int)
    tokens = kTOKENIZER.tokenize(text)
    for word in tokens:
        d[word]+=1
    d[ngrams(tokens,2)]+=1
    return d

def ngrams(lst, n):
    tlst = lst
    while True:
        a, b = tee(tlst)
        l = tuple(islice(a, n))
        if len(l) == n:
            yield l
            next(b)
            tlst = b
        else:
            break

class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(tokenizer = tokenize, stop_words = "english")

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            top10 = np.argsort(classifier.coef_[i])[-10:]
            print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":
    
    # Cast to list to keep it all in memory
    train = list(DictReader(open("train.csv", 'r')))
    test = list(DictReader(open("test.csv", 'r')))
    
    feat = Featurizer()
    
    labels = []
    for line in train:
        if not line['cat'] in labels:
            labels.append(line['cat'])

    x_train = feat.train_feature(x['text'] for x in train)
    x_test = feat.test_feature(x['text'] for x in test)
    
    y_train = array(list(labels.index(x['cat']) for x in train))
    
    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)
    
    feat.show_top10(lr, labels)
    
    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)

