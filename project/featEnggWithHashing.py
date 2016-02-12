from csv import DictReader, DictWriter
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from time import gmtime, strftime
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict
from collections import Counter
from math import sqrt
import ast
from nltk.corpus import wordnet as wn
from itertools import product
import nltk
from pattern.en import tag
from nltk.stem.lancaster import LancasterStemmer
import itertools

stemmer = LancasterStemmer()

class Featurizer:
    def __init__(self):
        self.vectorizer = HashingVectorizer(stop_words="english")

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)


if __name__ == "__main__":
    print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--accuracy', type=bool, default=False, help='test accuracy')
    args = parser.parse_args()

    feat = Featurizer()

    questions = list(DictReader(open("questions.csv", 'rU')))
    train = list(DictReader(open("train.csv", 'rU')))
    test = list(DictReader(open("test.csv", 'rU')))
    ind = 0

    features = []
    position = []
    x_train = []
    cv_test = []
    cv_train = []
    cv_y_test = []
    cv_y_train = []
    y = []
    for x in questions:
        features.append(x['questionText'])
        features.append(len(x['questionText']))
        features.append(x['cat'])
        features.append(len(x['answer']))

    #############Training data##########
    for x in train:

        features.append(x['user'])
        position.append(float(x['position']))

        if args.accuracy==False:
            x_train.append(features)
            y.append(float(x['position']))
        else:
            if ind%5 == 0:
                cv_test.append(features)
                cv_y_test.append(float(x['position']))
            else:
                cv_y_train.append(float(x['position']))
                cv_train.append(features)
            ind+=1
    
    ############Test set#############
    #print ' '.join(' '.join(cv_train))
    cv_train = feat.train_feature(" ".join(list(itertools.chain(*cv_train))))
    cv_test = feat.test_feature(" ".join(list(itertools.chain(*cv_test))))

    print cv_train
    print cv_test


