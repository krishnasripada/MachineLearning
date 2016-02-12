from csv import DictReader, DictWriter
from collections import defaultdict

import numpy as np
import nltk, string
import itertools
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.stem.lancaster import *
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords
import cPickle as pickle

def lancaster_stem(word):
    stemmer = LancasterStemmer()
    stem_word = stemmer.stem(word)
    if stem_word:
        return stem_word.lower()
    else:
        return word.lower()

class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words="english")
        self.filename = "feature.txt"

    def train_feature(self, examples):
        d_new = defaultdict(int)
        
        """
        d = self.features(''.join(examples))
        with open(self.filename,'wb') as fp:
            pickle.dump(d,fp)
        with open(self.filename,'rb') as fp:
            d = pickle.load(fp)
        fp.close()
        """
        d = self.features(''.join(examples))
        for i,j in d.items():
            if isinstance(i, str):
                d_new.update({i:j})
            else:
                d_new.update({''.join(i):j})
      
        return self.vectorizer.fit_transform(d_new)
    
    def features(self, text):
        d = defaultdict(int)
        tokenize = self.vectorizer.build_tokenizer()
        for ii in tokenize(text):
            d[lancaster_stem(ii)]+=1
        d.update(self.bigram_feature(text))
        return d
    
    def bigram_feature(self, text):
        bigram_colloc_finder = BigramCollocationFinder.from_words(text)
        bigrams = bigram_colloc_finder.nbest(BigramAssocMeasures.chi_sq,100)
        return dict([(bigram, True) for bigram in itertools.chain(text, bigrams)])

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            top10 = np.argsort(classifier.coef_[i])[-10:]
            print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    """
    import argparse
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--subsample', type=float, default=1.0, help="subsample this amount")
    args = parser.parse_args()
    """
    # Cast to list to keep it all in memory
    train = list(DictReader(open("train.csv", 'r')))
    test = list(DictReader(open("test.csv", 'r')))

    feat = Featurizer()

    labels = []
    for line in train:
        """
        if args.subsample<1.0 and int(line['id']) %100 > 100 * args.subsample:
            continue
        """
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
