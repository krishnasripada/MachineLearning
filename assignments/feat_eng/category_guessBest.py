from csv import DictReader, DictWriter
from collections import defaultdict
from collections import Counter

import numpy as np
import argparse
import nltk, string, itertools
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import LancasterStemmer
from nltk import pos_tag, word_tokenize
from itertools import tee, islice
from nltk.util import ngrams
from pattern.en import tag
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

kTOKENIZER = WordPunctTokenizer()

kTAGSET = ["", "JJ", "NN", "PP", "RB", "VB"]

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    d = defaultdict(int)
    tokens = kTOKENIZER.tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    #stemmer = LancasterStemmer()
    #stems = stem_tokens(tokens, stemmer)
    for word in tokens:
        d[word]+=1
    d[ngrams(tokens,2)]+=1
    #d.update(bigram_feature(text))
    #d.update(part_of_speech(text))
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

def bigram_feature(text):
    bigram_colloc_finder = BigramCollocationFinder.from_words(text)
    bigrams = bigram_colloc_finder.nbest(BigramAssocMeasures.chi_sq,100)
    return dict([(bigram, True) for bigram in itertools.chain(text, bigrams)])

def trigram_feature(text):
    trigram_colloc_finder = TrigramCollocationFinder.from_words(text)
    trigrams = trigram_colloc_finder.nbest(TrigramAssocMeasures.chi_sq,200)
    return dict([(trigram, True) for trigram in itertools.chain(text, trigrams)])

def part_of_speech(text):
    d = defaultdict(int)
    for word, tags in tag(text):
        if tags in kTAGSET:
            d[word]+=1
    return d


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
    
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--subsample', type=float, default=1.0, help='subsample this amount')

    args = parser.parse_args()
    
    feat = Featurizer()
    
    # Cast to list to keep it all in memory
    train = list(DictReader(open("train.csv", 'r')))
    test = list(DictReader(open("test.csv", 'r')))
    
    labels = []
    for line in train:
        if not line['cat'] in labels:
            labels.append(line['cat'])

    dev_train = []
    dev_train_text = []
    dev_test = []
    dev_test_text = []
    full_train_text=[]
    full_train=[]

    for ii in train:
        if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
            continue
    
        text = ii['text']
        
        full_train_text.append(text)
        full_train.append(ii['cat'])
        
        if int(ii['id']) % 5 == 0:
            
            dev_test.append((ii['cat']))
            dev_test_text.append(text)
        else:
            dev_train.append((ii['cat']))
            dev_train_text.append(text)

    print ("Dev : " + str(len(dev_train)) + " - " + str(len(dev_train_text)))
    print ("Test : " + str(len(dev_test)) + " - " + str(len(dev_test_text)))

    x_train = feat.train_feature(x for x in dev_train_text)
    x_test = feat.test_feature(x for x in dev_test_text)

    y_train = array(list(labels.index(x) for x in dev_train))
    # Train classifier
    lr = SGDClassifier(loss='log',penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    #feat.show_top10(lr, labels)

    # Dev Test
    predictions = lr.predict(x_test)

    right = 0
    total = len(dev_test)
    res = defaultdict(int)
    for ii in range(0,len(dev_test)):
        if labels[predictions[ii]] == dev_test[ii]:
            right += 1
            res[dev_test[ii]]+=1
        else:
            None

    print("Accuracy on dev: %f" % (float(right) / float(total)))

    full_test = []
    for ii in test:
        text = ii['text']
        full_test.append(text)
    
    x_train = feat.train_feature(x for x in full_train_text)
    x_test = feat.test_feature(x for x in full_test)
    
    y_train = array(list(labels.index(x) for x in full_train))
    
    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)
    
    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)

