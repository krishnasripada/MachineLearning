from csv import DictReader, DictWriter
from collections import defaultdict
from collections import Counter

import numpy as np
import argparse
import nltk, string, itertools
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.stem import PorterStemmer
from nltk import pos_tag, word_tokenize
from itertools import tee, islice
from pattern.en import tag
from nltk.util import ngrams

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    d = defaultdict(int)
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stemmer = PorterStemmer()
    stems = stem_tokens(tokens, stemmer)
    return stems

"""
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
"""

class Analyzer:
    def __init__(self, word, prev, next):
        self.word = word
        self.prev = prev
        self.next = next

    def __call__(self, feature_string):
        feats = feature_string.split()
        if self.word:
            yield feats[0]

        if self.prev:
            for ii in [x for x in feats if x.startswith("P:")]:
                yield ii

        if self.next:
            for ii in [x for x in feats if x.startswith("N:")]:
                yield ii

def normalize_tags(tag):
    if not tag or not tag[0] in string.uppercase:
        return "PUNC"
    else:
        return tag[:2]


kTAGSET = ["", "JJ", "NN", "NNS", "PP", "RB", "VB"]

def example(sentence, position):
    word = sentence[position][0]
    ex = word
    tag = normalize_tags(sentence[position][1])
    if tag in kTAGSET:
        target = kTAGSET.index(tag)
    else:
        target = None

    if position > 0:
        prev = " P:%s" % sentence[position - 1][0]
    else:
        prev = ""
        
    if position < len(sentence) - 1:
        next = " N:%s" % sentence[position + 1][0]
    else:
        next = ''

    char = ' '
    padded_word = "~%s^" % sentence[position][0]
    for ngram_length in xrange(2, 5):
        char += ' ' + " ".join("C:%s" % "".join(cc for cc in x) for x in ngrams(padded_word, ngram_length))

    ex += char
    ex += prev
    ex += next
    return ex

def all_examples(text, train=True):
    sent_num=0
    for ii in [tag(text)]:
        sent_num+=1
        for jj in xrange(len(ii)):
            ex = example(ii, jj)
            if tgt:
                if train and sent_num%5!=0:
                    yield ex
                if not train and sent_num%5==0:
                    yield ex

class Featurizer:
    def __init__(self, analyze):
        self.vectorizer = CountVectorizer(tokenizer = tokenize, analyzer = analyze, stop_words="english")

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(all_examples(''.join(examples)))

    def test_feature(self, examples):
        return self.vectorizer.transform(all_examples(''.join(examples), train=False))

    def show_top10(self, classifier, categories):
        feature_names = self.vectorizer.get_feature_names()
        for i, category in enumerate(categories):
            top10 = np.argsort(classifier.coef_[i])[-10:]
            print("%s: %s" % (category, " ".join(feature_names[top10])))


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--subsample', type=float, default=1.0, help='subsample this amount')
    parser.add_argument('--word', default=False, action='store_true', help="Use word features")
    parser.add_argument('--one_before', default=False, action='store_true', help="Use one word before context as feature")
    parser.add_argument('--one_after', default=False, action='store_true', help="Use one word after context as feature")
    #parser.add_argument('--all_before', default=False, action='store_true', help="Use all words before context as features")
    #parser.add_argument('--all_after', default=False, action='store_true', help="Use all words after context as features")
    
    args = parser.parse_args()
    
    analyzer = Analyzer(args.word, args.one_before, args.one_after)
    
    feat = Featurizer(analyzer)
    
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
    
    errorAna = defaultdict(int)
    for ii in train:
        if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
            continue

        text = ii['text']

        full_train_text.append(text)
        full_train.append(ii['cat'])
        
        if int(ii['id']) % 5 == 0:
            errorAna[ii['cat']]+=1
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
    """
    print "Error Analysis"
    for each in res:
        print each + " - " + str( (float(res[each])/float(errorAna[each]))*100 )
    """
    print("Accuracy on dev: %f" % (float(right) / float(total)))
    
    """

    x_train = feat.train_feature(x['text'] for x in train)
    x_test = feat.test_feature(x['text'] for x in test)
    
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
    """


