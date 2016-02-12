#Name: Krishna Chaitanya Sripada
from collections import defaultdict
from csv import DictReader, DictWriter

import nltk
import itertools
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.corpus import treebank
from nltk.stem.lancaster import *
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

kTOKENIZER = TreebankWordTokenizer()

def morphy_stem(word):
    """
    Lancester stemmer
    """
    stemmer = LancasterStemmer()
    stem_word = stemmer.stem(word)
    if stem_word:
        return stem_word.lower()
    else:
        return word.lower()

class FeatureExtractor:
    def __init__(self):
        self.stopset = set(stopwords.words('english'))

    """
       The feature which was already provided but with a better stemming logic
    """
    def features(self, text):
        d = defaultdict(int)
        for ii in kTOKENIZER.tokenize(text):
            d[morphy_stem(ii)] += 1
        d.update(self.stopword_feature(text))
        d.update(self.bigram_feature(text))
        return d

    """
       The stopword feature to filter unnecessary words from the vocabulary
    """

    def stopword_feature(self, text):
        return dict([(word, True) for word in text if word not in self.stopset])

    """
         Bigram feature which makes use of a scoring function to get better bigrams frequency
    """

    def bigram_feature(self, text):
        bigram_colloc_finder = BigramCollocationFinder.from_words(text)
        bigrams = bigram_colloc_finder.nbest(BigramAssocMeasures.chi_sq, 200)
        return dict([(bigram, True) for bigram in itertools.chain(text, bigrams)])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample this amount')
    args = parser.parse_args()
    
    # Create feature extractor (you may want to modify this)
    fe = FeatureExtractor()
    
    # Read in training data
    train = DictReader(open("train.csv", 'r'))
    
    # Split off dev section
    dev_train = []
    dev_test = []
    full_train = []

    for ii in train:
        if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
            continue
        feat = fe.features(ii['text'])
        if int(ii['id']) % 5 == 0:
            dev_test.append((feat, ii['cat']))
        else:
            dev_train.append((feat, ii['cat']))
        full_train.append((feat, ii['cat']))

    # Train a classifier
    print("Training classifier ...")
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)
    # classifier = nltk.classify.MaxentClassifier.train(dev_train, 'IIS', trace=3, max_iter=1000)

    right = 0
    total = len(dev_test)
    for ii in dev_test:
        prediction = classifier.classify(ii[0])
        if prediction == ii[1]:
            right += 1
    print("Accuracy on dev: %f" % (float(right) / float(total)))

    # Retrain on all data
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test)

    # Read in test section
    test = {}
    for ii in DictReader(open("test.csv")):
        test[ii['id']] = classifier.classify(fe.features(ii['text']))

    # Write predictions
    o = DictWriter(open('pred.csv', 'w'), ['id', 'pred'])
    o.writeheader()
    for ii in sorted(test):
        o.writerow({'id': ii, 'pred': test[ii]})
