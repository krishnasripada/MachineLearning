import cPickle as pickle
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

filename="feature.txt"
d = defaultdict(int)
d_new = defaultdict(int)

with open(filename,'rb') as fp:
    d = pickle.load(fp)
fp.close()

for i,j in d.items():
    if isinstance(i, str):
        d_new.update({i:j})
    else:
        d_new.update({''.join(i):j})

CountVectorizer().fit_transform(d_new)

