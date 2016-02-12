from sklearn import svm
import argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

class Numbers:
    """
        Class to store MNIST data
    """
    
    def __init__(self, location):
        
        import cPickle, gzip
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Support Vector Machine options')
    parser.add_argument('--limit', type=int, default=-1, help="Restrict training to this many examples")
    args = parser.parse_args()
    data = Numbers("../data/mnist.pkl.gz")

    if args.limit >0:
        print "Data limit: %i" % args.limit
        X = data.train_x[:args.limit]
        y = data.train_y[:args.limit]

    else:
        X = data.train_x
        y = data.train_y


    train_X = []
    train_y = []
    test_X = []
    test_y = []
    for i, x in enumerate(y):
        if x==3 or x==8:
            train_y.append(x)
            train_X.append(X[i])

    for i, x in enumerate(data.test_y):
        if x==3 or x==8:
            test_y.append(x)
            test_X.append(data.test_x[i])


    X_train = np.array(train_X)
    y_train = np.array(train_y)

    X_test = np.array(test_X)
    y_test = np.array(test_y)
    

    """
    C = [1,2,5,10,100]

    for c in C:
        clf = svm.SVC(kernel='rbf', C=c)
        clf.fit(X,y)

        for x in clf.support_vectors_:
            plt.imshow(x.reshape((28,28)), cmap = cm.Greys_r)

        plt.savefig("test/result_"+clf.kernel+"_"+str(c)+".png")
        plt.close()

        clf = svm.SVC(kernel='linear', C=c)
        clf.fit(X,y)
    
        for x in clf.support_vectors_:
            plt.imshow(x.reshape((28,28)), cmap = cm.Greys_r)

        plt.savefig("test/result_"+clf.kernel+"_"+str(c)+".png")
        plt.close()
    """
    """
    clf = svm.SVC(kernel='rbf')
    clf.fit(X,y)
    print clf.predict(X)
    print clf.score(X,y)
    """
    #print X_train, y_train, len(X_train)
    #print X_test, y_test, len(X_test)
    """
    C = [0.1, 0.01,0.001, 0.0001, 1, 2, 5, 10, 100]

    for c in C:

        clf = svm.SVC(kernel='linear', C = c)
        clf.fit(X_train,y_train)
        print "Prediction: ", clf.predict(X_test)
        print "Accuracy using Linear Kernel with C= ",c
        print clf.score(X_test, y_test)

        print "###############################"
        clf = svm.SVC(kernel='rbf', C = c)
        clf.fit(X_train,y_train)
        print "Prediction: ", clf.predict(X_test)
        print "Accuracy using RBF Kernel with C= ",c
        print clf.score(X_test, y_test)
        print "###############################"

    """
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train,y_train)
        
    plt.imshow(clf.support_vectors_[3].reshape((28,28)), cmap = cm.Greys_r)
    
    plt.show()
    plt.close()
