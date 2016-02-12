from csv import DictReader, DictWriter
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
import yaml
from time import gmtime, strftime
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
import ast
from collections import defaultdict
from sklearn import svm
from pattern.en import tag
from nltk.tag.stanford import NERTagger
import cPickle as pickle
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import string, re

st = NERTagger('stanford-ner-2015-04-20/classifiers/english.conll.4class.distsim.crf.ser.gz', 'stanford-ner-2015-04-20/stanford-ner.jar')
kTOKENIZER = WordPunctTokenizer()
stop = stopwords.words('english')

def tokenize(text):
    tokens = kTOKENIZER.tokenize(text)
    filtered_words = [w for w in tokens if not w in stop]###Removing Stop Words
    return " ".join(filtered_words)

class classify:
    
    def __init__(self):
        self.x_train = []
        self.x_test = []
        self.y = []

        self.test = []
        self.train = []        

        self.quesFeat = {}        
        self.predictions = []
        
        self.cv_train = []
        self.cv_test = []
        self.cv_y_train = []
        self.cv_y_test = []
        self.countries = defaultdict()
        self.cities = defaultdict()
        self.totalCountries = 0
        self.personCount = 0
        self.counter = 0
        with open("countries.txt", "r") as fp:
            for word in fp:
                word = word.lower()
                word = word.replace("\n","")
                spl = word.split(" ");
                if(len(spl)>2):
                    word = " ".join(spl[:2])
                self.countries[word]=1
        fp.close()

    def questionFeatures(self):
        questions = list(DictReader(open("questions.csv", 'rU')))
        self.train = list(DictReader(open("train.csv", 'rU')))
        total = len(questions)
        positions = {}
        for each in self.train:
            positions.update({each['question']: float(each['position'])})
        
        for ques in questions:
            unigrams = defaultdict()
            unigrams['_length_'] = len(ques['questionText'])
            unigrams['_cat_'] = ques['cat']
            unigrams['_answer_'] = ques['answer']
            """
            for pos, word in ast.literal_eval(ques['unigrams']).items():
                try:
                    word = str(word)
                    if(self.countries[word]):
                        if(word in unigrams.keys()):
                            unigrams[word]+=1
                        else:
                            unigrams[word]=1
                        self.totalCountries+=1
                except:
                    None
                try:
                    if isinstance(int(str(word)), int):
                        if len(str(word)) <= 3:
                            if "_number_" in unigrams.keys():
                                unigrams["_number_"]+=1
                            else:
                                unigrams["_number_"]= 1
                        else:
                            num = int(str(word))
                            if(num<1980):
                                if "_number_" in unigrams.keys():
                                    unigrams["_yearLt1900_"]+=1
                                else:
                                    unigrams["_yearLt1900_"]= 1
                            else:
                                if "_number_" in unigrams.keys():
                                    unigrams["_yearGt1900_"]+=1
                                else:
                                    unigrams["_yearGt1900_"]= 1
                        
                            if "_year_" in unigrams.keys():
                                unigrams["_year_"]+=1
                            else:
                                unigrams["_year_"]=1
                            
                except:
                    None
        
        
            ###########Capital Letter words##############################
            finalWord = ""
            for word in ques['questionText'].split()[1:]:
                if not word.islower():
                    finalWord=finalWord +" "+word
                else:
                    for w, tag in tag(finalWord):
                        if str(tag) in ['NNS','NN','NNP']:
                            if finalWord in unigrams.keys():
                                unigrams[finalWord.lstrip()]+=1
                            else:
                                unigrams[finalWord.lstrip()]=1
                
                    finalWord = ""

            ##########Adding the last one if any#################
            if len(finalWord)>0:
                for w, tag in tag(finalWord):
                    if str(tag) in ['NNS','NN','NNP']:
                        if finalWord in unigrams.keys():
                            unigrams[finalWord.lstrip()]+=1
                        else:
                            unigrams[finalWord.lstrip()]=1
        
            """
            finalWord_Org = ""
            finalWord_Loc = ""
            finalWord_Per = ""
            index1 = ques['questionText'].rfind("FTP")
            index2 = ques['questionText'].rfind("For ten points")
            index3 = ques['questionText'].rfind("For 10 points")
            index4 = ques['questionText'].rfind("for ten points")
            index5 = ques['questionText'].rfind("for 10 points")
            if index1!=-1:
                main_i = index1 + len("FTP")
                main_q = ques['questionText'][main_i+1:]
            elif index2!=-1:
                main_i = index2 + len("For ten points")
                main_q = ques['questionText'][main_i+1:]
            elif index3!=-1:
                main_i = index3 + len("For 10 points")
                main_q = ques['questionText'][main_i+1:]
            elif index4!=-1:
                main_i = index4 + len("for ten points")
                main_q = ques['questionText'][main_i+1:]
            elif index5!=-1:
                main_i = index5 + len("for 10 points")
                main_q = ques['questionText'][main_i+1:]
            
            
            main_q = main_q.strip()
            
            for w, ner in st.tag(main_q.split()):
                if str(ner)=="ORGANIZATION":
                    finalWord_Org = finalWord_Org+" "+str(w)
                elif str(ner)=="LOCATION":
                    finalWord_Loc = finalWord_Loc+ " "+str(w)
                elif str(ner)=="PERSON":
                    finalWord_Per = finalWord_Per+ " "+str(w)
                
                else:
                    if len(finalWord_Org)>0:
                        print str(finalWord_Org)
                        if finalWord_Org in unigrams.keys():
                            unigrams[str(finalWord_Org).lstrip().lower()]+=1
                        else:
                            unigrams[str(finalWord_Org).lstrip().lower()]=1
                                
                    elif len(finalWord_Loc)>0:
                        print str(finalWord_Loc)
                        if finalWord_Loc in unigrams.keys():
                            unigrams[str(finalWord_Loc).lstrip().lower()]+=1
                        else:
                            unigrams[str(finalWord_Loc).lstrip().lower()]=1
                
                    elif len(finalWord_Per)>0:
                        print str(finalWord_Per)
                        if finalWord_Per in unigrams.keys():
                            unigrams[str(finalWord_Per).lstrip().lower()]+=1
                        else:
                            unigrams[str(finalWord_Per).lstrip().lower()]=1
                    
                    finalWord_Org = ""
                    finalWord_Loc = ""
                    finalWord_Per = ""
                                
            
            ##########Adding the last one if any#################
            if len(finalWord_Org)>0:
                print str(finalWord_Org)
                if finalWord_Org in unigrams.keys():
                    unigrams[str(finalWord_Org).lstrip().lower()]+=1
                else:
                    unigrams[str(finalWord_Org).lstrip().lower()]=1
            
            elif len(finalWord_Loc)>0:
                print str(finalWord_Loc)
                if finalWord_Loc in unigrams.keys():
                    unigrams[str(finalWord_Loc).lstrip().lower()]+=1
                else:
                    unigrams[str(finalWord_Loc).lstrip().lower()]=1
                        
            elif len(finalWord_Per)>0:
                print str(finalWord_Per)
                if finalWord_Per in unigrams.keys():
                    unigrams[str(finalWord_Per).lstrip().lower()]+=1
                else:
                    unigrams[str(finalWord_Per).lstrip().lower()]=1

            self.quesFeat[ques['question']] = unigrams
                    
                
    def readData(self, accuracy):
        print self.totalCountries
        print "In readData"
        self.train = list(DictReader(open("train.csv", 'rU')))
        self.test = list(DictReader(open("test.csv", 'rU')))
        v = DictVectorizer(sparse=False)
               
        ind = 0
        # training set
        for each in self.train:
            features = {}
            features = self.quesFeat[each['question']]
            features['_user_'] = each['user']
            features['_question_'] = each['question']
            #features['_answer_'] = each['answer']
            
            if(accuracy==False):
                self.x_train.append(features)
                self.y.append(float(each['position']))
            else:
                if(ind%5 == 0):
                    self.cv_test.append(features)
                    self.cv_y_test.append(float(each['position']))
                else:
                    self.cv_y_train.append(float(each['position']))
                    self.cv_train.append(features)
                ind+=1
            
        ## test set
        if(accuracy == False):
            for each in self.test:
                features = {}
                features = self.quesFeat[each['question']]
                features['_user_'] = each['user']
                features['_question_'] = each['question']        
                self.x_test.append(features)
            self.x_train = v.fit_transform(self.x_train)
            self.x_test = v.transform(self.x_test)
        else:
            self.cv_train = v.fit_transform(self.cv_train)
            self.cv_test = v.transform(self.cv_test)
        
                
    def predict(self):
        print "In Predict"
        clf = linear_model.Lasso(alpha=0.01)
        clf = clf.fit(self.x_train, self.y)
        self.predictions = clf.predict(self.x_test)

    def writePredictions(self):
        print "In writePredictions"
        o = DictWriter(open("predictions.csv", 'w'), ["id", "position"])
        o.writeheader()
        for ii, pp in zip([x['id'] for x in self.test], self.predictions):
            d = {'id': ii, 'position': pp}
            o.writerow(d)
    
    def getAccuracy(self):
        
        print "In getAccuracy"
        #clf = tree.DecisionTreeClassifier()
        
        clf = linear_model.Lasso(alpha=0.01)
        #clf = DecisionTreeRegressor(max_depth=200)
        #clf = svm.SVR()
        clf = clf.fit(self.cv_train, self.cv_y_train)
        self.predictions = clf.predict(self.cv_test)
        
        acc = 0
        ind = 0
        for pred in self.predictions:
            if(pred == self.cv_y_test[ind]):
                acc+=1
            ind+=1
        print "Accuracy correct/total : "
        print float(acc)/float(ind)
        
        print "Mean Square Error cv set is : "
        print mean_squared_error(self.cv_y_test, self.predictions)
        
        
if __name__ == "__main__":
    print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--accuracy', type=bool, default=False,
                        help='test accuracy')
    args = parser.parse_args()
    
    obj = classify()
    obj.questionFeatures()
    
    obj.readData(args.accuracy)
    
    if(args.accuracy == True):
        obj.getAccuracy()
    else:
        obj.predict()
        obj.writePredictions()
    print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
    
    # play beep after finishing
    #import os
    #os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % ( 1, 150))


