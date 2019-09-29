import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

class Models(object):
    def __init__(self,X,y,kfolds):
        self.X = X
        self.y = y
        self.kfolds = kfolds
        
    def nb(self):
        nb = MultinomialNB()
        nb.fit(self.X,self.y)
        return nb
        
        
    def logistic(self):
        lr = make_pipeline(LogisticRegressionCV(cv=self.kfolds))
        lr.fit(self.X,self.y)

        return lr
    
    def main(self):
        print('START Fit')

        print(datetime.now(), 'NB')
        nb = self.nb()
        print('Training done !!')

        print(datetime.now(), 'logistics')
        logistic = self.logistic()
        print('Training done !!')
        
        return logistic,nb
        
    def save_model(self, model, path):
        with open(path, 'wb') as clf:
            pickle.dump(model, clf) 
              
              