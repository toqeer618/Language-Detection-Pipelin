#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

class LanguageDetectionHelper:
    
    def __init__(self):
        self.model = MultinomialNB()

    def read_file(self, filename):
        df=pd.read_csv(filename)
        return df 
    
    def read_dataframe(self,df):
        self.df = df
        #Separating Independent and Dependent features
        X = df["Text"] #treated as input
        y = df["Language"] #treated as output
        print("In ReadDataframe")
        df_list = self.preprocess_x(X)
        y_pred = self.labelEncoding_x_and_y(df_list,y)
        print(self.x_test)
        return self.y_pred,self.y_test,self.x_test
        
    def preprocess_x(self,X):
        print("Started preprocess_x")
        df_list = []
        # iterating through all the text
        for text in X:
               # removing the symbols and numbers
                #print(text)
                text = re.sub(r'\[\][!@#$(),n"%^*?:;~`0-9]', '', text)
                #print(text)
                text = re.sub(r'[[]]', ' ', text)
                # converting the text to lower case
                text = text.lower()
                #print(text)
                #print("\n")
                # appending to data_list
                df_list.append(text)
        print("Ended preprocess_x")
        return df_list
         
        
    def labelEncoding_x_and_y(self,df_list):
        print("Started labelEncoding_x_and_y")
        self.cv = CountVectorizer()
        X = self.cv.fit_transform(df_list).toarray()
        return X
    def train_test_split(self,X,y):
        x_train, self.x_test, y_train, self.y_test = train_test_split(X, y, test_size = 0.20)
        #print(x_train, x_test, y_train, self.y_test)
        print(self.x_test)
        y_pred = self.train_model(x_train, self.x_test, y_train, self.y_test)
        #return y_pred

    def train_model(self,x_train, x_test, y_train, y_test):
        #model = MultinomialNB()
        self.model.fit(x_train, y_train)
        self.save_model(self.model)
        self.y_pred = self.test_model(self.x_test,y_test)
        #return y_pred
    
    def save_model(self,model):
        model_path = "Pickle_MultinomailNB.pkl"
        with open(model_path, 'wb') as file:  
            pickle.dump(model, file)
        
    def load_model(self,x):
        model_path = "Pickle_MultinomailNB.pkl"
        # Load the Model back from file
        with open(model_path, 'rb') as file:  
            Pickle_MultinomailNB = pickle.load(file)
        #print(Pickle_MultinomailNB)
        y_pred = Pickle_MultinomailNB.predict(x)
        # print(y_pred)
        return y_pred
    
            
                 
    def test_model(self,x_test,y_test):
        y_pred = self.model.predict(x_test)
        print(y_pred)
        return y_pred 
    
    def predict_language(self,string):
        self.le.fit(self.df['Language'])
        self.le_name_mapping = dict(zip(self.le.classes_, self.le.transform(self.le.classes_)))
        #print(le_name_mapping)

        dict1 = {value:key for key, value in self.le_name_mapping.items()}
        #print(dict1)
        
        x = self.cv.transform([string])
        x.toarray()
       
      
        model_path = "Pickle_MultinomailNB.pkl"
        # Load the Model back from file
        with open(model_path, 'rb') as file:  
            Pickle_MultinomailNB = pickle.load(file)
           
        return dict[Pickle_MultinomailNB.predict(x)[0]]    


        
    


   