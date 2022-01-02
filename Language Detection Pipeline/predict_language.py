from itertools import Predicate
import pandas as pd
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
from language_detection_helper import LanguageDetectionHelper

obj = LanguageDetectionHelper()
obj.display()

class LanguageDetectionResult:
    def __init__(self):
        self.x = 0

    def data_load(self, filename):
        df = obj.read_file(filename)
        return df

    def preprocessing(self, df):
        preprocess = obj.preprocess_x(df)
        return preprocess

    def encoding(self, df, y):
        vector =obj.labelEncoding_x_and_y(df, y)
        return vector

    def loadmodel(self, vector):
        return obj.load_model(vector)

    def predict(self, filename):
        df=self.data_load(filename)
        prep_data = self.preprocessing(df["Text"])
        vector=self.encoding(prep_data,df["Language"])
        predicted_languages=self.loadmodel(vector)
        return predicted_languages

    # def predict_text(self, string):
    #     prep_data = self.preprocessing(str)
    #     label,vector=self.encoding(prep_data,df["Language"])
    #     predicted_languages=self.loadmodel(vector,label)
    #     return predicted_languages
# class LanguageDetectionResult:
#     def __init__(self):
#         self.x = 0

#         LanguageDetectionHelper.
        
        
#     def preprocess(self,X):
#         #print("Started preprocess_x")
#         df_list = []
#         # iterating through all the text
#         for text in X:
#                # removing the symbols and numbers
#                 #print(text)
#                 text = re.sub(r'\[\][!@#$(),n"%^*?:;~`0-9]', '', text)
#                 #print(text)
#                 text = re.sub(r'[[]]', ' ', text)
#                 # converting the text to lower case
#                 text = text.lower()
#                 #print(text)
#                 #print("\n")
#                 # appending to data_list
#                 df_list.append(text)
#         #print("Ended preprocess_x")
#         return df_list   
    
#     def labelEncoding(self,df_list,y):
#         #print("Started labelEncoding_x_and_y")
#         self.cv = CountVectorizer()
#         X = self.cv.fit_transform(df_list).toarray()
#         #X.shape # (10337, 39419)
#         self.le = LabelEncoder()
#         self.y = self.le.fit_transform(y)
        
    def test_model(self,x_test):
        self.df = pd.read_csv("Language Detection.csv")
        X = self.df["Text"] #treated as input
        y = self.df["Language"] #treated as output
        df_list = self.preprocess(X)
        y_pred = self.labelEncoding(df_list,y)
        
   
        self.le.fit(self.df['Language'])
        self.le_name_mapping = dict(zip(self.le.classes_, self.le.transform(self.le.classes_)))
        #print(le_name_mapping)

        dict1 = {value:key for key, value in self.le_name_mapping.items()}
        #print(dict1)
        
        x = self.cv.transform(x_test)
        x.toarray()

        #print(x)
       
      
        model_path = "Pickle_MultinomailNB.pkl"
        # Load the Model back from file
        with open(model_path, 'rb') as file:  
            Pickle_MultinomailNB = pickle.load(file)
        #print(Pickle_MultinomailNB)
        y_pred = Pickle_MultinomailNB.predict(x)
        #print(y_pred)
        predicted_languages = []
        for i in y_pred:
                predicted_languages.append(dict1[i])
        #print(predicted_languages)
        return predicted_languages
    
    def predict_language(self,string):
        self.df = pd.read_csv("Language Detection.csv")
        X = self.df["Text"] #treated as input
        y = self.df["Language"] #treated as output
        df_list = self.preprocess(X)
        y_pred = self.labelEncoding(df_list,y)
        
        
        self.le.fit(self.df['Language'])
        self.le_name_mapping = dict(zip(self.le.classes_, self.le.transform(self.le.classes_)))
        #print(le_name_mapping)

        dict1 = {value:key for key, value in self.le_name_mapping.items()}
        #print(dict1)
        
        x = self.cv.transform([string])
        x.toarray()

        #print(x)
       
      
        model_path = "Pickle_MultinomailNB.pkl"
        # Load the Model back from file
        with open(model_path, 'rb') as file:  
            Pickle_MultinomailNB = pickle.load(file)
        #print(Pickle_MultinomailNB)
        y_pred = Pickle_MultinomailNB.predict(x)[0]
        #print(y_pred)
           
        return dict1[y_pred] 

        
