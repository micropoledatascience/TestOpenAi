# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:01:42 2023

@author: DSONNE
"""
#imortation package
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score, classification_report, roc_auc_score

# Package nécessaire pour open AI
os.chdir(r'C:\Users\SAMRI\Desktop\deep\OpenAI\data')
data = pd.read_csv('Tweets.csv', sep=";")

data.columns = ['completion', 'prompt']

# on enleve les colonnes susceptive de  policy managment(détéectés à partir des 
# des messages d'erruers de la console)
#df_validation = df_validation.drop(labels=[344], axis=0)

# check missing values

data.drop_duplicates(inplace=True)
data.drop([6739, 4112, 6189, 13479, 4672, 10120, 7504, 7421, 13351], inplace = True)
data.dropna(inplace  = True)
data.isna().sum()

data['completion'].value_counts(normalize = True).plot.bar(title="Spam Frequencies")

manual_stop_word = ["virginamerica", "united", "southwestair", "jetblue", "usairways", "americanair"]

def text_preprocessing(df, text_columns):
    # Define the regular expression to remove unwanted characters and patterns
    clean_regex = r"[^a-zA-Z\s]+"
    
    # Preprocess the text data
    for column in text_columns:
        df[column] = df[column].str.lower().replace(clean_regex, '', regex=True)
        
        df[column] = df[column].apply(nltk.word_tokenize)
        stop_words = set(stopwords.words('english'))
        stop_words.update(manual_stop_word)
        df[column] = df[column].apply(lambda x: [word for word in x if word not in stop_words])
        
def traitement(data):
    data['completion'].value_counts().plot.bar(title="Sentiment Frequencies")
    
    # check missing values
    data.drop_duplicates(inplace=True)
    data.dropna(inplace  = True)
    data.isna().sum()    
    text_preprocessing(data, ["prompt"])
    
traitement(data)

X = data['prompt'].apply(lambda x: ' '.join(x))

y = data['completion']

# train test split (70% train - 30% test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)

print('Training Data :', X_train.shape)


print('Testing Data : ', X_test.shape)

cv = CountVectorizer()

X_train_cv = cv.fit_transform(X_train)

X_train_cv.shape
X_train_cv.shape

# Training Logistic Regression model

lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr.fit(X_train_cv, y_train)


X_test_cv = cv.transform(X_test)

# generate predictions

prediction = lr.predict(X_test_cv)

def metrics_nlp(y_test, pred):
    
    # confusion matrix
    matrix = confusion_matrix(y_test, pred)
    print("Matrice de confusion\n", matrix)
    
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test_series, pred)* 100
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)

    class_repor = classification_report(y_test, pred)
    print('Rapport de classification: ', class_repor)

pred = pd.Series(prediction).map({"positive": 1, "negative":2, "neutral": 3})
pred.value_counts(dropna=False)

y_test_series= pd.Series(y_test).map({"positive": 1, "negative":2, "neutral": 3})
metrics_nlp(y_test_series, pred)
