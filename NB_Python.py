
# coding: utf-8

# In[1]:


import json as j
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2


# In[2]:


json_data = None
with open('workfile.json', 'r') as f:
     json_data = j.load(f)

data = pd.DataFrame(json_data)

#------> Test Data <-----------#
test_data = None

with open('testfile.json', 'r') as f:
     test_data = j.load(f)

test_data = pd.DataFrame(test_data)


# In[3]:


stemmer = SnowballStemmer('english')
words = stopwords.words("english")

data['cleaned'] = data['Question'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
test_data['cleaned'] = test_data['Question'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

X_train, y_train = data['cleaned'], data.Context
X_test, y_test = test_data['cleaned'], test_data.Context



# In[ ]:


pipeline = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

model = pipeline.fit(X_train, y_train)


# In[15]:


print("accuracy score: " + str(model.score(X_test, y_test)))

print(model.predict(['What is a group of turkeys called ?']))


# In[6]:


# BernoulliNB
pipeline = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', BernoulliNB())])

model = pipeline.fit(X_train, y_train)


# In[7]:


print("accuracy score: " + str(model.score(X_test, y_test)))

print(model.predict(['What is a group of turkeys called ?']))


# In[15]:


#SGDClassifier
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=10000)),
                     ('clf', SGDClassifier( penalty='l1',max_iter=3000))])

model = pipeline.fit(X_train, y_train)


# In[16]:


print("accuracy score: " + str(model.score(X_test, y_test)))

print(model.predict(['What is a group of turkeys called ?']))


# In[20]:


#NearestCentroid
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=10000)),
                     ('clf', NearestCentroid())])

model = pipeline.fit(X_train, y_train)


# In[21]:


print("accuracy score: " + str(model.score(X_test, y_test)))

print(model.predict(['What is a group of turkeys called ?']))


# In[8]:


#Perceptron
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=10000)),
                     ('clf', Perceptron(max_iter=3000))])
model = pipeline.fit(X_train, y_train)


# In[10]:


print("accuracy score: " + str(model.score(X_test, y_test)))

print(model.predict(['What is a group of turkeys called ?']))


# In[11]:


#DecisionTree
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=10000)),
                     ('clf', tree.DecisionTreeClassifier())])

model = pipeline.fit(X_train, y_train)


# In[12]:


print("accuracy score: " + str(model.score(X_test, y_test)))

print(model.predict(['What is a group of turkeys called ?']))


# In[4]:


# BernoulliNB
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=10000)),
                     ('clf', BernoulliNB())])

model = pipeline.fit(X_train, y_train)


# In[5]:


print("accuracy score: " + str(model.score(X_test, y_test)))

print(model.predict(['What is a group of turkeys called ?']))

