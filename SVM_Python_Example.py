
# coding: utf-8

# In[4]:


import json as j
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
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

#------> Training Data <-----------#
json_data = None
# with open('data/yelp_academic_dataset_review.json') as data_file:
#     lines = data_file.readlines()
#     joined_lines = "[" + ",".join(lines) + "]"

#     json_data = j.loads(joined_lines)
with open('workfile.json', 'r') as f:
     json_data = j.load(f)

data = pd.DataFrame(json_data)


# In[2]:


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

pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=10000)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])


model = pipeline.fit(X_train, y_train)


# In[40]:


vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']

feature_names = vectorizer.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)

target_names = ['manner', 'cremat', 'animal', 'exp', 'ind']
print("top 10 keywords per class:")
for i, label in enumerate(target_names):
    top10 = np.argsort(clf.coef_[i])[-10:]
    print("%s: %s" % (label, " ".join(feature_names[top10])))

print("accuracy score: " + str(model.score(X_test, y_test)))

print(model.predict(['What is a group of turkeys called ?']))


# In[41]:


print(model.predict(['How far is Pluto from the sun ?']))


# In[7]:


#RandomForestClassifier
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=10000)),
                     ('clf', RandomForestClassifier())])


model = pipeline.fit(X_train, y_train)


# In[8]:


print("accuracy score: " + str(model.score(X_test, y_test)))

print(model.predict(['What is a group of turkeys called ?']))


# In[12]:


#PassiveAggressiveClassifier
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=10000)),
                     ('clf', PassiveAggressiveClassifier(C=1.0, max_iter=3000))])


model = pipeline.fit(X_train, y_train)


# In[13]:


print("accuracy score: " + str(model.score(X_test, y_test)))

print(model.predict(['What is a group of turkeys called ?']))

