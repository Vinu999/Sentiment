
# coding: utf-8

# In[1]:


import re
import nltk
from textblob.classifiers import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# In[2]:


from textblob.classifiers import DecisionTreeClassifier


# In[3]:


with open("train.txt") as f:
    text = f.read()
toc = text.split("\n")
x = len(toc) - 1
training_data = [[0 for p in range(2)] for y in range(x)]
for x in range(5452):
    toc[x].lower()
    word_list = toc[x].split()
    n = len(word_list)
    lists = word_list[1:n]
    start = word_list[0].split(':')
    sentence = ""
    for word in lists:
        sentence += word + " "
    re.sub("[^a-zA-Z]", " ", sentence)
    training_data[x][0] = sentence
    training_data[x][1] = start[1]

print(training_data[1:4])


# In[26]:


cl = NaiveBayesClassifier(training_data)


# In[23]:


cl.classify("How did serfdom develop in and then leave India ?")  # "pos"


# In[30]:


cl.classify("When did Hawaii become a state ?")  # "pos"


# In[4]:


with open("test.txt") as f:
    text = f.read()
tov = text.split("\n")
v = len(tov) - 1
test_data = [[0 for p in range(2)] for y in range(v)]
for x in range(500):
    word_list = tov[x].split()
    n = len(word_list)
    lists = word_list[1:n]
    start = word_list[0].split(':')
    sentence = ""
    for word in lists:
        sentence += word + " "
        re.sub("[^a-zA-Z]", " ", sentence)
    test_data[x][0] = sentence
    test_data[x][1] = start[1]


# In[28]:


cl.accuracy(test_data)


# In[42]:


dt = DecisionTreeClassifier(training_data)


# In[ ]:


dt.classify("When did Hawaii become a state ?")  # "pos"


# In[ ]:


dt.accuracy(test_data)


# In[48]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
f = open('data.txt', 'r');
data = f.read();


# In[51]:


from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)


# In[52]:


twenty_train.target_names #prints all the categories
print("\n".join(twenty_train.data[0].split("\n")[:3])) #prints first line of the first data file


# In[54]:


print(twenty_train)


# In[19]:


re.sub("[^a-zA-Z]", " ", "What films featured the character Popeye Doyle ? ")

