
# coding: utf-8

# In[1]:


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union, qc, wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.util import ngrams


# In[2]:


f = open('data.txt', 'r');
data = f.read();
sentences = sent_tokenize(data)
train_text = state_union.raw("2005-GWBush.txt")


# In[3]:


custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(data)


# In[4]:


for i in tokenized[:5]:
    words = nltk.word_tokenize(i)
    tagged = nltk.pos_tag(words)
    print(tagged)

