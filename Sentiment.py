
# coding: utf-8

# In[1]:


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union, qc, wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[5]:


f = open('data.txt', 'r');
data = f.read();
print(data);


# In[6]:


sentences = sent_tokenize(data)
print(sentences)


# In[7]:


words = word_tokenize(data)
print(words)


# In[8]:


stop_words = set(stopwords.words("english"))

filltered_words = []

for w in words:
    if w not in stop_words:
        filltered_words.append(w)
        
print(filltered_words)


# In[9]:


stemmed = []
ps = PorterStemmer()
for w in filltered_words:
    stemmed.append(ps.stem(w))
    
print(stemmed)


# In[10]:


from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


# In[11]:


train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")


# In[12]:


custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(data)


# In[13]:


for i in tokenized[:5]:
    words = nltk.word_tokenize(i)
    tagged = nltk.pos_tag(words)
    print(tagged)
    #chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NN>?}"""
    #chunkParser = nltk.RegexpParser(chunkGram)
    #chunked = chunkParser.parse(tagged)
    #chunked.draw()
    namedEnt = nltk.ne_chunk(tagged, binary=True)
    namedEnt.draw()


# In[14]:


lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))


# In[ ]:


sample = qc.raw("test.txt")
tok = sent_tokenize(sample)

for x in range(5):
    print(tok[x])


# In[ ]:


syns = wordnet.synsets("program")
print(syns[0].name())
print(syns[0].lemmas()[0].name())
print(syns[0].definition())
print(syns[0].examples())


# In[ ]:


synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))


# In[ ]:


w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))


# In[ ]:


w1 = wordnet.synset('cat.n.01')
w2 = wordnet.synset('dog.n.01')
print(w1.wup_similarity(w2))


# In[21]:


import json
with open('workfile.json', 'r') as f:
    data = json.load(f)

for x in data:
    print(x['Context'])
#classifier = nltk.NaiveBayesClassifier.train(training_set)

#print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

#classifier.show_most_informative_features(15)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog.",
        "The dog.",
        "The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())

