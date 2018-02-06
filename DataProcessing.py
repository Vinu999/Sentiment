
# coding: utf-8

# In[75]:


import nltk
import re
import json
from nltk.corpus import qc, state_union
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer

training_data = qc.raw('train.txt')
tok = sent_tokenize(training_data)

with open("train.txt") as f:
    text = f.read()

toc = text.split("\n")
with open('workfile.json', 'a') as outfile:
    print("[",file=outfile)

for x in range(5452):
    word_list = toc[x].split()
    n = len(word_list)
    lists = word_list[1:n]
    start = word_list[0].split(':')
    sentence = ""
    for word in lists:
        sentence += word + " "
    with open('workfile.json', 'a') as outfile:  
        print(json.dumps({'Question': sentence, 'Class': start[0], 'Context': start[1]}, sort_keys=True, indent=4),file=outfile)
        if x != 4:
            print(",",file=outfile)
            
with open('workfile.json', 'a') as outfile:
    print("]",file=outfile)


# In[77]:


with open("test.txt") as f:
    text = f.read()

toc = text.split("\n")

with open('testfile.json', 'a') as outfile:
    print("[",file=outfile)
    
for x in range(500):
    word_list = toc[x].split()
    n = len(word_list)
    lists = word_list[1:n]
    start = word_list[0].split(':')
    sentence = ""
    for word in lists:
        sentence += word + " "
    with open('testfile.json', 'a') as outfile:  
        print(json.dumps({'Question': sentence, 'Class': start[0], 'Context': start[1]}, sort_keys=True, indent=4),file=outfile)
        if x != 4:
            print(",",file=outfile)
            
with open('testfile.json', 'a') as outfile:
    print("]",file=outfile)


# In[68]:




