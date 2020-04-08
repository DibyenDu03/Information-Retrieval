# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 14:05:26 2020

@author: Dibyendu
"""

import os
import codecs
import string
import os
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import pickle
import math
import numpy as np
from nltk.tree import *
from nltk.stem import WordNetLemmatizer 
import random
import numpy as num
from math import exp
from nltk.tokenize import RegexpTokenizer 
from nltk.stem import WordNetLemmatizer 
  


def read(file):
	fp = codecs.open(file,"r",encoding='utf-8', errors='ignore')
	text = fp.read()
	return text

files = []
token = RegexpTokenizer('\s+|\-+|\.+|\@+|\t+|\n+|[0-9]+|\"+|\>+|\,+|\?+|\:+|\{+|\(+|\[+|\)+|\}+|\]+|\<+|\_+|\!+|\/+|\|+|\\+|\*+|\=+|\^+', gaps = True)
lem = WordNetLemmatizer() 
path='20_newsgroups/'
for r, d, f in os.walk(path):
	for file in f:
		files.append(os.path.join(r, file))

files.sort()
doc=[]
count=0
for i in files:
    text=read(i)
    m=token.tokenize(text)
    doc.append(m)
    count+=1
print("#Docs are ",count)
word_list={}
count=-1
for i in doc:
    count+=1
    index=0
    for tk in i:
        k=lem.lemmatize(tk)
        #k=tk
        if not k.lower()  in word_list.keys():
            
            word_list[k.lower()]=[]
            word_list[k.lower()].append(1)
            word_list[k.lower()].append([])
            word_list[k.lower()][1].append(count)
            
        else:
            if count != word_list[k.lower()][1][len(word_list[k.lower()][1])-1]:
                
                word_list[k.lower()][0]+=1
                word_list[k.lower()][1].append(count)
                
print('#Words are ',len(word_list))        
word_list['File_info']=[]
word_list['File_info'].append(len(files))
word_list['File_info'].append(files)
index = open('InvertedIndex_Q1', 'ab')
pickle.dump(word_list, index)                
index.close()
print('Index have been successfully stored')

'''	
# Create a reference variable for Class RegexpTokenizer 

	
# Create a string input 
gfg = "I@love1234Python\nhello"
	
# Use tokenize method 
geek = token.tokenize(gfg) 
	
print(geek)
    
'''