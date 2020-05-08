# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:39:25 2020

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
from tqdm import tqdm
import matplotlib.pyplot as plt

def read(file):
	fp = codecs.open(file,"r",encoding='utf-8', errors='ignore')
	text = fp.read()
	return text

files = []
token = RegexpTokenizer('\s+|\-+|\.+|\@+|\t+|\n+|[0-9]+|\"+|\>+|\,+|\?+|\:+|\{+|\(+|\[+|\)+|\}+|\]+|\<+|\_+|\!+|\/+|\|+|\\+|\*+|\=+|\^+|\$+|\&+|\#+|\*+|\++|;+', gaps = True)
lem = WordNetLemmatizer() 
path='C:/Users/Dibyendu/Desktop/Information Retrieval/A4_MT19034/20_newsgroups/'
for r, d, f in os.walk(path):
	for file in f:
		files.append(os.path.join(r, file))
files.sort()
doc=[]
size=len(files)
count=0
for i in tqdm(range(0,size)):
    text=read(files[i])
    m=token.tokenize(text)
    doc.append(m)
    count+=1
print("#Docs are ",count)
      
# Extracting all valid words
word_list={} # Words collection
count=-1

cosine_index=[] # Document index file

for i in tqdm(range(0,size)):
    count+=1
    index=0
    cosine_index.append({})
    cosine_index[count]['Unique']=0.0
    cosine_index[count]['Total']=0.0
    files[i]=[files[i],len(doc[i])]
    for tk in doc[i]:
        k=tk.lower()
        k=lem.lemmatize(k)
        #k=tk
        
        if  k.lower() not in cosine_index[count].keys():
            cosine_index[count][k.lower()]=1.0
            cosine_index[count]['Unique']+=1.0
            cosine_index[count]['Total']+=1.0
        else:
            cosine_index[count][k.lower()]+=1.0
            cosine_index[count]['Total']+=1.0
        
        if k.lower() not in word_list.keys():
            
            word_list[k.lower()]=[]
            word_list[k.lower()].append(1)
            word_list[k.lower()].append([])
            word_list[k.lower()][1].append([count,1])
            
        else:
            if count != word_list[k.lower()][1][len(word_list[k.lower()][1])-1][0]:
                
                word_list[k.lower()][0]+=1
                word_list[k.lower()][1].append([count,1])
            else:
                word_list[k.lower()][1][word_list[k.lower()][0]-1][1]+=1.0
    
print('#Words are ',len(word_list)) 
      
# Storing index table into pickle file

word_list['File_info']=[]
word_list['File_info'].append(len(files))
word_list['File_info'].append(files)
word_list['Cosine_sim']=cosine_index
print("Index table is created")
index = open('TF-Idf_Q1', 'ab')
pickle.dump(word_list, index)                
index.close()
print("Index table is stored")