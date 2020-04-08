# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:38:14 2020

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
import re
from A2_handle_numerical import Word2Number

def read(file):
	fp = codecs.open(file,"r",encoding='utf-8', errors='ignore')
	text = fp.read()
	return text.strip()

def Tf_cal(freq,length,ind,highest):
    
    res=0                   ######################################### 1) 
    if freq>0:
        res=1
        
    res=freq               ######################################### 2) 
    
    res=freq/length        ######################################### 3) 
    
    k=0.5
    #res=k+(1-k)*freq/highest[ind]               ######################################### 4) 
    
    #res=math.log10(1+freq)      ###############################################   5)
    
    return res 



def Idf_cal(freq,Doc):
    
    res=1+math.log10(Doc/(freq))    ######################################### 1) 
    
    res=math.log10(Doc/(freq))      ######################################### 2) 
    
    res=1+math.log10(Doc/(freq+1))  ######################################### 3) 
    
    return res

files = []
token = RegexpTokenizer('\s+|\-+|\.+|\@+|\t+|\n+|\"+|\>+|\,+|\?+|\:+|\{+|\(+|\[+|\)+|\}+|\]+|\<+|\_+|\!+|\/+|\|+|\\+|\*+|\=+|\^+', gaps = True)
lem = WordNetLemmatizer() 
path='Dataset/'
path_index='Index/'
indexpath=[]
title=[]
for r, d, f in os.walk(path_index):
    for file in f:
        indexpath.append(os.path.join(r, file))
indexpath.sort()
for i in indexpath:
    text=read(i)
    title+=re.findall('<BR><TD> (.*)\n', text)
    
for r, d, f in os.walk(path):
	for file in f:
		files.append(os.path.join(r, file))
files.sort()
doc=[]
count=0
for i in files:
    text=read(i)
    m=token.tokenize(text)
    
    store=Word2Number(m)        ################################ Changes Happen     ##############
    
    #store=m
    doc.append(store)
    
    count+=1
print("#Docs are ",count)

word_list={}
cosine_index=[]
count=-1
for i in doc:
    count+=1
    index=0
    cosine_index.append({})
    cosine_index[count]['Unique']=0.0
    cosine_index[count]['Total']=0.0
    for tk in i:
        k=tk.strip().lower()
        k=lem.lemmatize(tk)
        if not k.lower() in cosine_index[count].keys():
            cosine_index[count][k.lower()]=1.0
            cosine_index[count]['Unique']+=1.0
            cosine_index[count]['Total']+=1.0
        else:
            cosine_index[count][k.lower()]+=1.0
            cosine_index[count]['Total']+=1.0
        if not k.lower()  in word_list.keys():
            word_list[k.lower()]=[]
            word_list[k.lower()].append(1)
            word_list[k.lower()].append([])
            word_list[k.lower()][1].append(count)
            
        else:
            if count != word_list[k.lower()][1][len(word_list[k.lower()][1])-1]:
                
                word_list[k.lower()][0]+=1
                word_list[k.lower()][1].append(count)

for i in range(0,len(title)):
    tk=title[i].strip().lower()
    m=token.tokenize(tk)
    
    k=Word2Number(m)        ################################ Changes Happen     ##############
    
    #k=m
    s={}
    for j in k:
        tk=lem.lemmatize(j)
        tk=tk.lower()
        if tk not in s.keys():
            s[tk]=1.0
        else:
            s[tk]+=1.0
    title[i]=s

Number_doc=count+1

for i in range(0,Number_doc):
    for tk in title[i].keys():
        k=tk.lower()
        k=lem.lemmatize(k)
        if k.lower() in cosine_index[i].keys():
            cosine_index[i][k.lower()]+=1.0
            cosine_index[i]['Total']+=1.0
            word_list[k.lower()][0]+=1.0
        else:
            cosine_index[i][k.lower()]=1.0
            cosine_index[i]['Unique']+=1.0
            cosine_index[i]['Total']+=1.0
            word_list[k.lower()]=[]
            word_list[k.lower()].append(1.0)
            word_list[k.lower()].append([])
            word_list[k.lower()][1].append(i)
    
word_list['File_info']=[]
word_list['File_info'].append(Number_doc)
word_list['File_info'].append(files)
word_list['File_info'].append(title)          
print('#Words are ',len(word_list))
index = open('InvertedIndex_Q1', 'ab')
pickle.dump(word_list, index)                
index.close()
index = open('Cosine_index_Q1', 'ab')
pickle.dump(cosine_index, index)                
index.close()
print('Tf-Idf Matrix have been successfully stored')

