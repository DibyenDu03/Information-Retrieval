# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 09:11:25 2020

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


def Sort_Tuple(tuple):   
    tuple.sort(key = lambda x: x[1],reverse=True)  
    return tuple 

dbfile = open('InvertedIndex_Q1', 'rb')      
wordlist = pickle.load(dbfile) 
dbfile.close()
dbfile = open('ChampionList_Q1', 'rb')      
champion = pickle.load(dbfile) 
dbfile.close()

r=15        ######################_Value_of_R_###########################


high={}
low={}
for i in champion.keys():
    high[i]=champion[i][:r]
    low[i]=champion[i][r:]

token = RegexpTokenizer('\s+|\-+|\.+|\@+|\t+|\n+|[0-9]+|\"+|\>+|\,+|\?+|\:+|\{+|\(+|\[+|\)+|\}+|\]+|\<+|\_+|\!+|\/+|\|+|\\+|\*+|\=+|\^+', gaps = True)
lem = WordNetLemmatizer() 
Query=input("Enter a sentence- ")
Query=Query.lower()
Query=token.tokenize(Query)
query=[]
for k in Query:
    query.append(lem.lemmatize(k))

check=[]
similar=[]
for i in range(0,wordlist['File_info'][0]):
    check.append(0)
    similar.append([i,0])
k=int(input("Enter number of retrieve file-  "))
files=wordlist['File_info'][1]

top=0
checklist=[]
for i in query:
    if i in high.keys():
        for j in high[i]:
            similar[j[0]][1]+=files[j[0]][1]*j[1]
            if check[j[0]]==0:
                check[j[0]]=1
                top+=1
                checklist.append(j[0])              
if top<=k:
    for i in query:
        if i in low.keys():
            for j in low[i]:
                similar[j[0]][1]+=files[j[0]][1]*j[1]
                if check[j[0]]==0:
                    check[j[0]]=1
                    top+=1
                    checklist.append(j[0])

similar=Sort_Tuple(similar)
print("\n\nMost similar documents are- \n")
for i in range(k):
    print("\t",i+1,") ",files[similar[i][0]][0])
    
        