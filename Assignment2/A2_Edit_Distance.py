# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 08:29:44 2020

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
from tqdm import tqdm

def read(file):
	fp = codecs.open(file,"r",encoding='utf-8', errors='ignore')
	text = fp.read()
	return text.strip()

def  create_matrix(len2,len1):
    list1=[]
    for i in range(0,len2+1):
        a=[]
        for j in range(0, len1+1):
            a.append(0)
        list1.append(a)
    for i in range(0,len2+1):
        list1[i][0]=i
    for j in range(0,len1+1):
        list1[0][j]=j*2
    list1=np.array(list1)
    return list1

def SortTuple(turple):  
    turple.sort(key = lambda x: x[0], reverse=False)  
    return turple 
            
files = []
token = RegexpTokenizer('\s+|\-+|\.+|\@+|\t+|\n+|\"+|\>+|\,+|\?+|\:+|\{+|\(+|\[+|\)+|\}+|\]+|\<+|\_+|\!+|\/+|\|+|\\+|\*+|\=+|\^+', gaps = True)
lem = WordNetLemmatizer() 
path='english2/'
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
Query=input("Enter the query-  ")
Query=Query.strip().lower()
query_match=token.tokenize(Query)
queryword=[]
for i in query_match:
    k=k=lem.lemmatize(i.lower())
    queryword.append(k)
querylist={}
for i in queryword:
    if i not in querylist.keys():
        querylist[i]=1.0
    else:
        querylist[i]+=1.0
topk=int(input("How many words should be suggested?   "))
print()
check=True
for word in querylist.keys():
    if(word not in doc[0]):
        check=False
        suggestion=[]
        leng=len(doc[0])
        for indi in tqdm(range(0,leng)):
            each=doc[0][indi]
            len2=len(each)
            len1=len(word)
            dp=create_matrix(len1,len2)
            for ind in range(1,len1+1):
                for jnd in range(1,len2+1):
                    if(each[jnd-1]==word[ind-1]):
                        dp[ind][jnd]=dp[ind-1][jnd-1]
                    else:
                        delect=dp[ind-1][jnd]+1
                        insert=dp[ind][jnd-1]+2
                        op=min(insert,delect)
                        op1=(dp[ind-1][jnd-1]+3)
                        dp[ind][jnd]=min(op,op1)
            sum=dp[len1][len2]
            turple=(sum,)
            turple+=(each,)
            suggestion.append(turple)
        SortTuple(suggestion)
        print()
        print()
        print(word," Suggestions- ")
        print()
        print("\t\t ",suggestion[:topk])
        print()
        print()

if(check):
    print("\n\nNo word is out of dictionary")
            
