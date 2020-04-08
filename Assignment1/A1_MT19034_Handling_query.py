# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 18:26:03 2020

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


def check(position,ans,query,index,len1,pos,j):
    if(index==len1):
        return 1
    if(pos in position[query[index]][1][ans[j][index]][1]):
        return check(position,ans,query,index+1,len1,pos+1,j)
    else:
        return 0
    
    
        


dbfile = open('InvertedIndex_Q2', 'rb')      
wordlist = pickle.load(dbfile) 
dbfile.close()

dbfile = open('PositionalIndex_Q2', 'rb')      
position = pickle.load(dbfile) 
dbfile.close()

def andOperation(x,len1,y,len2):
    result=[]
    l1=0
    l2=0
    while(l1<len1 and l2<len2):
        if(x[l1]<y[l2]):
            l1+=1
        else:
            if(x[l1]==y[l2]):
                result.append(x[l1])
                l1+=1
                l2+=1
            else:
                l2+=1
    return result

Query=input("Please enter a phase- ")
token = RegexpTokenizer('\s+|\-+|\.+|\@+|\t+|\n+|[0-9]+|\"+|\>+|\,+|\?+|\:+|\{+|\(+|\[+|\)+|\}+|\]+|\<+|\_+|\!+|\/+|\|+|\\+|\*+|\=+|\^+', gaps = True)
query=token.tokenize(Query.strip().lower())

result=[]
if query[0] in wordlist.keys():
    result=wordlist[query[0]][1]
for i in range(1,len(query)):
    if query[i] in wordlist.keys():
        result=andOperation(result,len(result),wordlist[query[i]][1],len(wordlist[query[i]][1]))
    else:
        result=[]
    
mapping=[]
if len(result)>0:
    for j in range(0,len(query)):
        freq=[]
        len1=wordlist[query[j]][0]
        for i in range(0,len1):
            if(wordlist[query[j]][1][i] in result):
                freq.append(i)
        mapping.append(freq)
    ans=[]
    for i in range(0,len(result)):
        s=[]
        for j in range(0,len(query)):
            s.append(mapping[j][i])
        ans.append(s)
    
    final=[]
    posi=[]
    for j in range(0,len(ans)): 
        for i in position[query[0]][1][ans[j][0]][1]:
            if(check(position,ans,query,0,len(query),i,j)==1):
                final.append(result[j])
                posi.append(i)
    documentlist=[]
    print()
    print("This phase is present ",len(final),"times in all documents")
    print()
    for i in range(0,len(final)):
        documentlist.append(wordlist['File_info'][1][final[i]])
        print((i+1),") ",wordlist['File_info'][1][final[i]]," at position ",posi[i])
else:
    print()
    print("This phase is not present in any document")
    

    
            
