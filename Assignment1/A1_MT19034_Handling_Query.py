# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:09:12 2020

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


def andOperation(x,len1,y,len2,com):
    result=[]
    l1=0
    l2=0
    while(l1<len1 and l2<len2):
        com+=1
        if(x[l1]<y[l2]):
            l1+=1
        else:
            if(x[l1]==y[l2]):
                result.append(x[l1])
                l1+=1
                l2+=1
            else:
                l2+=1
    return result,com

def orOperation(x,len1,y,len2,com):
    result=[]
    l1=0
    l2=0
    while(l1<len1 and l2<len2):
        com+=1
        if(x[l1]<y[l2]):
            result.append(x[l1])
            l1+=1
        else:
            if(x[l1]==y[l2]):
                result.append(x[l1])
                l1+=1
                l2+=1
            else:
                result.append(y[l2])
                l2+=1
    while(l1<len1):
         result.append(x[l1])
         l1+=1
    while(l2<len2):
        result.append(y[l2])
        l2+=1
    return result,com

def notOperation(x,len1,len3,com):
    res=[]
    result=[]
    for i in range(0,len3):
        res.append(1)
    for i in range(0,len1):
        res[x[i]]=0
    for i in range(0,len3):
        com+=1
        if(res[i]==1):
            result.append(i)
    return result,com
        
        
        
dbfile = open('InvertedIndex_Q1', 'rb')      
wordlist = pickle.load(dbfile) 
dbfile.close()
print("Please enter word with Starting letter Capital")
print("For operation follow bellow keyword- ")
print(" 1) AND- 'and' \n 2) OR- 'or' \n 3) NOT- 'not' ")
Query=input("Please enter in a valid way ")
query=Query.strip().split(' ')
stack=[]
operator=[]
pre={'or':1,'and':2,'not':3}
com=0
for i in range(0,len(query)):
    #print(query[i])
    if (query[i][0]>='A' and query[i][0]<='Z'):
        res=1
        if(query[i].lower() in wordlist.keys()):
            res=wordlist[query[i].lower()][1]
        else:
            res=[]
        stack.append(res)
    else:
        if(len(operator)>0):
            op=operator[len(operator)-1]
            while(pre[op]>=pre[query[i]]):
                #print(op," has been performed ")
                resu=[]
                if(op=='not'):
                    res=stack.pop()
                    resu,com=notOperation(res,len(res),wordlist['File_info'][0],com)
                    stack.append(resu)
                else:
                    res=stack.pop()
                    res1=stack.pop()
                    if(op=='or'):
                        resu,com=orOperation(res,len(res),res1,len(res1),com)
                    else:
                        resu,com=andOperation(res,len(res),res1,len(res1),com)
                    stack.append(resu)
                    
                operator.pop()
                if(len(operator)>0):
                    op=operator[len(operator)-1]
                else:
                    break
        operator.append(query[i])
        
while(len(operator)>0):
    op=operator.pop()
    #print(op," XXX has been performed")
    resu=[]
    if(op=='not'):
        res=stack.pop()
        resu,com=notOperation(res,len(res),wordlist['File_info'][0],com)
        stack.append(resu)
    else:
        res=stack.pop()
        res1=stack.pop()
        if(op=='or'):
            resu,com=orOperation(res,len(res),res1,len(res1),com)
        else:
            resu,com=andOperation(res,len(res),res1,len(res1),com)
        stack.append(resu)

documentlist=[]
resu=stack[0]
print()
print("Number of comparisons is needed- ",com)
print()
print("The number of document retrieve ",len(resu))
print()
documentlist=[]
for i in range(0,len(resu)):
    documentlist.append(wordlist['File_info'][1][resu[i]])
    print((i+1),") ",wordlist['File_info'][1][resu[i]])

    


