# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:00:25 2020

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

def Sort_Tuple(tuple,m=0):   
    tuple.sort(key = lambda x: x[m],reverse=True)  
    return tuple

dcg50=0.0
dcg=0.0
count=0
path='IR-assignment-3-data.txt'
auth=read(path)
auth=auth.split('\r\n')
file=[]
qry=[]
for i in auth:
    line=i.split(' ')
    if(len(line)>138 and line[1]=='qid:4'):
        file.append([float(line[0]),i])
        count+=1
        dcg+=float(line[0])/(math.log(count+1.0)/math.log(2.0))
        if (count<=50):
            dcg50+=float(line[0])/(math.log(count+1.0)/math.log(2.0))
        qry.append([float(line[0]),float(line[76].split(':')[1])])
qry=Sort_Tuple(qry)
Idcg50=0.0
Idcg=0.0
count=0
tol=0.0
for i in qry:
    if(i[0]!=0):
        tol+=1.0
    count+=1
    Idcg+=i[0]/(math.log(count+1.0)/math.log(2.0))
    if(count<=50):
        Idcg50+=i[0]/(math.log(count+1.0)/math.log(2.0))
print("\n\nFor query id 4- ")
print('\nnDCG value upto 50- ',dcg50/Idcg50)
print('DCG value upto 50- ',dcg50)
print('IDCG value upto 50- ',Idcg50)
print('\nnDCG value for whole file- ',dcg/Idcg)
print('DCG value for whole file- ',dcg)
print('IDCG value for whole file- ',Idcg)
print('\n')
qry=Sort_Tuple(qry,1)
file=Sort_Tuple(file)
files=''
relv=[0,0,0,0,0]
for i in file:
    relv[int(i[0])]+=1
    files+=i[1]+"\n"
f=open('Max_dcg.txt','w+')
f.write(files)
f.close()
rel=0.0
count=0.0
x=[1]
y=[0]
for i in qry:
    count+=1.0
    if(i[0]>0):
        rel+=1.0
    x.append(rel/count)
    y.append(rel/tol)
plt.plot(y,x)
plt.xlabel('Recall ->')
plt.ylabel('Precision ->')
plt.show()

sum=1.0
for i in range(1,len(relv)):
    m=relv[i]
    sum1=1.0
    for j in range(1,m+1):     
        sum1*=j
    sum*=sum1
s=[2]
for i in range(0,relv[0]-1):
    s.append((1+(i+2)*s[i])) 
print("\nNumber of possible sequence with max DCG- ",s[relv[0]-1]*sum)
    