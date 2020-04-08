# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:44:55 2020

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

def Tf_cal(freq,length):
    
    res=0                   ######################################### 1) 
    if freq>0:
        res=1
        
    res=freq               ######################################### 2) 
    
    res=freq/length        ######################################### 3) 
    
    return res 



def Idf_cal(freq,Doc):
    
    res=1+math.log10(Doc/(freq))    ######################################### 1) 
    
    res=math.log10(Doc/(freq))      ######################################### 2) 
    
    res=1+math.log10(Doc/(freq+1))  ######################################### 3) 
    
    return res



def read(file):
	fp = codecs.open(file,"r",encoding='utf-8', errors='ignore')
	text = fp.read()
	return text


path1='file.txt'
auth=read(path1)
file_auth=auth.split('\r\n')

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
size=len(files)
sco=[]
for i in range(0,19997):
    s1=float(file_auth[i].split(' ')[1])
    sco.append(s1)
sco=np.array(sco)
maxi=np.max(sco)
for i in tqdm(range(0,size)):
    text=read(files[i])
    score=float(file_auth[i].split(' ')[1])
    files[i]=[files[i],float(score/maxi)]
    m=token.tokenize(text)
    doc.append(m)
    count+=1
print("#Docs are ",count)
word_list={}
count=-1
size=len(doc)
for i in tqdm(range(0,size)):
    count+=1
    index=0
    for tk in doc[i]:
        k=tk.lower()
        k=lem.lemmatize(k)
        #k=tk
        if not k.lower()  in word_list.keys():
            
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
word_list['File_info']=[]
word_list['File_info'].append(len(files))
word_list['File_info'].append(files)

champion_list={}
for i in word_list.keys():
    if i !='File_info':
        champion_list[i]=[]
        for j in word_list[i][1]:
            Tf_Idf=Tf_cal(j[1],len(doc[int(j[0])]))*Idf_cal(word_list[i][0],word_list['File_info'][0])#+files[j[0]][1]
            champion_list[i].append([j[0],Tf_Idf])
        champion_list[i]=Sort_Tuple(champion_list[i])
print('Champion list is created') 
index = open('InvertedIndex_Q1', 'ab')
pickle.dump(word_list, index)                
index.close()
index = open('ChampionList_Q1', 'ab')
pickle.dump(champion_list, index)                
index.close()
print('Champion list have been successfully stored')
            