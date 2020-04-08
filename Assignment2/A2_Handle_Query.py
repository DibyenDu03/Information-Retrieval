# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:50:13 2020

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
from A2_handle_numerical import Word2Number

def count_cosine(cosine,querylist,wordlist):
    list1=[]
    length=wordlist['File_info'][0]
    highest=cal_highest(cosine,querylist,length)
    for ind in range(0,length):
        sum=0.0
        for k in cosine[ind].keys():
            if k!='Unique' and k!='Total':
                tf=Tf_cal(cosine[ind][k],cosine[ind]['Total'],ind,highest)
                idf=Idf_cal(wordlist[k][0],wordlist['File_info'][0])
                sum+=(tf*idf)*(tf*idf)
        sum=math.sqrt(sum)
        list1.append(sum)
    sum1=0.0
    for k in querylist.keys():
        if k!='Unique' and k!='Total' and k in wordlist.keys():
            tf=Tf_cal(querylist[k],querylist['Total'],length,highest)
            idf=Idf_cal(wordlist[k][0],wordlist['File_info'][0])
            sum1+=(tf*idf)*(tf*idf)
    sum1=math.sqrt(sum1)
    list1.append(sum1)
    return list1,highest

def cal_highest(cosine,querylist,length):
    high=[]
    for ind in range(0,length):
        m=0.0
        for k in cosine[ind].keys():
            if k!='Unique' and k!='Total':
                if m<cosine[ind][k]:
                    m=cosine[ind][k]
        high.append(m)
    m=0.0
    for k in querylist.keys():
        if k!='Unique' and k!='Total' and k in wordlist.keys():
            if m<querylist[k]:
                    m=querylist[k]
    high.append(m)
    return high
    
        
    
    
            

def SortTuple(turple):  
    turple.sort(key = lambda x: x[0], reverse=True)  
    return turple  

def andOperation(x,len1,y,len2):
    result=[]
    l1=0
    l2=0
    count=0.0
    while(l1<len1 and l2<len2):
        if(x[l1]<y[l2]):
            l1+=1
        else:
            if(x[l1]==y[l2]):
                result.append(x[l1])
                count+=1.0
                l1+=1
                l2+=1
            else:
                l2+=1
    return result,count

def orOperation(x,len1,y,len2):
    result=[]
    l1=0
    l2=0
    com=0.0
    while(l1<len1 and l2<len2):
        if(x[l1]<y[l2]):
            result.append(x[l1])
            com+=1.0
            l1+=1
        else:
            if(x[l1]==y[l2]):
                result.append(x[l1])
                com+=1.0
                l1+=1
                l2+=1
            else:
                result.append(y[l2])
                com+=1.0
                l2+=1
    while(l1<len1):
         result.append(x[l1])
         com+=1.0
         l1+=1
    while(l2<len2):
        result.append(y[l2])
        com+=1.0
        l2+=1
    return result,com

def Jacard(cosine,querylist,topk):
    doc=[]
    length=len(cosine)
    for ind in range(0,length):
        list1=[]
        list2=[]
        len1=cosine[ind]['Unique']+2
        len2=querylist['Unique']+2
        list1=list(cosine[ind].keys())
        list2=list(querylist.keys())
        list1.sort()
        list2.sort()
        intersect,leng=andOperation(list1,len1,list2,len2)
        union,leng1= orOperation(list1,len1,list2,len2)
        jacard=(leng-2)/(leng1-2)
        turple=(jacard,)
        turple+=(ind,)
        doc.append(turple)
    SortTuple(doc)
    return doc[:(topk)]

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


   
def Tf_IdF(cosine,querylist,topk,wordlist,highest,gweight=1):
    doc=[]
    length=len(cosine)
    for ind in range(0,length):
        sum=0.0
        for i in querylist.keys():
            if i in cosine[ind].keys() and i!='Unique' and i!='Total':
                gval=attention_title(cosine,wordlist,ind,i,gweight)
                gval=1.0
                tf=Tf_cal(cosine[ind][i],cosine[ind]['Total'],ind,highest)*gval
                idf=Idf_cal(wordlist[i][0],wordlist['File_info'][0])
                sum+=tf*idf
        turple=(sum,)
        turple+=(ind,)
        doc.append(turple)
    SortTuple(doc)
    return doc[:(topk)]

def attention_title(cosine,wordlist,index,word,gweight):
    weight=0.0
    if word in wordlist.keys():
        if word not in wordlist['File_info'][2][index].keys():
            weight=(1-gweight)
        else:
            if wordlist['File_info'][2][index][word]>=cosine[index][word]:
                weight=(gweight)
            else:
                weight=1.0
                
        
    return weight

def cosine_similarity(cosine,querylist,topk,wordlist,lis1,highest,gweight=1):
    doc=[]
    length=len(cosine)
    for ind in range(0,length):
        sum=0.0
        for i in querylist.keys():
            if i in cosine[ind].keys() and i!='Unique' and i!='Total':
                gval=attention_title(cosine,wordlist,ind,i,gweight)
                gval=1.0
                tf1=Tf_cal(cosine[ind][i],cosine[ind]['Total'],ind,highest)*gval
                tf=Tf_cal(querylist[i],querylist['Total'],wordlist['File_info'][0],highest)
                idf=Idf_cal(wordlist[i][0],wordlist['File_info'][0])
                sum+=(tf*idf)*(tf1*idf)
        if sum>0:
            sum=sum/(list1[ind]*list1[length])
        else:
            sum=0
        turple=(sum,)
        turple+=(ind,)
        doc.append(turple)
    SortTuple(doc)
    return doc[:topk]
                
                
                
                
        

Query=input("Enter the query-  ")
Query=Query.strip().lower()
Query=Query.replace(',','')
token = RegexpTokenizer('\s+|\-+|\.+|\@+|\t+|\n+|\"+|\>+|\,+|\?+|\:+|\{+|\(+|\[+|\)+|\}+|\]+|\<+|\_+|\!+|\/+|\|+|\\+|\*+|\=+|\^+', gaps = True)
lem = WordNetLemmatizer()
query_match=token.tokenize(Query)
query=[]
for i in query_match:
    k=k=lem.lemmatize(i.lower())
    query.append(k)
    
query=Word2Number(query)

dbfile = open('InvertedIndex_Q1', 'rb')      
wordlist = pickle.load(dbfile) 
dbfile.close()

dbfile = open('Cosine_index_Q1', 'rb')      
cosine = pickle.load(dbfile) 
dbfile.close()
querylist={}
querylist['Unique']=0.0
querylist['Total']=0.0
for i in query:
    if i not in querylist.keys():
        querylist[i]=1.0
        querylist['Unique']+=1.0
        querylist['Total']+=1.0
    else:
        querylist[i]+=1.0
        querylist['Total']+=1.0
list1,highest=count_cosine(cosine,querylist,wordlist)
top_match=int(input("How many document should be retrieved?   "))
filename=wordlist['File_info'][1]
doc=Jacard(cosine,querylist,top_match)
doc1=Tf_IdF(cosine,querylist,top_match,wordlist,highest)
doc2=cosine_similarity(cosine,querylist,top_match,wordlist,list1,highest)
print()
print()
print('Jaccard Coefficient result:  top ',top_match," documents are-")
print()
for i in range(0,top_match):
    print("\t",(i+1),") ",filename[doc[i][1]])
print()
print()
print('Tf-Idf based document retrieval result:  top ',top_match," documents are-")
print()
for i in range(0,top_match):
    print("\t",(i+1),") ",filename[doc1[i][1]])
print()
print()
print('Cosine similarity result:  top ',top_match," documents are-")
print()
for i in range(0,top_match):
    print("\t",(i+1),") ",filename[doc2[i][1]])



        
    


