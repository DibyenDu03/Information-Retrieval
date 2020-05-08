# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:44:35 2020

@author: Dibyendu
@Rollno: MT19034
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
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) 

def read(file):
	fp = codecs.open(file,"r",encoding='utf-8', errors='ignore')
	text = fp.read()
	return text

def KNN(train,test,val):
    
    arr=[]
    for i in range(len(train)):
        sum=0
        for j in range(len(train[i])):
            sum+=(train[i][j]-test[j])**2
        arr.append([math.sqrt(sum),train[i][1]])
    arr.sort()

def display(confuse):
    label=['comp.graphics     ', 'sci.med           ', 'talk.politics.misc', 'rec.sport.hockey  ', 'sci.space         ']
    label.sort()
    print(" "*18,end="\t")
    for i in label:
        print(i,end="  ")
    print()
    for i in range(5):
        print(label[i],end="\t ")
        sum=0.0
        for j in range(5):
            sum+=confuse[i][j]
            a=len(str(confuse[i][j]))
            
            print(confuse[i][j],(15-a)*' ',end="\t")
        print("Acc: ",confuse[i][i]/sum*100)
        print()

def KNN(train,test):
    
    arr=[]
    for i in range(len(train)):
        ##vec=np.linalg.norm(train[i][0]-test)
        vec=train[i][0]-test
        vec=vec*vec
        vec=np.sum(vec)
        vec=math.sqrt(vec)
        arr.append([vec,train[i][1]])
        
    arr.sort()
    return arr

def Tf_cal(freq,length):  # Tf calculation
    
    res=0                   ######################################### 1) 
    if freq>0:
        
        res=1
        
    #res=freq               ######################################### 2) 
    
    #res=freq/length        ######################################### 3) 
    
    return res



def Idf_cal(freq,Doc):  # Idf Calculation
    
    res=math.log10((Doc+1)/(freq+1))  ######################################### 3) 
    
    return res

def MutualInfor(N11,N10,N01,N00,N):
    sum=0.0
    if N11!=0:
        sum=N11/N*math.log2((N11*N)/((N11+N10)*(N01+N11))) 
    if N10!=0:
        sum+= N10/N*math.log2((N10*N)/((N11+N10)*(N00+N10)))
    if N01!=0:
        sum+= N01/N*math.log2((N01*N)/((N00+N01)*(N11+N01))) 
    if N00!=0:
        sum+= N00/N*math.log2((N00*N)/((N00+N01)*(N00+N10)))
    
    return sum

def NaiveBayes(cosine,test,feature,toldoc,B):
    
    sum=toldoc.copy()
    sum=np.array(sum)
    sum=np.sum(sum)
    
    max=0
    ind=0
    for i in range(len(toldoc)):
        prob=toldoc[i]/sum
        for k in cosine.keys():
            if k!='File_info' and k!='Cosine_sim' and k!='Total' and k!='Unique' and k not in stop_words and len(k)>1:
                if feature[k]==1:
                    prob=prob*(test[i][k]+1)/(test[i]['Total']+B-1)
        if prob>max:
            max=prob
            ind=i
    
    return ind



    
    
dbfile = open('TF-Idf_Q1', 'rb')      
wordlist = pickle.load(dbfile) 
dbfile.close()
tol=wordlist['File_info'][0]
data_df=[]
for i in range(tol):
    data_df.append([i,int(i/1000)])
random.shuffle(data_df)
ratio=float(input("Enter ratio of dataset should be used for training- "))
train=data_df[:int(tol*ratio)]
test=data_df[int(tol*ratio):]
check=[0 for i in range(tol)]
toldoc=[0 for i in range(5)]
print("\n\t\tTotal train points- ",len(train)," Total test points- ",len(test))

ttol=0

for i in range(len(train)):
    check[train[i][0]]=1
    toldoc[int(train[i][0]/1000)]+=1
    ttol+=1
docclass=[[] for i in range(5)]
doclen=[0 for i in range(5)]
miclass=[[] for i in range(5)]

for k in range(tol):
    if check[k]==1:
        doclen[int(k/1000)]+=wordlist['Cosine_sim'][k]['Total']
    
featuretf={}
featuremi={}
wclass=[{'Total':0} for i in range(5)]

for k in wordlist.keys():
    if k!='File_info' and k!='Cosine_sim' and k not in stop_words and len(k)>1:
        
        featuretf[k]=0
        featuremi[k]=0
        
        arr=[0 for i in range(5)]
        doccou=[0 for i in range(5)]
        for each in wordlist[k][1]:
            ind=int(each[0]/1000)
            if check[each[0]]==1:
                arr[ind]+=each[1]
                doccou[ind]+=1
        cou=0
        for i in range(len(arr)):
            if arr[i]>=1:
                doclen[i]+=arr[i]
                cou+=1
        
        idf=Idf_cal(cou,5)
        cou=doccou.copy()
        cou=np.array(cou)
        cou=np.sum(cou)
        for i in range(len(arr)):
            
            wclass[i][k]=arr[i]
            wclass[i]['Total']+=arr[i]
            docclass[i].append([arr[i]*idf/doclen[i],k,idf,arr[i]])
            N11=doccou[i]
            N10=(cou-doccou[i])
            N01=(toldoc[i]-doccou[i])
            N00=(ttol-doccou[i]-(cou-doccou[i])-(toldoc[i]-doccou[i]))
            N=ttol
            mi=MutualInfor(N11,N10,N01,N00,N)
            miclass[i].append([mi,k,N11,N10,N01,N00,N])

for i in range(5):
    docclass[i].sort(reverse=True)
    miclass[i].sort(reverse=True)
    
#number=int(input("Number of feature need to be selected- "))
number=10
acclist=[]
y=[]
fwordtf=[]
fwordmi=[]
for no in tqdm(range(25,1000,25)):
    sum=0.0
    number=no
    #print()
    #print("\t ",number," top features word will be choosen from each classes ")
    
    
    for i in range(5):
        for k in range(number):
            word=docclass[i][k][1]
            if featuretf[word]!=1:
                featuretf[word]=1
                fwordtf.append(word)
    
            
    for i in range(5):
        for k in range(number):
            word=miclass[i][k][1]
            if featuremi[word]!=1:
                featuremi[word]=1
                fwordmi.append(word)       
            
    confuse=[[0 for j in range(5)] for i in range(5)]
            
    #print("\nNaive Bayes Algorithm-\n")  
    cou=0     
    BB=len(wclass[0])
    for i in test:
        cosine=wordlist['Cosine_sim'][i[0]]
        o=NaiveBayes(cosine,wclass,featuretf,toldoc,BB)
        if o==i[1]:
            cou+=1
            confuse[i[1]][i[1]]+=1
        else:
            confuse[i[1]][o]+=1
    
    #confuse=np.matrix(confuse)
    #print("\n Confusion Matrix-\n ")
    #display(confuse)
    #print()
    
    confuse=[[0 for j in range(5)] for i in range(5)]
    
    print("\nTf-Idf feature based Naive Bayes accuracy- ",cou/len(test)*100)
    #print()
    sum+=cou/len(test)*(100)/8
    
    cou=0     
    for i in test:
        cosine=wordlist['Cosine_sim'][i[0]]
        o=NaiveBayes(cosine,wclass,featuremi,toldoc,BB)
        if o==i[1]:
            cou+=1
            confuse[i[1]][i[1]]+=1
        else:
            confuse[i[1]][o]+=1
            
    #confuse=np.matrix(confuse)
    #print("\n Confusion Matrix-\n ")
    #display(confuse)
    #print()
    
    print("MI feature based Naive Bayes accuracy- ",cou/len(test)*100)
    #print()
    sum+=cou/len(test)*(100)/8
    
    
    cosine=wordlist['Cosine_sim']
    #print("\n\tTf-Idf feature based KNN Algorithm-\n")
    #print()
    #kvalue=int(input("Enter value of K in KNN- "))
    kval=[1,3,5]
    
    lt=len(train)
    t=[]
    for i in range(len(train)):
        vec=[]
        for k in fwordtf:
            a=0
            if k in cosine[train[i][0]].keys():
                
                    freq=cosine[train[i][0]][k]
                    length=cosine[train[i][0]]['Total']
                    a=Tf_cal(freq,length)
                    #a=1
    
                    
            vec.append(a)
        vec=np.array(vec)
        t.append([vec,train[i][1]])
    
    lt=len(test)    
    tt=[]
    for i in range(len(test)):
        vec=[]
        for k in fwordtf:
            a=0
            if k in cosine[test[i][0]].keys():
                
                    freq=cosine[test[i][0]][k]
                    length=cosine[test[i][0]]['Total']
                    a=Tf_cal(freq,length)
                    #a=1
                
                    
            vec.append(a)
        vec=np.array(vec)
        tt.append([vec,test[i][1]])
        
    #print()
    
      
    o_tf=[]
    for i in range(len(test)):
        o=KNN(t,tt[i][0])
        o_tf.append(o)
    
    for kvalue in kval:
        
        confuse=[[0 for j in range(5)] for i in range(5)]
        cou=0.0
        for i in range(len(test)):
            
            c=0
            av=[0 for i in range(5)]
            m=0
            ind=0
            o=o_tf[i][:kvalue]
            for j in o:
    
                av[j[1]]+=1
                if av[j[1]]>m:
                    m=av[j[1]]
                    ind=j[1]
    
                if j[1]==test[i][1]:
                    c+=1
            if c>=kvalue/2:
                cou+=1
                confuse[test[i][1]][test[i][1]]+=1
            else:
                confuse[test[i][1]][ind]+=1
    
        print("\nTf-Idf feature based KNN (k= ",kvalue,") accuracy- ",cou/len(test)*100)
        #print()
        sum+=cou/len(test)*(100)/8
    
    # KNN Algorithm using MI feature selection
    
    #print()
    #print("\n\tMI feature based KNN Algorithm-\n")
    #print()
    
    lt=len(train)
    t=[]
    for i in range(len(train)):
        vec=[]
        for k in fwordmi:
            a=0
            if k in cosine[train[i][0]].keys():
                    
                    freq=cosine[train[i][0]][k]
                    length=cosine[train[i][0]]['Total']
                    a=Tf_cal(freq,length)
                    #a=1
                    
    
                    
            vec.append(a)
        vec=np.array(vec)
        t.append([vec,train[i][1]])
    
    lt=len(test)    
    tt=[]
    for i in range(len(test)):
        vec=[]
        for k in fwordmi:
            a=0
            if k in cosine[test[i][0]].keys():
                    
                    freq=cosine[test[i][0]][k]
                    length=cosine[test[i][0]]['Total']
                    a=Tf_cal(freq,length)
                    #a=1
    
                    
            vec.append(a)
        vec=np.array(vec)
        tt.append([vec,test[i][1]])
    
    #print()    
    
    o_mi=[]
    for i in range(len(test)):
        o=KNN(t,tt[i][0])
        o_mi.append(o)
    
    for kvalue in kval:
        confuse=[[0 for j in range(5)] for i in range(5)]
        cou=0.0
        for i in range(len(test)):
            c=0
            o=o_mi[i][:kvalue]
    
            av=[0 for i in range(5)]
            m=0
            ind=0
    
            for j in o:
                av[j[1]]+=1
                if av[j[1]]>m:
                    m=av[j[1]]
                    ind=j[1]
    
                if j[1]==test[i][1]:
                    c+=1
            if c>=kvalue/2:
                cou+=1
                confuse[test[i][1]][test[i][1]]+=1
            else:
                confuse[test[i][1]][ind]+=1
    
        #print()
        #confuse=np.matrix(confuse)
        #print("\n Confusion Matrix-\n ")
        #display(confuse)
        #print()
        print("\nMI feature based KNN (k= ",kvalue,") accuracy- ",cou/len(test)*100)
        #print()
        sum+=cou/len(test)*(100)/8
    print(sum)
    acclist.append(sum)
    y.append(no)

plt.plot(y,acclist, label = "Accuracy") 
plt.scatter(y,acclist)
plt.ylabel('Accuracy ->') 
plt.xlabel('No features') 
plt.title('Accuracy vs no feature') 
plt.legend() 
plt.show()