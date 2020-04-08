# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 04:17:16 2020

@author: Dibyendu
"""

from num2words import num2words
from word2number import w2n


def Number2Word(list19):
    new_list=[]
    for each in list19:
        if(each.isnumeric()):
            m=num2words(each).replace(',','')
            new_list.append(m)
        else:
            new_list.append(each)
    #print(new_list)
    return new_list

def Word2Number(list18):
    list20=Number2Word(list18)
    new=[]
    count=0
    i=0
    while(i<len(list20)):
        w=list20[i]
        try: 
            result = w2n.word_to_num(w)
            count+=1.0
            i+=1
            if(i<len(list20)):
                w=w+' '+list20[i]
            while(i<len(list20) and result!=w2n.word_to_num(w)):
                result=w2n.word_to_num(w)
                i+=1
            new.append(str(result))
            i-=1
        except: 
            new.append(list20[i])
        i+=1
    #print(new)
    return new

'''
s='I have one hundred rupees which is greater than twenty one two thousand less 2 billion hello 50,000 1987'
list=['i', 'have', 'one', 'hundred', 'rupee', 'which', 'is', 'greater', 'than', 'twenty', 'one', 'two', 'thousand', 'le', '2', 'billion', 'hello', '1987']
#list=s.split(' ')
list=['11','18','91']
Word2Number(list)
'''

