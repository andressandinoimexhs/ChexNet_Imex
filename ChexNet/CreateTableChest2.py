# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 09:01:21 2021

@author: Andres
"""

import csv
import pandas as pd
import numpy as np

path='./data/Data_Entry_2017_v2020.csv'
#csv_reader = csv.reader(path)
df = pd.read_csv(path)


labels = df['Finding Labels']

# for col in df.columns:
#     print(col)


#%%
def multilabel(label,flag):
    lisi=[]
    kk=""
    if flag>=1:
    
            for i,j in zip(label,range(len(label))):
                
                if i!='|':
                    kk+=i
                    # lisi.append(kk)
                    # kk=""
                else:
                    #print(kk)
                    ll.append(kk)
                    kk=""
            
            #print(kk)
            ll.append(kk)
    #lisi.append(ll)
    return ll


def createrowtable(name,valor):        
    
    binarylabel = valor
    
    df2 =  pd.DataFrame(columns = ['filename', 'A1','A2','A3',
                                   'A4','A5','A6','A7','A8',
                                   'A9','A10','A11','A12',
                                   'A13','A14','A15'])
    
    df2 = df2.append({'filename' : name,
                     'A1' : binarylabel[0][0], 
                     'A2' : binarylabel[0][1],
                     'A3' : binarylabel[0][2],
                     'A4' : binarylabel[0][3], 
                     'A5' : binarylabel[0][4],
                     'A6' : binarylabel[0][5],
                     'A7' : binarylabel[0][6],
                     'A8' : binarylabel[0][7],
                     'A9' : binarylabel[0][8],
                     'A10' : binarylabel[0][9],
                     'A11' : binarylabel[0][10],
                     'A12' : binarylabel[0][11],
                     'A13' : binarylabel[0][12],
                     'A14' : binarylabel[0][13],
                     'A15' : binarylabel[0][14],
                     }, 
                    ignore_index = True)
    return df2


def binarizelabel(valor):
    
    mm = []
    thoraxlabels = ["Atelectasis","Cardiomegaly",
                    "Effusion","Infiltration",
                    "Mass","Nodule","Pneumonia",
                    "Pneumothorax","Consolidation",
                    "Edema","Emphysema","Fibrosis",
                    "Pleural_Thickening","Hernia","No Finding"]
    
    #labelmax = np.int16(np.zeros((1,14)))
    labelmax = np.zeros((1,15))
    #print(type(labelmax))
    
   
    for thlab,ind in zip(thoraxlabels,range(len(thoraxlabels))):
        #print(thlab)
        for i in range(len(valor)):
            if valor[i]==thlab:
                mm.append(ind)
    
    for ix in mm:
        labelmax[0][ix]=np.int16(1)
    
    return labelmax
    
#%%
from tqdm import tqdm

nq =  pd.DataFrame(columns = ['filename', 'A1','A2','A3',
                                   'A4','A5','A6','A7','A8',
                                   'A9','A10','A11','A12',
                                   'A13','A14','A15'])
kk=""
pepito=[]
for i in tqdm(range(len(labels))):
#for i in tqdm(range(0,10)):
    
    ll=[]
    label = df['Finding Labels'][i]  
    filename = df['Image Index'][i] 
    
    flag = label.count('|')
    label2=[label]
    #print(flag)
    if flag>=1:
        label2=multilabel(label,flag)
        
    npm=binarizelabel(label2)
    #print(filename)

    ll=createrowtable(filename,npm)
    #print(ll)
    nq = nq.append(ll,ignore_index=True)

#%%     
nq.to_csv('C:/Users/Andres/Desktop/dataframe15features.csv')
print("Fin del proceso")

#np.savetxt('C:/Users/Andres/Desktop/np.txt', df.values, fmt='%d')                 


#%%