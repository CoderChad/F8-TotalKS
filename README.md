# F8: TotalKS

Specific things we would like this function will do:


a) Take in three inputs as arguments: X1 (Sample 1), X2 (Sample2) and a flag, where 1 = classical KS and 2 =
total KS.


b) If the flag in a) is set to 1, compute the largest distance between the two cumulative sample distributions. If
the flag is set to 2, compute the total difference between the two sample distributions.


c) The function should return the value computed in b)


d) Assumptions: The two samples can be input as 1D numpy arrays, dataframes or even lists. By default, you
should assume the inputs to be 1D numpy arrays of arbitrary – but equal – length.


e) Make sure the function has a clear header as to what inputs the function assumes, what outputs it produces
and when it was written.
 

# Takes in three inputs as arguments: X1 (Sample 1), X2 (Sample2) and a flag, where 1 = classical KS and 2 = total KS. If the flag in is set to 1, computes the largest distance between the two cumulative sample distributions. If the flag is set to 2, computes the total difference between the two sample distributions.


# Code returns the percentile


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

%matplotlib inline

data1=np.genfromtxt('./data/kSinput1.csv',delimiter=",")
data2=np.genfromtxt('./data/kSinput2.csv',delimiter=",")

X1,X2=data1[:,0],data1[:,1]
eX1,eX2=data2[:,0],data2[:,1]


def totalKs(X1,X2,flag):
    
    headers=['source','index','value']

    labels_X1=['X1']*len(X1)
    labels_X2=['X2']*len(X2)
    indices=np.arange(len(X1))

    df_labels=np.concatenate((labels_X1,labels_X2))
    df_indices=np.concatenate((indices,indices))
    df_vals=np.concatenate((X1,X2))

    d1={'source':df_labels,'index':df_indices,'value':df_vals}
    df1=pd.DataFrame(d1)
    df1=df1.sort_values('value',axis=0,ascending=True,ignore_index=True)
    
    D=[]
    Y1=[]
    Y2=[]
    Y1_val=0
    Y2_val=0
    
    for source in np.array(df1['source']):

        if source=='X1':
            Y1_val+=0.1
            Y1.append(Y1_val)
            Y2.append(Y2_val)
        elif source=='X2':
            Y2_val+=0.1
            Y1.append(Y1_val)
            Y2.append(Y2_val)

        D.append(abs(Y1_val-Y2_val))
    
    if flag==1:
        return max(D)
    
    elif flag==2:
        return np.mean(D)
