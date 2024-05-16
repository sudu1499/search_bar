import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import hashing_trick
from keras.utils import pad_sequences
import numpy as np
import pickle as pkl
import random
import re
enc={}

def prepare_data(config):

    data=pd.read_csv(config["data"])

    # we want only product name and product url for this case 

    x=data['Product Name'].iloc[:config['total_classes']].values
    y=data['Uniq Id'].iloc[:config['total_classes']].values

    y=y.reshape((-1,1))

    ### pad_sequences will add 0's and make all vectors of same dimension

    ## for One_hot approximate total voacabulory and give vocabulory as 'n'
    max_length=1
    for i in x:
        if len(i.split())>=max_length:
                max_length=len(i.split())

    x=[hashing_trick(word,n=10000,hash_function="md5") for word in x]

    # pattern="[!@#$%^&*()\{\[\]\}\-_=+,<>,\|.:;'`~\\\\/0-9]+"
    # x=[re.sub(pattern,"",word) for word in x]

    # x=[encode(word,100000) for word in x]
    # print(x)
    # x=pad_sequences(x,maxlen=73,padding="pre")
    x=pad_sequences(x,maxlen=max_length+1,padding="pre")
    x=np.array(x)
    print("X>SHAPE")
    print(x.shape)
    return x,y,max_length


def prepare_data_v2(config):
    global enc
    x,y,max_length=prepare_data(config)
    tempx=[]
    tempy=[]
    for i in zip(x,y):
          for j in range(20):
               tempx.append(i[0])
               tempy.append(i[1])

    tempx,tempy=np.array(tempx),np.array(tempy)
    ohe=OneHotEncoder()
    tempy=ohe.fit_transform(tempy).toarray()
    pkl.dump(ohe,open("y_OHE.pkl",'wb'))
    config['OHE']="y_OHE.pkl"
    json.dump(config,open("config.json","w"))
    json.dump(enc,open("encoder.json","w"))

    return tempx,tempy,max_length

########## custom encoder 

def encode(snt,n):
    global enc
    temp=[]
    for i in snt.split():
        i=i.lower()
        if i not in enc.keys():
            enc[i]=random.randint(1,n)
        temp.append(enc[i])
    return temp
    