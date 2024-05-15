import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences
import numpy as np
import pickle as pkl

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

    x=[one_hot(word,n=10000) for word in x]
    # x=pad_sequences(x,maxlen=73,padding="pre")
    x=pad_sequences(x,maxlen=max_length+1,padding="pre")
    x=np.array(x)

    return x,y,max_length


def prepare_data_v2(config):

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
    pkl.dumps(open("y_OHE.pkl",'wb'))
    config['OHE']="y_OHE.pkl"
    json.dump(config,open("config.json","w"))
    
    return tempx,tempy,max_length

