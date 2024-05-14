import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences
import numpy as np

def prepare_data(config):
    
    data=pd.read_csv(config["data"])

    # we want only product name and product url for this case 

    x=data['Product Name'].iloc[:].values
    y=data['Product Url'].iloc[:].values

    y=y.reshape((-1,1))

    ### pad_sequences will add 0's and make all vectors of same dimension

    ## for One_hot approximate total voacabulory and give vocabulory as 'n'

    x=[one_hot(word,n=10000) for word in x]
    x=pad_sequences(x,maxlen=100,padding="pre")
    x=np.array(x)
    ohe=OneHotEncoder()
    y=ohe.fit_transform(y)

    return x,y
