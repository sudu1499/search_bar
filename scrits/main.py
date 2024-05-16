from scrits.data_collection import prepare_data,prepare_data_v2
from scrits.model import design_model
import json
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import hashing_trick
from keras.utils import pad_sequences
import numpy as np
import pickle as pkl

config=json.load(open("config.json","r"))

# x,y,max_length=prepare_data(config)
x,y,max_length=prepare_data_v2(config)

if config['max_length']!= max_length:
    config['max_length']=max_length+1
    json.dump(config,open("config.json","w"))


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33)

#  got know that max 72 words for each sentence
model=design_model(config)
model.summary()


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20)



def prepare_data_for_testing(string):

    x=[hashing_trick(w,10000,hash_function="md5") for w in string.split() ]
    x=pad_sequences([x],maxlen=44,padding="pre")
    x=np.array(x)
    return x


xtest=prepare_data_for_testing("Woodstock- ")

r=model.predict(xtest.reshape(1,-1))
result=np.zeros((1,config['total_classes']))
result[0,np.argmax(r)]=1
ohe=pkl.load(open(config["OHE"],'rb'))