from scrits.data_collection import prepare_data
from scrits.model import design_model
import json
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences

config=json.load(open("config.json","r"))

x,y,max_length=prepare_data(config)

if config['max_length']!= max_length:
    config['max_length']=max_length+1
    json.dump(config,open("config.json","w"))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33)

#  got know that max 72 words for each sentence
model=design_model(config)
model.summary()
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=200)

x=[one_hot(word,n=5000) for word in  "Woodstock- Collage 500 pc Puzzle".split()]

x=pad_sequences(x,maxlen=max_length+1,padding="pre")
r=model.predict(x)