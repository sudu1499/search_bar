from scrits.data_collection import prepare_data
from scrits.model import design_model
import json
from sklearn.model_selection import train_test_split


config=json.load(open("config.json","r"))

x,y=prepare_data(config)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33)

#  got know that max 72 words for each sentence
model=design_model(config)

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=32)

