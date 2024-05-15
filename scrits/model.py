from keras.models import Sequential
from keras.layers import Dense,Embedding,Flatten,BatchNormalization
import json


def design_model(config):
    
    number_of_classes=config['total_classes']
    max_length=config['max_length']
    model=Sequential()
    model.add(Embedding(input_dim=5000,output_dim=50,input_length=max_length))
    # model.add(Embedding(input_dim=5000,output_dim=50,input_length=73))
    model.add(Flatten())
    model.add(Dense(units=1000,activation='relu'))
    model.add(Dense(units=number_of_classes,activation='softmax'))
    model.compile(optimizer="adam",metrics="accuracy",loss="categorical_crossentropy")
    return model