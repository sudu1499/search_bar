from keras.models import Sequential
from keras.layers import Dense,Embedding
import json


def design_model(config):
    
    number_of_classes=config['total_classes']
    model=Sequential()
    model.add(Embedding(input_dim=10000,output_dim=300))
    model.add(Dense(units=number_of_classes,activation='softmax'))
    model.compile(optimizer="adam",metrics="accuracy",loss="categorical_crossentropy")