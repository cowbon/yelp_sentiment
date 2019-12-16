#!/usr/bin/env python

'''
Author: Ian Chin Wang
The base case for all models
'''
from abc import abstractmethod
from keras.models import Sequential


class BaseModel:
    def __init__(self):
        self.model = Sequential()
    
    def summary(self):
        self.model.summary()
        
    def compile(self, **kwargs):
        self.model.compile(**kwargs)

    @abstractmethod
    def save(self):
        pass

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
