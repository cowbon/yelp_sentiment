#!/usr/bin/env python

'''
Author: Ian Chin Wang
Implementation for Convolutional Neural Network
'''

from .BaseModel import BaseModel
from datetime import datetime
from keras.layers import Embedding, Dense, Conv1D, Flatten, Dropout, GlobalMaxPooling1D, concatenate
from keras.models import Sequential


class CNNModel(BaseModel):
    model_name = 'CNN'
    def __init__(self, max_input_length, num_words, embedding_dim, embeddings=None, trainable=False):
        if embeddings is not None:
            embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_input_length,
                                trainable=trainable)
        else:
            embedding_layer = Embedding(num_words,
                                embedding_dim,
                                input_length=max_input_length,
                                trainable=True)
        super().__init__()
        self.model.add(embedding_layer)
        self.model.add(Conv1D(64, 3, border_mode='same'))
        self.model.add(Conv1D(32, 3, border_mode='same'))
        self.model.add(Conv1D(16, 3, border_mode='same'))
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(180,activation='sigmoid'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(3,activation='sigmoid'))

    def save(self, train_percentage):
        now = datetime.now()
        self.model.save('{}_{}_{}%'.format(now.strftime('%Y-%m-%d_%H:%M:%S'), model_name, train_percentage))


