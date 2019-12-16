#!/usr/bin/env python
'''
Author: Ian Chin Wang
A front-end of our project, including preprocessing and classification
'''
import os
import re
import csv
import json
import nltk
import string
import shutil
import pickle
import argparse
import logging
import numpy as np
import pandas as pd

from models import cnn, rnn
from nltk import pos_tag, download
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Sequential
from keras.utils import to_categorical
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from gensim import models


workdir = os.getcwd()
dst = os.path.join(workdir, 'test')
keys = ['business_id', 'text']
download_list = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
nltk.data.path.append(os.path.join(workdir, 'nltk_data'))
word2vec_path = 'GoogleNews-vectors-negative300.bin.gz'
url = ''
MAX_SEQ_LEN = 50
EMBEDDING_DIM = 300
EPOCH_NUM = 10
BATCH_SIZE = 128
NUM_CATEGORY = 3
NLTK_DIR_NAME= 'nltk_data'


def check_env(path):
    path = os.path.join(path, NLTK_DIR_NAME)
    if not os.path.exists(path):
        for i in download_list:
            nltk.download(i, download_dir=path)


def init_pipeline():
    punc = string.punctuation.replace('\'', '').replace('.', '')
    stop = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return (punc, stop, lemmatizer)


def preprocess(text, extract_senti=False):
    (punc, stop, lemmatizer) = init_pipeline()
    text = re.sub('[0-9]+', '', text)
    origin = nltk.pos_tag(word_tokenize(text))
    # origin = word_tokenize(text)
    sentence, filtered, prev = '', [], None
    for x in origin:
        if x[0] in ',...?!;':
            sentence = sentence.translate(str.maketrans('', '', punc))
            if sentence != ' ':
                filtered.append(sentence[1:])
            sentence = ''
        else:
            low = x[0].lower()
            if (low not in stop or low == 'not'):
                # sentence = '{} {}'.format(sentence, lemmatizer.stem(low))
                if (x[1] == 'NNS'):
                    sentence = '{} {}'.format(sentence, lemmatizer.lemmatize(low, pos=wordnet.NOUN))
                elif (x[1] == 'RBR' or x[1] == 'RBS'):
                    sentence = '{} {}'.format(sentence, lemmatizer.lemmatize(low, pos=wordnet.ADV))
                elif (x[1].startswith('V') and x[1] != 'VB'):
                    sentence = '{} {}'.format(sentence, lemmatizer.lemmatize(low, pos=wordnet.VERB))
                elif (x[1] == 'JJR' or x[1] == 'JJS'):
                    sentence = '{} {}'.format(sentence, lemmatizer.lemmatize(low, pos=wordnet.ADJ))
                else:
                    sentence = '{} {}'.format(sentence, low)


    if len(sentence) > 0:
        sentence = sentence.translate(str.maketrans('', '', punc))
        filtered.append(sentence[1:])

    return filtered if len(filtered) > 0 else None


def parse_label(root, files, data):
    with open(os.path.join(root, files), 'r') as f:
        reader = csv.reader(f.readlines(), delimiter=',')
        for row in reader:
            # Convert to vctor
            text = preprocess(row[0])
            if text and len(row) > 1 and row[1] != '':
                data = data.append({'sentence': text[0], 'label': row[1], 'business_id': files}, ignore_index=True)
    return data


def gen_word_vector(data, vector):
    ret = [vector[w] if w in vector else np.zeros(300) for w in data]
    length = len(ret)
    average = np.divide(np.sum(ret, axis=0), length)
    return average


def split_raw(text):
    sentence, filtered = '', []
    origin = word_tokenize(text)
    for x in origin:
        if x in ',.?!;:':
            filtered.append(sentence[1:] + '\n')
            sentence = ''
        else:
            sentence =  '{} {}'.format(sentence, x)

    if len(sentence) > 0:
        filtered.append(sentence[1:] + '\n')
        sentence = ''

    return filtered if len(filtered) > 0 else None


def write_to_file(data, business_id):
    fp = open(os.path.join(dst, business_id), 'a+')
    for i in data:
        fp.write(i)


def gen_preprocessed(text, business_id):
    data = split_raw(text)
    write_to_file(data, business_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['preprocess', 'train', 'validate'])
    parser.add_argument('-i', '--input', type=str, help='Input training file', default=os.path.join(workdir, 'pratice.json'))
    parser.add_argument('-k', type=int, help='Leaves k precent data for testing', default=10)
    parser.add_argument('-t', '--type', choices=['cnn', 'lstm'], type=str, help='The training algorithm', default='cnn')
    parser.add_argument('-m', '--input-model', type=str, help='Filename of the trained model')
    args = parser.parse_args()

    print(args.task, 'mode')
    check_env(workdir)
    if args.task == 'preprocess':
        # Break the reviews into sentences
        try:
            if os.path.exists(dst):
                shutil.rmtree(dst)
            if not os.path.exists(args.input):
                logging.error('{} does not exists!'.format(args.input))
                exit(0)

            data = []

            with open(args.input, 'r') as f:
                for lines in f:
                    data.append({key: json.loads(lines)[key] for key in keys})
    
            d = pd.DataFrame.from_dict(data)
            print(1)
            if not os.path.exists(dst):
                os.mkdir(dst)
            d.apply(lambda x: gen_preprocessed(x['text'], x['business_id']), axis=1)
        except ValueError as e:
            logging.error(e)
    elif args.task == 'train':
        # Load input
        data = pd.DataFrame(columns=['sentence', 'label', 'business_id'])
        for root, dirs, files in os.walk(args.input):
            for name in files:
                print('Prasing file ', name) 
                data = parse_label(root, name, data)

        # Split training and testing data
        # kf = KFold(n_splits=int(args.k))

       # Split training set and testing set
        test_size = int(args.k)/100.0
        training_set, testing_set = train_test_split(data, train_size=1-test_size,
                                        test_size=test_size, stratify=None)#data['label'])
        print(testing_set)
        training_words = [word for words in training_set['sentence'] for word in words.split()]
        training_vocabs = list(set(training_words))
        testing_words = [word for words in testing_set['sentence'] for word in words.split()]
        testing_vocabs = list(set(testing_words))
        tokenizer = Tokenizer(num_words=len(training_vocabs+testing_vocabs), lower=True, char_level=False)
        tokenizer.fit_on_texts(training_set['sentence'].tolist())
        training_sequences = tokenizer.texts_to_sequences(training_set['sentence'].tolist())
        train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQ_LEN)

        testing_sequences = tokenizer.texts_to_sequences(testing_set['sentence'].tolist())
        test_cnn_data = pad_sequences(testing_sequences, maxlen=MAX_SEQ_LEN)

       # Dump tokenizer
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        training_label = to_categorical(training_set['label'].tolist())
        testing_label_list = np.array(testing_set['label'].tolist(), dtype='int32')
        testing_label = to_categorical(testing_label_list)
       
        word_index = tokenizer.word_index
        num_words = len(word_index)+1

        embeddings = vector = None

        # Load pretrained word embeddings
        print('Loading pretrained vector')
        if not os.path.exists(word2vec_path):
            import gensim.downloader as api
            vector = api.load('word2vec-google-news-300')
        else:
            vector = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        embeddings = np.zeros((num_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            embeddings[i, :] = vector[word] if word in vector else np.random.rand(EMBEDDING_DIM)


        if args.type == 'cnn':
            model = cnn.CNNModel(max_input_length=MAX_SEQ_LEN, 
                            num_words=num_words, embedding_dim=EMBEDDING_DIM, embeddings=embeddings, 
                            trainable=(vector is None))
        else:
            model = rnn.LSTMModel(max_input_length=MAX_SEQ_LEN,
                            num_words=num_words,embedding_dim=EMBEDDING_DIM, embeddings=embeddings, 
                            trainable=(vector is None))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        model_log = model.fit(train_cnn_data, training_label, epochs=EPOCH_NUM,
                validation_split=0.1, shuffle=True, batch_size=BATCH_SIZE,
                validation_data=(test_cnn_data, testing_label))

        score, acc = model.evaluate(test_cnn_data, testing_label)
        print('Score:', score, ' Accuracy:', acc)
        pred = model.predict(test_cnn_data).argmax(axis=1)
        confusion = confusion_matrix(testing_label_list, pred, labels=np.arange(3))
        report = classification_report(testing_label_list, pred, labels=np.arange(3))
        print(confusion)
        print(report)
        # Dump the model to the disk
        model.save()
    else:
        # Load tokenizer
        check_env(workdir)
        if not os.path.exists(args.input):
            logging.error('{} does not exists!'.format(args.input))
            exit(0)
        data = []
        with open(args.input, 'r') as f:
            data.append({key: json.loads(lines)[key] for key in keys})

        d = pd.DataFrame.from_dict(data)
        d = d.apply(lambda x: prepocess(x['text']), axis=1)
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        testing_sequences = tokenizer.texts_to_sequences(d['text'].tolist())
        testing_data = pad_sequences(testing_sequences, maxlen=MAX_SEQ_LEN)

        model = load_model(args.m, compile=true)
        result = model.predict(testing_data)
        


if __name__ == '__main__':
    main()
