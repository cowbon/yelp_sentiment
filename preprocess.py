#!/usr/bin/env python

import os
import re
import csv
import json
import nltk
import string
import shutil
import argparse
import logging
import pandas as pd

#from nltk import pos_tag, download
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer


workdir = os.getcwd()
dst = os.path.join(workdir, 'test')
keys = ['business_id', 'text']
nltk.data.path.append(os.path.join(workdir, 'nltk_data'))


def check_env(path):
    path = os.path.join(path, 'nltk_data')
    if not os.path.exists(path):
        nltk.download('punkt', download_dir=path)
        nltk.download('stopwords', download_dir=path)
        nltk.download('wordnet', download_dir=path)
        nltk.download('averaged_perceptron_tagger', download_dir=path)


def init_pipeline():
    punc = string.punctuation.replace('\'', '').replace('.', '')
    stop = set(stopwords.words('english'))
    lemmatizer = SnowballStemmer('english')#WordNetLemmatizer()
    return (punc, stop, lemmatizer)


def prepocess(text, extract_senti=False):
    (punc, stop, lemmatizer) = init_pipeline()
    text = re.sub('[0-9]+', '', text)
    origin = nltk.pos_tag(word_tokenize(text))
    sentence, filtered, prev = '', [], None
    for x in origin:
        if x[0] in ',.?!;':
            sentence = sentence.translate(str.maketrans('', '', punc))
            if sentence != ' ':
                filtered.append(sentence[1:] + '\n')
            sentence = ''
        else:
            low = x[0].lower()
            if (low not in stop and low != 'not'):
                sentence += ' {}'.format(lemmatizer.stem(low))

    return filtered


def split_raw(text):
    sentence, filtered = '', []
    origin = word_tokenize(text)
    for x in origin:
        if x in ',.?!;:':
            filtered.append(sentence[1:] + '\n')
            sentence = ''
        else:
            sentence =  '{} {}'.format(sentence, x)

    return filtered


def write_to_file(data, business_id):
    fp = open(os.path.join(dst, business_id), 'a+')
    for i in data:
        fp.write(i)


def gen_preprocessed(text, business_id):
    data = split_raw(text)#prepocess(text)
    write_to_file(data, business_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['preprocess', 'train', 'predict'])
    parser.add_argument('-i', '--input', type=str, help='Input training file', default=os.path.join(workdir, 'pratice.json'))
    parser.add_argument('-k', type=int, help='Number of k for K-Fold', default=10)
    args = parser.parse_args()

    print(args.task)
    if args.task == 'preprocess':
        try:
            if os.path.exists(dst):
                shutil.rmtree(dst)
            check_env(workdir)
            if not os.path.exists(args.input):
                logging.error('{} does not exists!'.format(args.input))
                exit(0)

            print('WTF')
            data = []
            print('laaaa')
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

        sentance, labels = [], []
        for root, dirs, files in os.walk(args.input):
            for name in files:
                with open(os.path.join(root, name), 'r') as f:
                    pass


if __name__ == '__main__':
    main()
