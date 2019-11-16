#!/usr/bin/env python

import os
import re
import string
import shutil
import argparse
import logging
import pandas as pd

from nltk import pos_tag, download
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer


workdir = os.getcwd()
dst = os.path.join(workdir, 'test')


def check_env(path):
    path = os.path.join(path, 'nltk_data')
    if not os.path.exists(path):
        download('punkt', download_dir=path)
        download('stopwords', download_dir=path)
        download('wordnet', download_dir=path)
        download('averaged_perceptron_tagger', download_dir=path)


def init_pipeline():
    punc = string.punctuation.replace('\'', '').replace('.', '')
    stop = set(stopwords.words('english'))
    lemmatizer = SnowballStemmer('english')#WordNetLemmatizer()
    return (punc, stop, lemmatizer)


def prepocess(text, extract_senti=False):
    (punc, stop, lemmatizer) = init_pipeline()
    text = re.sub('[0-9]+', '', text)
    origin = pos_tag(word_tokenize(text))
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


def write_to_file(data, business_id):
    fp = open(os.path.join(dst, business_id), 'a+')
    for i in data:
        fp.write(i)


def gen_preprocessed(text, business_id):
    data = prepocess(text)
    write_to_file(data, business_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input training file', default=os.path.join(workdir, 'pratice.json'))
    args = parser.parse_args()


    try:
        if os.path.exists(dst):
            shutil.rmtree(dst)
        check_env(workdir)
        if not os.path.exists(args.input):
            logging.error('{} does not exists!'.format(args.input))
            exit(0)

        data = pd.read_json(args.input, lines=True)
        if not os.path.exists(dst):
            os.mkdir(dst)
        data.apply(lambda x: gen_preprocessed(x['text'], x['business_id']), axis=1)

    except pd.errors.EmptyDataError as e:
        logging.error(e)


if __name__ == '__main__':
    main()
