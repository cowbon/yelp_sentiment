#!/usr/bin/env python

import os
import re
import string
import argparse
import logging
import pandas as pd

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


workdir = '/tmp/tmux-12016' #os.getcwd()
dst = os.path.join(workdir, 'test')


def check_env(path):
    os.chdir(path)
    def download():
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')

    if not os.path.exists(os.path.join(path, 'nltk_data')):
        download()


def init():
    punc = string.punctuation.replace('\'', '').replace('.', '')
    stop = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return (punc, stop, lemmatizer)


def prepocess(text, business_id):
    (punc, stop, lemmatizer) = init()
    text = re.sub('[0-9]+', '', text)
    origin = pos_tag(word_tokenize(text))
    sentence, filtered = '', []
    for x in origin:
        if x[0] in ',.?!;':
            sentence = sentence.translate(str.maketrans('', '', punc))
            filtered.append(sentence[1:] + '\n')
            sentence = ''
        else:
            low = x[0].lower()
            if (low not in stop and low != 'not'):
                sentence += ' {}'.format(lemmatizer.lemmatize(low))

    fp = open(os.path.join(dst, business_id), 'w+')
    for i in filtered:
        fp.write(i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input training file', default=os.path.join(workdir, 'pratice.json'))
    args = parser.parse_args()

    try:
        check_env(workdir)
        data = pd.read_json(args.input, lines=True)
        if not os.path.exists(dst):
            os.mkdir(dst)
        data.apply(lambda x: prepocess(x['text'], x['business_id']), axis=1)

    except (IOError, OSError) as e:
        logging.error(e)


if __name__ == '__main__':
    main()
