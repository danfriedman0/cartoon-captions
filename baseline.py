# n-gram-based baseline for the cartoon captioning task

from __future__ import division
from __future__ import print_function

import sys
import argparse
import dill as pickle
import os
import json

import data_reader

from nltk import word_tokenize


def load_data(fn='data/dataset.json'):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data


with open('data/dataset.json', 'r') as f:
    data = json.load(f)


_data = []
for d,cs in data:
    d_toks = word_tokenize(d)
    _d = ' '.join(d_toks)
    _cs = []
    for c in cs:
        c_toks = word_tokenize(c)
        _c = ' '.join(c_toks)
        _cs.append(_c)
    _data.append(_d, _cs)

with open('data/_dataset.json', 'w') as f:
    json.dump(_data)
