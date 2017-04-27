# n-gram-based baseline for the cartoon captioning task
from __future__ import division
from __future__ import print_function

import sys
import json
import math
import argparse

from collections import Counter
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.model.ngram import NgramModel


parser = argparse.ArgumentParser(description="Train a baseline n-gram model")
parser.add_argument("mode", help="train or sample")
parser.add_argument("save_dir", help="Directory with the checkpoints")
args = parser.parse_args()


UNK = "[UNK]"
START = "[START]"
STOP = "[STOP]"


def get_data(fn='data/dataset.json'):
    # load
    with open(fn, 'r') as f:
        raw_data = json.load(f)

    stops = set(stopwords.words('english'))
    def tokenize(s):
        toks = ' '.split(s.lower())
        toks = [tok for tok in toks if tok not in stops]
    
    # tokenize and remove stopwords
    print('Tokenizing...')
    data = []
    for d,cs in raw_data:
        d_toks = tokenize(d)
        cs_toks = [tokenize(c) for c in cs]
        data.append((d_toks, cs_toks))
    print('Done.')
    return data

def get_keyword_extractor(descriptions):
    # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    
    # term frequency
    tf = {}
    for i,description in enumerate(descriptions):
        tf[i] = Counter(description)

    # inverse document frequency
    idf = {}
    words = [word for description in descriptions for word in description]
    N = len(descriptions)
    for word in words:
        idf[word] = math.log(N / len([d for d in descriptions if word in d]))

    # term frequency-inverse document frequency
    tf_idf = {}
    for i,description in enumerate(descriptions):
        for word in description:
            tf_idf[word] = tf[i][word] * idf[word]

    def get_keywords(words, k):
        words = [word for word in words if word in tf_idf]
        if len(words) < k:
            return words + [UNK for _ in xrange(k - len(words))]
        words = sorted(words, key=lambda w: -tf_idf[w])
        return words[:k]

    return get_keywords

def get_training_data(data, get_keywords, k):
    train = []
    for d,cs in data:
        keywords = get_keywords(d, k) + [START]
        train += [d + c + [STOP] for c in cs]
    return train

def train():
    data = get_data()
    descriptions = [d for d,_ in data]

    print("Getting keyword extractor...")
    get_keywords = get_keyword_extractor(descriptions)
    print("Done. Preparing training data...")
    train = get_training_data(data, get_keywords, n-2)
    print("Done. Building n-gram model...")
    model = NgramModel(n, train)
    print("Done.")

    with open(args.save_dir, 'w') as f:
        json.dump(model, f)

    for description,_ in data:
        keywords = get_keywords(description)
        print(description)
        print('keywords: ' + ' '.join(keywords))
        print(model.generate(20, context=keywords))


def main():
    if args.mode == 'train':
        train()


if __name__ == '__main__':
    main()

