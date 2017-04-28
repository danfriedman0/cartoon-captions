# n-gram-based baseline for the cartoon captioning task
from __future__ import division
from __future__ import print_function

import sys
import json
import math
import argparse
import string
import dill as pickle
import numpy as np

from collections import Counter
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams as get_ngrams


parser = argparse.ArgumentParser(description="Train a baseline n-gram model")
parser.add_argument("mode", help="train or sample")
parser.add_argument("save_dir", help="Directory with the checkpoints")
parser.add_argument("--debug", help="Debug", action="store_true",
                    default=False)
args = parser.parse_args()


UNK = "[UNK]"
START = "[START]"
STOP = "[STOP]"


def get_data(fn='data/dataset.json'):
    # load
    with open(fn, 'r') as f:
        raw_data = json.load(f)
    if args.debug:
        raw_data = raw_data[:5]

    stops = set(stopwords.words('english'))
    stops.update(string.punctuation)
    def tokenize(s, remove_stop_words=False):
        # toks = ' '.split(s.lower())
        toks = word_tokenize(s.lower())
        if remove_stop_words:
            toks = [tok for tok in toks if tok not in stops]
        return toks
    
    # tokenize and remove stopwords
    print('Tokenizing...')
    data = []
    for d,cs in raw_data:
        d_toks = tokenize(d, True)
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
        words = list(set([word for word in words if word in tf_idf]))
        if len(words) < k:
            return words + [UNK for _ in xrange(k - len(words))]
        words = sorted(words, key=lambda w: -tf_idf[w])
        return words[:k]

    return get_keywords

def get_training_data(data, get_keywords, k):
    train = []
    for d,cs in data:
        keywords = get_keywords(d, k) + [START]
        train += [keywords + c + [STOP] for c in cs]
    return train


def get_normalized_log_probs(items):
    count = Counter(items)
    total = sum(count.values())
    probs = {item: math.log(count[item]/total) for item in count}
    return probs


class NgramModel(object):
    def __init__(self, n, train):
        self.n = n
        if n == 1:
            words = [(word,) for sent in train for word in sent]
            counts = self.counts = Counter(words)
            total = sum(counts.values())
            self.probs = {word: math.log(counts[word]/total)
                          for word in counts}
            self.backoff = None
            return

        backoff = self.backoff = NgramModel(n-1, train)

        ngrams = [ngram for sent in train
                        for ngram in get_ngrams(sent, n)]
        counts = self.counts = Counter(ngrams)
        self.probs = {}  
        for ngram in counts:
            context = tuple(ngram[:-1])
            token = ngram[-1]
            if context not in self.probs:
                self.probs[context] = {}
            self.probs[context][token] = (counts[ngram] /
                                          backoff.counts[context])
            if self.probs[context][token] == 0:
                print(context, token)
                print(counts[ngram])
                print(backoff.counts[context])
                break


        # normalize the probabilities
        for context in self.probs:
            tokens, probs = zip(*self.probs[context].iteritems())
            total = sum(probs)
            if total == 0:
                print(context, tokens, probs)
            for token,prob in zip(tokens,probs):
                self.probs[context][token] = prob / total


    def generate_one(self, context):
        if self.n == 1:
            return '_'
        if context in self.probs:
            tokens, probs = zip(*self.probs[context].iteritems())
            token = np.random.choice(tokens, p=probs)
            return token
        else:
            return self.backoff.generate_one(tuple(context[1:]))


    def generate(self, num_words, seed):
        if len(seed) != self.n - 1:
            raise ValueError("Seed must be size {}".format(self.n - 1))

        output = list(seed)
        for i in xrange(num_words):
            context = tuple(output[-self.n+1:])
            next_word = self.generate_one(context)
            output.append(next_word)
            if next_word == STOP:
                break

        return output


def train():
    n = 4

    data = get_data()
    descriptions = [d for d,_ in data]

    print("Getting keyword extractor...")
    get_keywords = get_keyword_extractor(descriptions)
    print("Done. Preparing training data...")
    train = get_training_data(data, get_keywords, n-2)
    print("Done. Building n-gram model...")
    model = NgramModel(n, train)
    print("Done. Saving...")

    # with open(args.save_dir + 'model.pkl', 'w') as f:
    #     pickle.dump(model, f)

    # print("Done.")

    seed1 = "Two men sit at the bar and talk over drinks . The man with the tail is talking to the man without a tail . Both men are wearing suits and look professional ."
    seed2 = "A doctor is talking to a patient . The doctor is reading from a piece of paper in his hands . The paper is on fire ."
    descriptions = [seed1, seed2]
    for description in descriptions:
        keywords = get_keywords(description, n-2)
        print('description: ' + ' '.join(description))
        print('keywords: ' + ' '.join(keywords))
        for _ in xrange(10):
            print(' '.join(model.generate(20, tuple(keywords + [START]))))


def main():
    if args.mode == 'train':
        train()


if __name__ == '__main__':
    main()

