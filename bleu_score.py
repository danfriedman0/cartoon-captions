# Sample from a trained LSTM language model

from __future__ import division
from __future__ import print_function

import sys
import argparse
import dill as pickle
import os

from copy import deepcopy
from timeit import default_timer as timer
from nltk.translate import bleu_score

import tensorflow as tf

import lstm_ops
import data_reader

parser = argparse.ArgumentParser(description="Sample from an LSTM language model")
parser.add_argument("save_dir", help="Directory with the checkpoints")
parser.add_argument("--debug", help="Debug", action="store_true",
                    default=False)
args = parser.parse_args()


def sample(save_dir):
    path_to_config = save_dir + "/config"
    if not os.path.isfile(path_to_config):
        raise IOError("Could not find " + path_to_config)
    with open(path_to_config, "rb") as f:
        config = pickle.load(f)

    # # Load vocabulary encoder
    # glove_dir = '/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/glove/glove.6B/glove.6B.50d.txt'
    # #glove_dir = '/data/corpora/word_embeddings/glove/glove.6B.50d.txt'
    if config.use_glove:
        _, _, _, L = data_reader.glove_encoder(config.glove_dir)
    else:
        L = None

    # Load the data
    data = data_reader.load_data(config.data_fn)
    descriptions = [d for d,_ in data]
    raw_captions = [cs for _,cs in data]
    tokenize, join = data_reader.get_tokenizer(config.token_type)
    captions = [[tokenize(c) for c in cs] for cs in raw_captions]

    # Rebuild the model
    with tf.variable_scope("LSTM"):
        model = lstm_ops.seq2seq_model(
                      encoder_seq_length=config.d_len,
                      decoder_seq_length=config.c_len,
                      num_layers=config.num_layers,
                      embed_size=config.embed_size,
                      batch_size=config.batch_size,
                      hidden_size=config.hidden_size,
                      vocab_size=config.vocab_size,
                      dropout=config.dropout,
                      max_grad_norm=config.max_grad_norm,
                      use_attention=config.use_attention,
                      embeddings=L,
                      is_training=False,
                      is_gen_model=True,
                      token_type=config.token_type,
                      reuse=False)

    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session,tf.train.latest_checkpoint('./' + save_dir))

        def generate(description, temperature=1.0):
            return lstm_ops.generate_text_beam_search(
                        session=session,
                        model=model,
                        encode=model.encode,
                        decode=model.decode,
                        description=description,
                        d_len=model.d_len,
                        beam=5,
                        stop_length=model.c_len,
                        temperature=temperature)


        hypotheses = []
        references = []

        for i,d in enumerate(descriptions):
            sys.stdout.write('{}/{}\r'.format(i, len(descriptions)))
            sys.stdout.flush()
            hypothesis = tokenize(generate(d))
            stop_idx = hypothesis.index('[STOP]')
            if stop_idx > -1:
                hypothesis = hypothesis[len(d)+3:stop_idx]
            print(hypothesis)
            hypotheses.append(hypothesis)
            references.append(captions[i])

    score = bleu_score.corpus_bleu(references, hypotheses)
    print('Score: {}'.format(score))


if __name__ == "__main__":
    sample(args.save_dir)

