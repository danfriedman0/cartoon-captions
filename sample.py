# Sample from a trained LSTM language model

from __future__ import division
from __future__ import print_function

import sys
import argparse
import pickle
import os

from copy import deepcopy
from timeit import default_timer as timer

import tensorflow as tf

import lstm_ops
import data_reader

parser = argparse.ArgumentParser(description="Sample from an LSTM language model")
parser.add_argument("save_dir", help="Directory with the checkpoints")
args = parser.parse_args()


def sample(save_dir):
    path_to_config = save_dir + "/config"
    if not os.path.isfile(path_to_config):
        raise IOError("Could not find " + path_to_config)
    with open(path_to_config, "rb") as f:
        gen_config = pickle.load(f)

    # # Load vocabulary encoder
    # glove_dir = '/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/glove/glove.6B/glove.6B.50d.txt'
    # #glove_dir = '/data/corpora/word_embeddings/glove/glove.6B.50d.txt'
    # encode, decode, vocab_size, L = data_reader.glove_encoder(glove_dir)
    print("Loading data...")
    data = data_reader.load_data(args.data_fn)
    if args.debug:
        data = sorted(data, key=lambda d: len(d[0]))
        data = data[:10]

    # Split data
    num_train = int(0.8*len(data))
    train_data = data[:num_train]


    L = None
    encode, decode, vocab_size = data_reader.make_encoder(
                                    train_data, config.token_type)

    # Rebuild the model
    with tf.variable_scope("LSTM"):
        gen_model = lstm_ops.seq2seq_model(
                      encoder_seq_length=50,
                      decoder_seq_length=1,
                      num_layers=gen_config.num_layers,
                      embed_size=gen_config.embed_size,
                      batch_size=gen_config.batch_size,
                      hidden_size=gen_config.hidden_size,
                      vocab_size=vocab_size,
                      dropout=gen_config.dropout,
                      max_grad_norm=gen_config.max_grad_norm,
                      use_attention=False,
                      embeddings=L,
                      is_training=False,
                      is_gen_model=True,

                      reuse=False)

    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session,tf.train.latest_checkpoint('./' + args.save_dir))


        def generate(description, temperature):
            return lstm_ops.generate_text_beam_search(
                        session, gen_model, encode, decode,
                        description, 100, temperature=temperature)

        seed = "A doctor in a mouse costume takes notes on the mice in cages. The doctor in the mouse costume is talking to the doctor in the labcoat. There are many mice in cages all around the two doctors."
        temp = 1.0

        print(generate(seed, temp))

        while raw_input("Sample again? (y/n): ") != "n":
            new_seed = raw_input("seed: ")
            if len(encode(seed)) > 200:
                print("Description must be < 200 chars")
                continue
            new_temp = raw_input("temp: ")

            if new_seed != "":
                seed = new_seed
            if new_temp != "":
                temp = float(new_temp)

            print(generate(seed, temp))


if __name__ == "__main__":
    sample(args.save_dir)

