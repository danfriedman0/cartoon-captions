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
        config = pickle.load(f)

    # Load vocabulary encoder
    data = lstm_ops.load_data('data/dataset.json', 1, 'words',
                min_count=3, split_size=0.8, debug=False)
    encode, decode, vocab_size = data[4:7]


    # Rebuild the model
    with tf.variable_scope("LSTM"):
        gen_model = lstm_ops.seq2seq_model(50, 1,
                    config.num_layers, config.num_layers, config.embed_size,
                    1, config.hidden_size, vocab_size, config.dropout,
                    config.max_grad_norm, 50,
                    is_training=False, is_gen_model=True, reuse=False)

    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session,tf.train.latest_checkpoint('./' + args.save_dir))

        def generate(description, temperature):
            # print(seed)
            # print(len(seed))
            return lstm_ops.generate_text(
                        session, gen_model, encode, decode, description, 50,
                        temperature=temperature)

        seed = "A doctor in a mouse costume takes notes on the mice in cages. The doctor in the mouse costume is talking to the doctor in the labcoat. There are many mice in cages all around the two doctors."
        temp = 1.0

        print(generate(seed, temp))

        while raw_input("Sample again? (y/n): ") != "n":
            new_seed = raw_input("seed: ")
            if len(encode(seed)) > 50:
                print("Description must be < 50 words")
                continue
            new_temp = raw_input("temp: ")

            if new_seed != "":
                seed = new_seed
            if new_temp != "":
                temp = float(new_temp)

            print(generate(seed, temp))


if __name__ == "__main__":
    sample(args.save_dir)

