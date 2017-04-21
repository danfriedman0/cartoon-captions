# Trains the LSTM language model defined in lstm.py
# Dan Friedman 03/17
#
# Implementation based on:
#   http://cs224d.stanford.edu/assignment2/index.html
#   http://colah.github.io/posts/2015-08-Understanding-LSTMs/
#   https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py
# and
#   https://github.com/karpathy/char-rnn
from __future__ import division
from __future__ import print_function

import argparse
import pickle
import os

from copy import deepcopy
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow.python.client import timeline

import lstm_ops
import data_reader
from configs import Config

test_description = "A doctor in a mouse costume takes notes on the mice in cages. The doctor in the mouse costume is talking to the doctor in the labcoat. There are many mice in cages all around the two doctors."

parser = argparse.ArgumentParser(description="Train an LSTM language model")
parser.add_argument("--data_fn", help="Path to data file",
                    default="data/dataset.json")
parser.add_argument("--config_size", help="Config size {small, medium, large}",
                    default="small")
parser.add_argument("--temperature", help="Temperature for sampling [0,1.0]",
                    type=float, default=1.0)
parser.add_argument("--sample_every", help="How often to sample (in epochs)",
                    type=int, default=1)
parser.add_argument("--save_every", help="How often to save (in epochs)",
                    type=int, default=10)
parser.add_argument("--early_stopping",help="Stop after n epochs w. flat loss",
                    type=int, default=2)
parser.add_argument("--num_layers", help="Number of RNN layers",
                    type=int, default=None)
parser.add_argument("--batch_size", help="Batch size",
                    type=int, default=None)
parser.add_argument("--hidden_size", help="Size of the RNN hidden state",
                    type=int, default=None)
parser.add_argument("--dropout", help="keep_prob for dropout",
                    type=float, default=None)
parser.add_argument("--token_type", help="Predict words or chars",
                    default="chars")
parser.add_argument("--save_dir", help="Name of directory for saving models",
                                        default="cv/test/")

parser.add_argument("--debug", help="Debug", action="store_true",
                    default=False)

args = parser.parse_args()


# Some sample configurations

configs = {}

configs["test"] = Config(
    max_grad_norm = 5,
    num_layers = 1,
    hidden_size = 20,
    max_epochs = 1,
    max_max_epoch = 1,
    dropout = 1.0,
    batch_size = 5,
    embed_size = 16,
    token_type= "words")

configs["small"] = Config(
    max_grad_norm = 5,
    num_layers = 1,
    hidden_size = 64,
    max_epochs = 8,
    max_max_epoch = 13,
    dropout = 1.0,
    batch_size = 50,
    embed_size = 25,
    token_type= "words")

configs["medium"] = Config(
    max_grad_norm = 5,
    num_layers = 2,
    hidden_size = 128,
    max_epochs = 32,
    max_max_epoch = 39,
    dropout = 0.9,
    batch_size = 50,
    embed_size = 50,
    token_type = "words")

configs["large"] = Config(
    max_grad_norm = 10,
    num_layers = 2,
    hidden_size = 256,
    max_epochs = 32,
    max_max_epoch = 55,
    dropout = 0.7,
    batch_size = 50,
    embed_size = 100,
    token_type = "words")

def train(config):
    # Load the data
    print("Loading data...")
    data = lstm_ops.load_data(
                args.data_fn, config.batch_size, config.token_type,
                min_count=3, split_size=0.8, debug=args.debug)
    (train_producer, valid_producer, num_train, num_valid, encode, decode,vocab_size, d_len, c_len) = data
    print("Done. Building model...")

    # Create a duplicate of the training model for generating text
    gen_config = deepcopy(config)
    gen_config.batch_size = 1
    gen_config.dropout = 1.0

    # Save gen_model config so we can sample later
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    path_to_model = os.path.join(args.save_dir, "config")
    with open(path_to_model, "wb") as f:
        pickle.dump(gen_config, f)

    path_to_index = os.path.join(args.save_dir, "index")
    with open(path_to_index, "w") as f:
        f.write("loss per epoch:\n")
        f.write("---------------\n")


    # Create training model
    with tf.variable_scope("LSTM") as scope:
        model = lstm_ops.seq2seq_model(
            d_len, c_len, config.num_layers,
            config.num_layers, config.embed_size, config.batch_size,
            config.hidden_size, vocab_size, config.dropout,
            config.max_grad_norm,
            is_training=True, is_gen_model=False, reuse=False)
        gen_model = lstm_ops.seq2seq_model(
            len(encode(test_description)), 1, gen_config.num_layers,
            gen_config.num_layers, gen_config.embed_size,
            gen_config.batch_size, gen_config.hidden_size, vocab_size, 
            gen_config.dropout, gen_config.max_grad_norm,
            is_training=False, is_gen_model=True, reuse=True)

    print("Done.")

    with tf.Session() as session:
        best_val_pp = float('inf')
        best_val_epoch = 0

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # Sample some text
        print(lstm_ops.generate_text(session, gen_model, encode, decode,
            description=test_description, temperature=args.temperature))

        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            start = timer()

            # Train on the epoch and validate
            train_pp = lstm_ops.run_epoch(
                session, model, train_producer, num_train)
            print("Validating:")
            valid_pp = lstm_ops.run_epoch(
                session, model, valid_producer, num_valid)
            print("Validation loss: {}".format(valid_pp))

            # Save the model if validation loss has dropped
            if valid_pp < best_val_pp:
                with open(path_to_index, "a") as f:
                    f.write("{}: {}*\n".format(epoch, valid_pp))
                best_val_pp = valid_pp
                best_val_epoch = epoch
                path_to_ckpt = os.path.join(args.save_dir, "epoch.ckpt")
                print("Saving model to " + path_to_ckpt)
                saver.save(session, "./" + path_to_ckpt)

            # Otherwise just record validation loss in save_dir/index
            else:
                with open(path_to_index, "a") as f:
                    f.write("{}: {}\n".format(epoch, valid_pp))

            # Stop early if validation loss is getting worse
            if epoch - best_val_epoch > args.early_stopping:
                print("Stopping early")
                break

            print('Total time: {}\n'.format(timer() - start))
            if epoch % args.sample_every == 0:
                print(lstm_ops.generate_text(
                    session, gen_model, encode, decode,
                    description=test_description,temperature=args.temperature))

        print(lstm_ops.generate_text(session, gen_model, encode, decode,
                description=test_description,temperature=args.temperature))


def main():
    config = configs[args.config_size]

    if args.num_layers is not None: config.num_layers = args.num_layers
    if args.batch_size is not None: config.batch_size = args.batch_size
    if args.hidden_size is not None: config.hidden_size = args.hidden_size
    if args.dropout is not None: config.dropout = args.dropout
    if args.token_type == "words":
        config.token_type = "words"

    train(config)


if __name__ == "__main__":
    main()
