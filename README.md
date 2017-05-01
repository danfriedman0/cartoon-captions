# cartoon-captions
LSTM-based text generation for New Yorker cartoon caption contest.

This code works with Python 2.7 and Tensorflow 1.0. To run:

`python train.py`

To sample from a saved model:

`python sample.py [path/to/save/directory]`

The data for this project is private, but you can run the code on your own data set. The data should be a JSON file consisting of a list of tuples, each one containing a query (e.g. a cartoon description) and a list of responses (captions). Update `data_dir` to match your location. To run the the file using pretrained Glove vectors, download the glove.6B.50d file and update the glove_dir accordingly. See train.py for more arguments.

Example outputs:

Description: "Two men sit at the bar and talk over drinks . The man with the tail is talking to the man without a tail . Both men are wearing suits and look professional ."

Outputs (medium config with attention):
* it might be a tail , but i 'm a monkey .
* `` my ass . '’
* i just do n't have a tail .
