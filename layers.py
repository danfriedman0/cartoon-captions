# Long Short Term Memory language model
# Dan Friedman, 03/2017
#
# Implementation based on:
#   http://cs224d.stanford.edu/assignment2/index.html
#   http://colah.github.io/posts/2015-08-Understanding-LSTMs/
#   https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py
# and
#   https://github.com/karpathy/char-rnn
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py
import tensorflow as tf

from cells import make_lstm_cell, make_rnn_cell, make_stacked_rnn_cell


def make_lstm(input_size, state_size, batch_size, num_steps, num_layers, dropout, cell_type="lstm"):
  """
  Returns:
    lstm: a function (feeds forward through an lstm)
    init_state: the initial state of the lstm
  """

  # Create the lstm cell
  if cell_type == "tf_lstm":
    def lstm_cell(state_size):
      return tf.contrib.rnn.DropoutWrapper(
        tf.contrib.rnn.LSTMCell(state_size), output_keep_prob=dropout)
    cell = tf.contrib.rnn.MultiRNNCell(
      [lstm_cell(state_size) for _ in range(num_layers)])
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
  else:
    make_cell = make_lstm_cell if cell_type == "lstm" else make_rnn_cell
    init_cells = [make_cell(input_size, state_size, batch_size, dropout)]
    for _ in range(num_layers - 1):
      init_cells.append(make_cell(state_size, state_size, batch_size, dropout))
    cell, init_state = make_stacked_rnn_cell(init_cells)

  # Define the forward pass
  def lstm(state, inputs, reuse):
    # Forward pass through the LSTM
    outputs = []
    with tf.variable_scope("forward_pass", reuse=reuse) as scope:
      for time_step in range(num_steps):
        if time_step > 0 and not reuse:
          scope.reuse_variables()
        output, state = cell(inputs[:, time_step, :], state)
        outputs.append(output)
    return outputs, state

  return lstm, init_state


def make_char_cnn(char_embed_size, in_channels, filter_widths, filters_by_width, max_filters, reuse=False):
  """Returns a char_cnn function"""

  embed_size = 0

  # Create filters and biases
  params = []
  for i, width in enumerate(filter_widths):
    with tf.variable_scope("conv{}".format(i), reuse=reuse) as scope:
      out_channels = filters_by_width * width
      if max_filters is not None:
        out_channels = min(out_channels, max_filters)
      f = tf.get_variable("weights",
        shape=(char_embed_size, width, in_channels, out_channels),
        initializer=tf.random_uniform_initializer(-0.05, 0.05),
        dtype=tf.float32)
      b = tf.get_variable("biases", shape=(out_channels), dtype=tf.float32,
        initializer=tf.constant_initializer(0.0))
      params.append((f, b))
      embed_size += out_channels

  def char_cnn(inputs):
    outputs = []
    for f, b in params:
      height = inputs.shape[1]
      width = inputs.shape[2]
      conv = tf.nn.conv2d(inputs, f, strides=[1, height, 1, 1], padding="SAME")
      activation = tf.tanh(conv + b)
      pool = tf.nn.max_pool(activation, ksize=[1, 1, width, 1],
        strides=[1, 1, width, 1], padding="SAME")
      output = tf.squeeze(pool, squeeze_dims=[1, 2])
      outputs.append(output)
    output = tf.concat(outputs, 1)
    return output

  return char_cnn, embed_size


def highway_layer(inputs, reuse=False):
  # Forward pass through the highway layer described in
  # https://arxiv.org/pdf/1508.06615.pdf
  with tf.variable_scope("highway", reuse=reuse):
    rand_init = tf.random_uniform_initializer(-0.08, 0.08)
    embed_size = inputs.shape[-1]
    
    # Get variables
    Wt = tf.get_variable("Wt", shape=(embed_size, embed_size),
      dtype=tf.float32, initializer=rand_init)
    # Initialize bt to be negative to encourage carrying
    bt = tf.get_variable("bt", shape=(embed_size,),
      dtype=tf.float32, initializer=tf.constant_initializer(-2.0))
    Wh = tf.get_variable("Wh", shape=(embed_size, embed_size),
      dtype=tf.float32, initializer=rand_init)
    bh = tf.get_variable("bh", shape=(embed_size,),
      dtype=tf.float32, initializer=tf.constant_initializer(0.0))

    t = tf.sigmoid(tf.matmul(inputs, Wt) + bt)
    g = tf.tanh(tf.matmul(inputs, Wh) + bh)
    z = t * g + (1 - t) * inputs

    return z

def embed_inputs(inputs, vocab_size, embed_size, reuse=False):
  rand_init = tf.random_uniform_initializer(-0.08, 0.08)
  with tf.variable_scope("embed", reuse=reuse):
    embeddings = tf.get_variable("embeddings",
      shape=(vocab_size, embed_size), dtype=tf.float32, initializer=rand_init)
  embedded_inputs = tf.nn.embedding_lookup(embeddings, inputs)
  return embedded_inputs

def project_output(outputs, state_size, vocab_size, reuse=False):
  rand_init = tf.random_uniform_initializer(-0.08, 0.08)
  # Project outputs over the vocabulary
  with tf.variable_scope("projection", reuse=reuse):
    output = tf.reshape(tf.concat(outputs, 1), [-1, state_size])
    softmax_W = tf.get_variable("softmax_W",
      [state_size, vocab_size], dtype=tf.float32, initializer=rand_init)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32,
      initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(output, softmax_W) + softmax_b
    predictions = tf.nn.softmax(tf.cast(logits, 'float64'))
  return logits, predictions


def calculate_sequence_loss(logits, labels, batch_size, num_steps, vocab_size):
  loss = tf.contrib.seq2seq.sequence_loss(
    tf.reshape(logits, [batch_size, num_steps, vocab_size]),
    labels,
    tf.ones([batch_size, num_steps], dtype=tf.float32))
  return loss


def get_train_op(loss, learning_rate, max_grad_norm):
  # Clip gradients and optimize
  optimizer = tf.train.AdamOptimizer(learning_rate)
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(
    tf.gradients(loss, tvars), max_grad_norm)
  train_op = optimizer.apply_gradients(zip(grads, tvars))
  return train_op


