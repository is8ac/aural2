import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import argparse
import codecs
import json
import logging
import os
import shutil
import sys

import tensorflow as tf

from tensorflow.python.framework import graph_util
import numpy as np
from six import iteritems
import time


# Disable Tensorflow logging messages.
logging.getLogger('tensorflow').setLevel(logging.WARNING)

class CharRNN(object):
  """Character RNN model."""

  def __init__(self, params, is_training=True, use_batch=True, seq_len=7):
    batch_size = params['batch_size']
    num_unrollings = seq_len
    if not use_batch:
      batch_size = 1
    hidden_size = params['hidden_size']
    max_grad_norm = params['max_grad_norm']
    num_layers = params['num_layers']
    embedding_size = params['embedding_size']
    dropout = params['dropout']
    input_dropout = params['input_dropout']
    input_size = params['input_size']
    output_size = params['output_size']
    learning_rate = params['learning_rate']
    # Placeholder to feed in input and targets/labels data.
    self.input_data = tf.placeholder(tf.float32,
                                     [batch_size, num_unrollings, input_size],
                                     name='inputs')
    self.targets = tf.placeholder(tf.int32,
                                  [batch_size, num_unrollings,],
                                  name='targets')

    cell_fn = tf.contrib.rnn.BasicLSTMCell

    cell = cell_fn(hidden_size, reuse=tf.get_variable_scope().reuse, forget_bias=0.0, state_is_tuple=True)

    cells = [cell]
    # params['input_size'] = self.hidden_size
    # more explicit way to create cells for MultiRNNCell than
    # [higher_layer_cell] * (self.num_layers - 1)
    for i in range(num_layers-1):
      higher_layer_cell = cell_fn(hidden_size, reuse=tf.get_variable_scope().reuse, forget_bias=0.0, state_is_tuple=True)
      cells.append(higher_layer_cell)

    if is_training and dropout > 0:
      cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0-self.dropout) for cell in cells]

    multi_cell = tf.contrib.rnn.MultiRNNCell(cells)

    with tf.name_scope('initial_state'):
      # zero_state is used to compute the intial state for cell.
      self.zero_state = multi_cell.zero_state(batch_size, tf.float32)
      # Placeholder to feed in initial state.
      # self.initial_state = tf.placeholder(
      #   tf.float32,
      #   [self.batch_size, multi_cell.state_size],
      #   'initial_state')

      self.initial_state = create_tuple_placeholders_with_default(
        multi_cell.zero_state(batch_size, tf.float32),
        extra_dims=(None,),
        shape=multi_cell.state_size)

    with tf.name_scope('slice_inputs'):
      # Slice inputs into a list of shape [batch_size, 1] data colums.
      sliced_inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(axis=1, num_or_size_splits=num_unrollings, value=self.input_data)]

    # Copy cell to do unrolling and collect outputs.
    outputs, final_state = tf.contrib.rnn.static_rnn(
      multi_cell, sliced_inputs,
      initial_state=self.initial_state)

    self.final_state = final_state

    # outputs is a python list of seq_len tensors of shape [batch_size, state]
    with tf.name_scope('flatten_outputs'):
      # Flatten the outputs into one dimension.
      concated_outputs = tf.concat(axis=1, values=outputs)
      flat_outputs = tf.reshape(concated_outputs, [-1, hidden_size])

    with tf.name_scope('flatten_targets'):
      # Flatten the targets too.
      flat_targets = tf.reshape(self.targets, [-1])

    # Create softmax parameters, weights and bias.
    with tf.variable_scope('softmax') as sm_vs:
      softmax_w = tf.get_variable("softmax_w", [hidden_size, output_size])
      softmax_b = tf.get_variable("softmax_b", [output_size])
      self.logits = tf.matmul(flat_outputs, softmax_w) + softmax_b
      self.probs = tf.nn.softmax(self.logits, name='output')

    with tf.name_scope('loss'):
      # Compute mean cross entropy loss for each output.
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=flat_targets)
      self.mean_loss = tf.reduce_mean(loss)

    with tf.name_scope('loss_monitor'):
      # Count the number of elements and the sum of mean_loss
      # from each batch to compute the average loss.
      count = tf.Variable(1.0, name='count')
      sum_mean_loss = tf.Variable(1.0, name='sum_mean_loss')

      self.reset_loss_monitor = tf.group(sum_mean_loss.assign(0.0),
                                         count.assign(0.0),
                                         name='reset_loss_monitor')
      self.update_loss_monitor = tf.group(sum_mean_loss.assign(sum_mean_loss +
                                                               self.mean_loss),
                                          count.assign(count + 1),
                                          name='update_loss_monitor')
      with tf.control_dependencies([self.update_loss_monitor]):
        self.average_loss = sum_mean_loss / count
        self.ppl = tf.exp(self.average_loss)

      # Monitor the loss.
      loss_summary_name = "average loss"
      ppl_summary_name = "perplexity"

      average_loss_summary = tf.summary.scalar(loss_summary_name, self.average_loss)
      ppl_summary = tf.summary.scalar(ppl_summary_name, self.ppl)

    # Monitor the loss.
    self.summaries = tf.summary.merge([average_loss_summary, ppl_summary],
                                      name='loss_monitor')

    self.global_step = tf.get_variable('global_step', [],
                                       initializer=tf.constant_initializer(0.0))

    self.learning_rate = tf.constant(learning_rate)
    if is_training:
      # learning_rate = tf.train.exponential_decay(1.0, self.global_step,
      #                                            5000, 0.1, staircase=True)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars),
                                        max_grad_norm)
      # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay_rate)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)

      self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                global_step=self.global_step)


def create_tuple_placeholders_with_default(inputs, extra_dims, shape):
  if isinstance(shape, int):
    result = tf.placeholder_with_default(
      inputs, list(extra_dims) + [shape])
  else:
    subplaceholders = [create_tuple_placeholders_with_default(
      subinputs, extra_dims, subshape)
                       for subinputs, subshape in zip(inputs, shape)]
    t = type(shape)
    if t == tuple:
      result = t(subplaceholders)
    else:
      result = t(*subplaceholders)
  return result


def create_tuple_placeholders(dtype, extra_dims, shape):
  if isinstance(shape, int):
    result = tf.placeholder(dtype, list(extra_dims) + [shape])
  else:
    subplaceholders = [create_tuple_placeholders(dtype, extra_dims, subshape)
                       for subshape in shape]
    t = type(shape)

    # Handles both tuple and LSTMStateTuple.
    if t == tuple:
      result = t(subplaceholders)
    else:
      result = t(*subplaceholders)
  return result


def main():
    params = {
            "batch_size": 7,
            "dropout": 0.0,
            "embedding_size": 0,
            "hidden_size": 64,
            "input_dropout": 0.0,
            "input_size": 13,
            "learning_rate": 0.001,
            "max_grad_norm": 5.0,
            "num_layers": 2,
            "num_unrollings": 100,
            "full_seq_len": 312,
            "output_size": 40,
            "vocab_name": "intent",
            }

    # Create graphs
    print('Creating graph')
    # make graphs for training
    with tf.name_scope('training'):
        train_model = CharRNN(params, is_training=True, use_batch=True, seq_len=params['num_unrollings'])
    tf.get_variable_scope().reuse_variables()
    # make graph for doing inference one mfcc at a time.
    with tf.name_scope('step_inference'):
        step_inference_model = CharRNN(params, is_training=False, use_batch=False, seq_len=1)
        initialStateNames = []
        finalStateNames = []
        for i in step_inference_model.initial_state:
            initialStateNames.append(i.c.op.name)
            initialStateNames.append(i.h.op.name)
        for i in step_inference_model.final_state:
            finalStateNames.append(i.c.op.name)
            finalStateNames.append(i.h.op.name)
        step_initialStateConst = tf.constant(initialStateNames, name="initial_state_names")
        step_finalStateNamesConst = tf.constant(finalStateNames, name="final_state_names")

    zeros = tf.zeros([1, params['hidden_size']], dtype=tf.float32, name="zeros")
    init = tf.global_variables_initializer()
    print(train_model.input_data.op.name)
    print(step_inference_model.input_data.op.name)
    print(train_model.targets.op.name)
    print(train_model.train_op.name)
    print(train_model.ppl.op.name)
    print(step_inference_model.probs.op.name)
    print(init.name)
    tf.train.write_graph(tf.get_default_graph().as_graph_def(), 'models', 'lstm_train.pb', as_text=False)



if __name__ == '__main__':
    main()
