from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import graph_util

import argparse
import os
from six.moves import cPickle

from model import Model

from six import text_type


def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                        help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, training=False)
    initialStateNames = []
    finalStateNames = []
    for i in model.initial_state:
        initialStateNames.append(i.c.name)
        initialStateNames.append(i.h.name)
    for i in model.final_state:
        finalStateNames.append(i.c.name)
        finalStateNames.append(i.h.name)

    initialStateConst = tf.constant(initialStateNames, name="initial_state_names")
    finalStateNamesConst = tf.constant(finalStateNames, name="final_state_names")
    zeros = tf.zeros([1,saved_args.rnn_size], dtype=tf.float32, name="zeros")
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            frozenGraph = graph_util.convert_variables_to_constants( # freeze the graph
                sess,
                tf.get_default_graph().as_graph_def(), # use the default graph
                ["probs", "output", "cell_zero_state", "initial_state_names", "final_state_names", "zeros"], # preserve all these OPs
            )
            tf.train.write_graph(frozenGraph, 'models', 'char_rnn.pb', as_text=False)

if __name__ == '__main__':
    main()
