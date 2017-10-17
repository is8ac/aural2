import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import argparse
import codecs
import json
import logging
import os
import shutil
import sys

from tensorflow.python.framework import graph_util
import numpy as np
from model import *
from six import iteritems
import time


def main():
    params = json.loads('''
            {
            "batch_size": 10,
            "dropout": 0.0,
            "embedding_size": 0,
            "hidden_size": 512,
            "input_dropout": 0.0,
            "input_size": 13,
            "learning_rate": 0.002,
            "max_grad_norm": 5.0,
            "model": "lstm",
            "num_layers": 2,
            "num_unrollings": 50,
            "output_size": 40
            }
            ''')
    #logging.info('Parameters are:\n%s\n', json.dumps(params, sort_keys=True, indent=4))

    # Create graphs
    print('Creating graph')
    graph = tf.Graph()
    with graph.as_default():
        # make graphs for training
        with tf.name_scope('training'):
            train_model = CharRNN(is_training=True, use_batch=True, **params)
        tf.get_variable_scope().reuse_variables()
        with tf.name_scope('evaluation'):
            test_model = CharRNN(is_training=False, use_batch=False, **params)

    print('Start training\n')

    with tf.Session(graph=graph) as session:
        with tf.name_scope('training_data'):
            graph_def = tf.GraphDef()
            graph_path = 'trainingdata.pb'
            with open(graph_path, "rb") as f:
                proto_b = f.read()
                graph_def.ParseFromString(proto_b)
            (inputsOP, outputsOP, hashesOP) = tf.import_graph_def(graph_def, name="training_data", return_elements=["inputs/Identity:0", "outputs/Identity:0", "clip_hashes/Const:0"])
            hashes = session.run(hashesOP)
            print("from", len(hashes), "audio files")

        graph_info = session.graph
        print("got graph_info")

        train_writer = tf.summary.FileWriter('train/', graph_info)

        print("done creating tf writers")
        tf.global_variables_initializer().run()
        for i in range(2000):
            batch_start_time = time.time()
            inputs, targets = session.run([inputsOP, outputsOP])
            # training step

            # Prepare initial state and reset the average loss
            # computation.
            state = session.run(train_model.zero_state)
            train_model.reset_loss_monitor.run()

            ops = [train_model.average_loss, train_model.final_state, train_model.train_op,
             train_model.summaries, train_model.global_step, train_model.learning_rate, train_model.ppl]
            feed_dict = {train_model.input_data: inputs, train_model.targets: targets,
                   train_model.initial_state: state}

            average_loss, state, _, train_summary_str, global_step, lr, ppl = session.run(ops, feed_dict)

            # record the summary
            train_writer.add_summary(train_summary_str, global_step)
            train_writer.flush()
            print("batch:", i, "ppl:", ppl, "time:", time.time() - batch_start_time)

        initialStateNames = []
        finalStateNames = []
        for i in test_model.initial_state:
            initialStateNames.append(i.c.name)
            initialStateNames.append(i.h.name)
        for i in test_model.final_state:
            finalStateNames.append(i.c.name)
            finalStateNames.append(i.h.name)

        initialStateConst = tf.constant(initialStateNames, name="initial_state_names")
        finalStateNamesConst = tf.constant(finalStateNames, name="final_state_names")
        zeros = tf.zeros([1, params['hidden_size']], dtype=tf.float32, name="zeros")
        frozenGraph = graph_util.convert_variables_to_constants( # freeze the graph
            session,
            tf.get_default_graph().as_graph_def(), # use the default graph
            ["evaluation/input", "evaluation/softmax/output", "initial_state_names", "final_state_names", "zeros"], # preserve all these OPs
        )
        tf.train.write_graph(frozenGraph, 'models', 'cmd_rnn.pb', as_text=False)

        #test_ppl, _, _ = test_model.run_epoch(session, test_size, test_batches, is_training=False, verbose=args.verbose, freq=args.progress_freq)



if __name__ == '__main__':
    main()
