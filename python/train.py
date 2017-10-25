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
    params = {
            "batch_size": 5,
            "dropout": 0.0,
            "embedding_size": 0,
            "hidden_size": 256,
            "input_dropout": 0.0,
            "input_size": 13,
            "learning_rate": 0.0005,
            "max_grad_norm": 5.0,
            "num_layers": 2,
            "num_unrollings": 100,
            "full_seq_len": 312,
            "output_size": 20,
            "num_batches": 20000,
            "vocab_name": "intent",
            }

    # Create graphs
    print('Creating graph')
    graph = tf.Graph()
    with graph.as_default():
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
        # make graph for doing inference on a full length sequence
        with tf.name_scope('seq_inference'):
            seq_inference_model = CharRNN(params, is_training=False, use_batch=False, seq_len=params['full_seq_len'])
            initialStateNames = []
            finalStateNames = []
            for i in seq_inference_model.initial_state:
                initialStateNames.append(i.c.op.name)
                initialStateNames.append(i.h.op.name)
            for i in seq_inference_model.final_state:
                finalStateNames.append(i.c.op.name)
                finalStateNames.append(i.h.op.name)
            seq_initialStateConst = tf.constant(initialStateNames, name="initial_state_names")
            seq_finalStateNamesConst = tf.constant(finalStateNames, name="final_state_names")



    print('Start training\n')

    with tf.Session(graph=graph) as session:
        with tf.name_scope('training_data'):
            graph_def = tf.GraphDef()
            graph_path = 'trainingdata/' + params["vocab_name"] + '.pb'
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
        for i in range(params['num_batches']):
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

        zeros = tf.zeros([1, params['hidden_size']], dtype=tf.float32, name="zeros")
        frozenGraph = graph_util.convert_variables_to_constants( # freeze the graph
            session,
            tf.get_default_graph().as_graph_def(), # use the default graph
            [
            step_inference_model.input_data.op.name,
            step_inference_model.probs.op.name,
            step_initialStateConst.op.name,
            step_finalStateNamesConst.op.name,
            seq_inference_model.input_data.op.name,
            seq_inference_model.probs.op.name,
            seq_initialStateConst.op.name,
            seq_finalStateNamesConst.op.name,
            zeros.op.name
            ], # preserve all these OPs
        )
        tf.train.write_graph(frozenGraph, 'models', params["vocab_name"] + '_rnn.pb', as_text=False)



if __name__ == '__main__':
    main()
