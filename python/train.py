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
    start_time = time.time()
    parser = argparse.ArgumentParser()

    # Parameters for saving models.
    parser.add_argument('--output_dir', type=str, default='output',
                        help=('directory to store final and'
                              ' intermediate results and models.'))
    parser.add_argument('--n_save', type=int, default=1,
                        help='how many times to save the model during each epoch.')
    parser.add_argument('--max_to_keep', type=int, default=5,
                        help='how many recent models to keep.')


    # Parameters to control the training.
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='minibatch size')
    # test_frac is computed as (1 - train_frac - valid_frac).
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate, default to 0 (no dropout).')

    parser.add_argument('--input_dropout', type=float, default=0.0,
                        help=('dropout rate on input layer, default to 0 (no dropout),'
                              'and no dropout if using one-hot representation.'))


    # Parameters for logging.
    parser.add_argument('--log_to_file', dest='log_to_file', action='store_true',
                        help=('whether the experiment log is stored in a file under'
                              '  output_dir or printed at stdout.'))
    parser.set_defaults(log_to_file=False)

    parser.add_argument('--progress_freq', type=int,
                        default=100,
                        help=('frequency for progress report in training'
                              ' and evalution.'))

    parser.add_argument('--verbose', type=int,
                        default=0,
                        help=('whether to show progress report in training'
                              ' and evalution.'))


    # Parameters for unittesting the implementation.
    parser.add_argument('--test', dest='test', action='store_true',
                        help=('use the first 1000 character to as data'
                              ' to test the implementation'))
    parser.set_defaults(test=False)

    args = parser.parse_args()

    print("creating data loading sess at", time.time() - start_time)
    with tf.Session() as sess:
        graph_def = tf.GraphDef()
        graph_path = 'trainingdata.pb'
        with open(graph_path, "rb") as f:
            proto_b = f.read()
            graph_def.ParseFromString(proto_b)
        ops = tf.import_graph_def(graph_def, name="training_data", return_elements=["inputs/Identity:0", "outputs/Identity:0", "clip_hashes/Const:0"])
        #print(ops)
        (inputs, outputs, hashes) = sess.run(ops)
    print(inputs.shape, outputs.shape)
    print("got data at", time.time() - start_time)


    # Specifying location to store model, best model and tensorboard log.
    args.save_model = os.path.join(args.output_dir, 'save_model/model')
    args.save_best_model = os.path.join(args.output_dir, 'best_model/model')
    args.tb_log_dir = os.path.join(args.output_dir, 'tensorboard_log/')
    args.vocab_file = ''

    # Create necessary directories.
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    for paths in [args.save_model, args.save_best_model,
                  args.tb_log_dir]:
        os.makedirs(os.path.dirname(paths))

    # Specify logging config.
    if args.log_to_file:
        args.log_file = os.path.join(args.output_dir, 'experiment_log.txt')
    else:
        args.log_file = 'stdout'

    # Set logging file.
    if args.log_file == 'stdout':
        logging.basicConfig(stream=sys.stdout,
                            format='%(asctime)s %(levelname)s:%(message)s',
                            level=logging.INFO,
                            datefmt='%I:%M:%S')
    else:
        logging.basicConfig(filename=args.log_file,
                            format='%(asctime)s %(levelname)s:%(message)s',
                            level=logging.INFO,
                            datefmt='%I:%M:%S')


    params = json.loads('''
            {
            "batch_size": 6,
            "dropout": 0.0,
            "embedding_size": 0,
            "hidden_size": 128,
            "input_dropout": 0.0,
            "input_size": 13,
            "learning_rate": 0.002,
            "max_grad_norm": 5.0,
            "model": "lstm",
            "num_layers": 2,
            "num_unrollings": 156,
            "output_size": 40
            }
            ''')
    #logging.info('Parameters are:\n%s\n', json.dumps(params, sort_keys=True, indent=4))

    print("creating graphs at", time.time() - start_time)

    # Create graphs
    logging.info('Creating graph')
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('training'):
            train_model = CharRNN(is_training=True, use_batch=True, **params)
        tf.get_variable_scope().reuse_variables()
        with tf.name_scope('validation'):
            valid_model = CharRNN(is_training=False, use_batch=True, **params)
        with tf.name_scope('evaluation'):
            test_model = CharRNN(is_training=False, use_batch=False, **params)
            saver = tf.train.Saver(name='checkpoint_saver', max_to_keep=args.max_to_keep)
            best_model_saver = tf.train.Saver(name='best_model_saver')

    print("input name", test_model.input_data.name)
    logging.info('Model size (number of parameters): %s\n', train_model.model_size)
    logging.info('Start training\n')

    print("creating main sess at", time.time() - start_time)

    with tf.Session(graph=graph) as session:
        graph_info = session.graph
        print("got graph_info")

        train_writer = tf.summary.FileWriter(args.tb_log_dir + 'train/', graph_info)
        valid_writer = tf.summary.FileWriter(args.tb_log_dir + 'valid/', graph_info)

        logging.info("initing")
        tf.global_variables_initializer().run()
        for i in range(args.num_epochs):
            for j in range(args.n_save):
                    logging.info(
                        '=' * 19 + ' Epoch %d: %d/%d' + '=' * 19 + '\n', i+1, j+1, args.n_save)
                    logging.info('Training on training set')
                    # training step
                    ppl, train_summary_str, global_step = train_model.run_epoch(
                        session,
                        6,
                        inputs,
                        outputs,
                        is_training=True,
                        verbose=args.verbose,
                        freq=args.progress_freq,
                        divide_by_n=args.n_save)
                    # record the summary
                    train_writer.add_summary(train_summary_str, global_step)
                    train_writer.flush()
                    # save model
                    saved_path = saver.save(session, args.save_model,
                                            global_step=train_model.global_step)
                    logging.info('Latest model saved in %s\n', saved_path)
                    logging.info('Evaluate on validation set')

                    # valid_ppl, valid_summary_str, _ = valid_model.run_epoch(
                    valid_ppl, valid_summary_str, _ = valid_model.run_epoch(
                        session,
                        6,
                        inputs,
                        outputs,
                        is_training=False,
                        verbose=args.verbose,
                        freq=args.progress_freq)

                    # save and update best model
                    best_model = best_model_saver.save(
                        session,
                        args.save_best_model,
                        global_step=train_model.global_step)
                    best_valid_ppl = valid_ppl
                    valid_writer.add_summary(valid_summary_str, global_step)
                    valid_writer.flush()
                    logging.info('Best model is saved in %s', best_model)
        saver.restore(session, best_model)
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
        print(test_model.probs.name)
        frozenGraph = graph_util.convert_variables_to_constants( # freeze the graph
            session,
            tf.get_default_graph().as_graph_def(), # use the default graph
            ["evaluation/input", "evaluation/softmax/output", "initial_state_names", "final_state_names", "zeros"], # preserve all these OPs
        )
        tf.train.write_graph(frozenGraph, 'models', 'cmd_rnn.pb', as_text=False)

        #test_ppl, _, _ = test_model.run_epoch(session, test_size, test_batches, is_training=False, verbose=args.verbose, freq=args.progress_freq)



if __name__ == '__main__':
    main()
