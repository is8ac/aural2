import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1


        cells = []
        for _ in range(args.num_layers): # for each layer in the unroled representation,
            cell = rnn.BasicLSTMCell(args.rnn_size) # create the LSTM cell,
            cells.append(cell) # and append it to the list of all cells.

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True) # combine all the cells in the stack
        zeroState = tf.identity(cell.zero_state(1, tf.float32), name="cell_zero_state")
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name="input_data") # place holder for the input (as onehot?)
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name="targets") # not sure what this is. For training?
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size], name="output")


        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits, name="probs")
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        prime = "The "
        state = sess.run(self.cell.zero_state(1, tf.float32))
        print(state[0][0][0][0:10])
        for char in prime[:-1]:
            x = np.zeros((1, 1)) # init x as empty 2d 1 x 1
            x[0, 0] = vocab[char] # set it to number of char
            print(x[0, 0])
            feed = {self.input_data: x, self.initial_state: state} # construct the feed
            [state] = sess.run([self.final_state], feed) # pull on final_state. Don't bother pulling on probs.
            print(state[0][0][0][0:5])

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime # ret is just the concat of chars
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            #print(x)
            feed = {self.input_data: x, self.initial_state: state} #feed it both data and state,
            [probs, state] = sess.run([self.probs, self.final_state], feed) # pull on puth probs and state
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
