Classifying a finite number of human intents from voice with slight negative latency, and solving the use mention distinction
=========================

# Abstract
We present Aural2, an LSTM model and labeling and training infrastructure, capable of learning to use voice to classify the action which a user wished to be performed.
It is usually capable of correctly classify an intent before the user has finished speaking;
assuming latency to be  measured from end of utterance, aural2 has slight negative latency.
Furthermore, Aural2 will automatically learn to integrate past context into its classification, allowing it to learn to accurately solve the use-mention distinction, negating the need for wake-words.

We describe the architecture of Aural2, its failings, and future directions for development.


# Introduction
Most natural language parsing systems operate at the level of words.
Given a string of words, they return the action which the user wishes to be performed.
To pass raw speech to such a system, it must first be transcribed into a series of words.
Such a series of words contains less information then the original audio.
Therefore, purely on information theoretic grounds, we should expect word level NLP on top of speech to text (STT), to be less accurate then a system which transforms sound directly into intents.

However, this improvement in accuracy comes with a significantly cost to Aural2s ability to scale to large vocabularies.

Aural2 contains multiple vocabularies, one for intents, another for words, etc.
However this paper will primarily focus on the intent vocabulary.

# Technologies used
## TensorFlow compute graph (TF graph)
TF Compute Graph is a purely functional language for defining graphs of transformations on multidimensional tensors, which the TensorFlow runtime lazily evaluates using whatever hardware is available.
Each node in the graph takes zero or more tensors as input, and returns one or more tensors as output.
As recursion and loops are forbidden, TF compute graphs are provably halting, execute in approximately fixed time, and are not Turing complete.
A TF graph can be stored as a GraphDef protobuf file.
This GraphDef is cross platform, able to be evaluated by the TensorFlow runtime on any sported hardware, whether that be an x86 or ARM CPU, or NVIDIA GPU.

Although, it is possible to write a text encoded GraphDef by hand, it is far more common to construct the graph using some more general purpose language such as python or golang, either for export as a GraphDef for later use, or for immediate evaluation.

Aural2 does both, constructing the main training graph in python at build time, and the numerous other supporting graph at initialization time.

Aural2s extensive use of TF compute graphs allows it to take advantage of dedicated hardware accelerators such as NVIDIA GPUs, while still running with full capabilities, albeit somewhat slower, on a generic CPU.

The shape of a tensor is denoted with a list of its dimensions.
For example, `[1,2,3]` denotes a three dimensional tensor; a list containing one list of two lists of three numbers.
All numbers are float32 unless otherwise noted.

## Mel-frequency cepstral coefficient (MFCC)
It is computationally expensive to train a neural net directly on the wave form of audio.
Therefore, it is common practice to train the NN on fingerprints of windows of the waveform.
MFCC is a fingerprinting algorithm optimized for human speech.
In the configuration used in Aural2, MFCC produces 13 float32s.

## Long short-term memory neural nets (LSTM)
LSTMs are a type on Recurrent Neural Net (RNN).
Unlike traditional RNNs each cell of which must actively choose to pass state to the next cell, LSTMs must actively choose to forget state.
They are therefore suitable for use where the easy persistence of long term state is desirable.
A single cell of an LSTM takes the current state and an input, and returns a next state and an output of the same size as the state.
As used in Aural2, the state consists of 256 float32s and the input is the 13 values of the MFCC.

# Architecture
The primary TF compute graphs use by Aural are as follows.

- Step MFCC: Takes 1024 bytes of int16 PCM. Returns a `[13]` tensor.
- Clip MFCC: Takes 160,000 bytes of int16 PCM. Returns a `[312,13]` tensor.
- Step inference LSTM: Takes a `[256]` state, and a `[13]` input tensor. Returns a `[50]` one-hot output, and a final state of `[256]`.
- Clip inference LSTM: Takes a `[312,13]` input tensor. Returns a `[312, 50]` output tensor.
- Train LSTM: Takes a `[7,100,13]` input tensor and a `[7,100]` int32 target tensor. Updates the weights and biases when evaluated.

Note that the step and clip inference and training LSTM graphs share weight and bias variables.

There are also various TF graphs for generating visualizations of data.
These graphs will not be discuses in detail here.

Sound is recorded at a sample rate of 16000Hz with 16 bit depth.
512 sample windows are read and, both written to a ring buffer and fed into a TF graph to compute the MFCC, producing a tensor of shape `[13]`.
This tensor is used as the input to the one or more inference LSTM graphs.
The output of the LSTMs, once `matmul`ed, is a list of 50 floats between 0 and 1.
The `n`th element of the output is the probability that the world is in state `n`.
As the world must be in one and only one state, the probabilities of the various states always sum to 1.

## Training
### Data collection
As mentioned before, aural2 maintains a ring buffer of the past 10 seconds of audio.
At any time, the past 10 seconds may be written to disk.
This may be triggered by a REST API, or by the user saying "upload", "mistake", or otherwise expressing their intent that the audio should be saved.
Raw audio is stores as files in a directory on the local storage.
Metadata about the raw audio clip is stored in a local boltDB.

### Tagging
Aural2 serves a web UI listing the audio clips which have been captured, each clip name linking to the labeling UI for that clip.
The labeling UI contains various visualizations of the audio and the labels assigned to it by the current state of the clip inference graph.
The visualizations of the labels are regularly reloaded, allowing users to watch the output of the model change in real time.
The user may listen to the audio, see visualizations of it, and create labels on the audio.
Each label contains the hash of the audio clip which it applies to, the beginning and end of the period of the clip to which it applies, and the state which the user says the world was in during that period.
The user uses the labeling UI to create a set of labels marking all periods during which the world was in a state other then the nil state.
It is forbidden for two label in a set to overlap.
Once the user has created all labels for the clip, the label set is submitted to the aural2 server which both writes it to the boltDB, and adds both the label set and the corresponding audio clip to the training data object.

### Training
The training data object contains two maps, one of inputs and one of target, where each input is of shape `[312, 13]` and type float32 and each target is of shape `[312]` and type int32.

When aural2 starts, it reads the list of label sets from the boltDB and the audio clips to which they refer, and adds them to the training data object.

When a label set and clip are added to the training data object, the clip is fed into the batch MFCC TF graph to transform it to a tensor of shape `[312, 13]`, which is added to the inputs map, and the label sets are transformed into a list of the integer state ID at each of its 312 time steps, which is added to the targets map.

In this way, a set of preprocessed inputs and targets is created from existing training data on startup, and added to when new labels are submitted.

However, these are sequences with a length of 312 steps.
To train an LSTM on sequences of `n` time steps, one must unroll the cells, creating a training graph containing `n` copies of the LSTM cell.
It is computationally expensive and unnecessary to train on 312 sample sequences.
Aural2 currently trains on 100 sample long sequences.

The data preparation loop chooses 7 audio clip IDs at random, and, for each clip, creates a random 100 sample long period for it.
It then slices this period from both the input and the target.
The data preparation thread then converts the 7 100 sample periods of inputs and corresponding 7 100 sample periods of outputs to a pair of tensors of shape `[7, 100, 13]`, and `[7, 100]`.
It writes this mini batch to the mini batch channel, blocking until the channel has only three more mini batches.

The training loop reads a mini batch from the mini batch channel and evaluates the training LSTM graph on the inputs and targets, thereby updating the weights and biases.

In this way, the training loop is always suppled with a buffer of mini batches randomly drawn from a recent state of the training data, and training data preparation is free to hog CPU resources while the train loop is blocked by the GPU doing training.

It should be reiterated that the graph used for performing inference on incoming audio, the graph for performing batch inference on whole audio clips, and the training graph, while distinct graphs, share a single set of variables.
The weight and bias variables are updated by the training graph, and the accuracy with which aural2 classifies the state of the world increases.
The variables stay on the GPU, or whatever compute device TensorFlow has decided to use,  and need never leave.
TensorFlow transparently handles locking to ensure that the various graphs can read and write to the shared memory safely.


# Shortcomings
Although perhaps superior to existing technology in latency, simplicity, and speed of training, aural2 is inherently of limited scalability, and as such, can have no ambition for use in full vocabulary natural language parsing.

Additionally, aural2 leaves much to be desirable with regard to the labeling of training data.
While it can save audio on command, this merely helps to collect unlabeled audio rich in states which a past state of the user decided the model had misclassified; it does nothing to label the audio with the true state.

A significant improvement to aural2 would be to make use of user feedback to directly train via reinforcement learning.
This would require an additional model to classify user voice, facial expressions, etc, into an emotional state.
This hybrid system of training an emotion classifier via supervised learning, which can then be used to train an intent model via reinforcement learning is somewhat inelegant however.
