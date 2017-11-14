# aural2
LSTM based speech command recognition.

## High Level Data Flow
16000Hz int16 PCM is captured from microphone.
512 sample wide strides are taken, and transformed into Mel-frequency cepstral coefficients (MFCCs).
Each MFCC consists of 13 float32s.
If state is of size 256, and we have 5 exclusive outputs, each cell of the LSTM takes the initial state of shape `[256]` and the input of shape `[13]`, and returns the final state of size `[256]`, and an output of shape `[5]`.

```
 [5]          [5]          [5]
(cell)[256]->(cell)[256]->(cell)[256]
  ^            ^            ^
 [13]         [13]         [13]
```

This output can be any set of exclusive states of the world.
Examples of states of the world which an LSTM running on MFCCs of audio can be used to classify include:
- The word which the user is saying.
- The action which the user wants the computer to perform.
- The emotional state of the user.
- The person who is currently talking.
Each set of exclusive states is treated as its own vocabulary, and is classified using its own, separately trained model.

When training, MFCCs are fed into the graph, producing actual outputs.
These actual outputs are subtracted from what I have defined as the target outputs.
The difference squared is used as the loss, and is back propagated over the variables in the model, updating them slightly.

Training with batch size of 7, and 100 cells unrolled, therefore takes an float32 input of shape `[7, 100, 13]`, and an int32 target of shape `[7, 100]`.

Running on a GTX 1060, training for 3000 mini batches, takes ~2.5 minutes, producing a 5.4MB model.

vsh is a wrapper lib around the LSTM.
It maintains a 10 second ring buffer of audio, allowing the last 10 seconds of audio to be written to disk when needed.
vmpc is a simple music player client built using vsh.
vmpc uses an intent model to recognize the action which the user wants it to perform.
One such intent is the mistake intent, whose action is to writes the last 10 seconds of audio to disk.
As I use vmpc to control music, when its actions fail to match my wishes, I tell it so, triggering the mistake intent, writing the last 10 seconds of audio to disk.
In this way, I collect many 10 second clips containing utterances which the current model does not correctly recognize.
These clips, I label and train on.

## Aural RL
To learn, an NN does not need to be told what the correct was output, just that the output it gave is wrong.
Therefore, provided that we have a model capable of classification emotional state from face and voice, we can use pure RL to train an LSTM to better interact with the user.

As a further refinement, I propose using an LSTM running backwards in time to identify the utterance whose misinterpretation caused the user to experience suffering.
