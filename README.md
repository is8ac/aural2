# aural2
LSTM based speech command recognition.


# Quickstart
## x86:
Create a network:
```
docker network create aural2
```
Then start the microphone container:
```
docker run -it --rm --privileged --name microphone --net=aural2 --net-alias=microphone --publish=48926:48926 is8ac/example_ms_x86_microphone:1.2.6
```
It should print `Listening for tcp connections on microphone:48926` on the last line.

Then start the aural2 container:
```
docker run -it --rm --name aural2 --net=aural2 -p 48125:48125 --net-alias=aural2 -v /tmp/aural2:/persist is8ac/aural2_x86:0.3.4
```
Say "upload" a few times and then use `curl http://localhost:48125/saveclip` to save a 10 second sample of audio.
Go to `http://localhost:48125/intent/index`, and click on the sample you just captured.
Use space to play/pause, and arrow keys to navigate. Hold down the 'u' key while the cursor moves over the word you just said. When you are done labeling the intents, use alt+s, or close the tab to save.

Use `curl -X POST -d "5" http://localhost:48125/sleepms` to make it train faster.

# build
## With docker
```
make
```
This will use multi stage docker build.

## Directly on host

### Dependencies
- Python Tensorflow (build):
```
pip install tensorflow
```
- Tensorflow C libs. GPU or CPU. https://www.tensorflow.org/install/install_go

- Golang 1.9
- Gopherjs:
```
go get -u github.com/gopherjs/gopherjs
```

Generate the training graph:
```
python gen_train_graph.py
```
Transpile the web GUI to JS:
```
gopherjs build -o webgui/static/main.js webgui/main.go
```

Finally build the golang binary.
```
go build
```
To run directly on host:
```
arecord -D plughw:3 -r 16000 -t raw -f S16_LE -c 1 - | ./aural2
```
Or when developing:
```
arecord -D plughw:3 -r 16000 -t raw -f S16_LE -c 1 - | go run *.go
```

Replace `plughw:3` with the name of your microphone.

You will need to install and configure mpd if you want aural2 to be able to control your music.

Raw audio will be stored in `audio/`.
The database of labels will be stored at `label_store.db`.

# Usage
The index page for the intent vocabulary is served at `http://localhost:48125/intent/index`.
Initially it will be empty.

As the upload intent is not yet trained, you must bootstrap with the `/saveclip` REST API.
Say "upload" or other wise express your desire that aural should save some audio.
Then run:
```
curl http://localhost:48125/saveclip
```
Please be careful when using this API.
It has fewer safeties then the upload intent.

Refresh the index page. You should now see a link. Click on it.

You should see spectrogram of the audio. This is the labeling UI.
To play and pause use the space bar.
To navigate forward and back use arrow keys.
Up / Down for large movements, Left / Right for small.

To label the audio, position the red cursor at the beginning of the utterance.
Depress the 'u' key.
Play until the cursor is positioned at the end of the utterance, then release the 'u' key.
You should now see a green bar labeled "UploadClip"
Repeat until all the utterance are labeled.
Then press Alt-s to save the labels.

Now that it has some data, aural2 should start to train its model.
In the terminal in which aural2 is running you should see:
```
arl2: train.go:138: intent 1.5893356
arl2: train.go:138: intent 1.0943471
arl2: train.go:138: intent 0.8438847
...
```
If all goes well, perplexity should steadily decline.

Back in the browser window, you should see the lowest bar change from black to orange, and then fill in with colors to approximately match your labels. Do not worry if they do not exactly match.

Once you have trained the upload intent, you can start training other intents.

Say "Play". Observe that the `playAudio` intent fails to trigger.
Say "Upload". Aural should save the last 10 seconds of audio.
Refresh the index page, and label the new sample using the 'p' key for the `playAudio` intent.

For a complete list of intents and key bindings, see `vsh/intent/intent.go`

Repeat until aural2 does your bidding consistently.

Trained models are written to disk every 10 minutes.
If you wish to save models before terminating aural2, call the `/savemodels` API.
```
curl http://localhost:48125/savemodels
```

You can also train the `ShutDown` intent, which will write models to disk before terminating.

By default, Aural2 will sleep for 300 ms between each training mini batch so as to not starve the other applications running on the same hardware of CPU time.
But if you do not care about starving other applications of CPU, you can change the sleep time with the `/sleepms` REST API.
For example, to tell Aural2 to sleep for only 5 ms each iteration:
```
curl -X POST -d "5" http://<ipaddr>:48125/sleepms
```
Note this setting does not persist across restarts, so you will need to call this API each time you start Aural2 and want to train fast.


# Caveats:
- When running in docker, vsh cannot connect to mpd. It will fall back to just printing its actions.
- Currently only tested on x86_64 and aarch64 Linux. Will probably work OSX, or Linux on armhf, but has not been tested.

To reduce TensorFlow logging:
```
export TF_CPP_MIN_LOG_LEVEL=2
```

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

vsh is a wrapper lib around the LSTM.
It maintains a 10 second ring buffer of audio, allowing the last 10 seconds of audio to be written to disk when needed.
vmpc is a simple music player client built using vsh.
vmpc uses an intent model to recognize the action which the user wants it to perform.
One such intent is the mistake intent, whose action is to writes the last 10 seconds of audio to disk.
As I use vmpc to control music, when its actions fail to match my wishes, I tell it so, triggering the mistake intent, writing the last 10 seconds of audio to disk.
In this way, I collect many 10 second clips containing utterances which the current model does not correctly recognize.
These clips, I label and train on.
