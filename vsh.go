package main

import (
	"time"

	"github.com/faiface/beep"
	"github.com/faiface/beep/speaker"
	"github.com/faiface/beep/wav"
	"github.com/fhs/gompd/mpd"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/vsh"
	"github.ibm.com/Blue-Horizon/aural2/vsh/intent"
	"github.ibm.com/Blue-Horizon/aural2/vsh/word"
	"os"
)

func startVsh(
	saveClip func(*libaural2.AudioClip),
	stepInferenceFuncs map[libaural2.VocabName]func(*tf.Tensor) ([]float32, error),
	beforeShutdown func(),
) (
	dump func() *libaural2.AudioClip,
) {
	var pauseAudio func()
	var playAudio func()
	var skipTrack func()
	client, err := mpd.Dial("tcp", "localhost:6600")
	if err != nil {
		logger.Println("failed to connect to mpd")
		pauseAudio = func() { logger.Println("pausing (but no mpd connection)") }
		playAudio = func() { logger.Println("playing (but no mpd connection)") }
		skipTrack = func() { logger.Println("skiping (but no mpd connection)") }
	} else {
		pauseAudio = func() { client.Pause(true) }
		playAudio = func() { client.Pause(false) }
		skipTrack = func() { client.Next() }
		go func() {
			for {
				client.Ping()
				time.Sleep(time.Second)
			}
		}()
	}

	resultChan, dump, err := vsh.Init(os.Stdin, stepInferenceFuncs)
	if err != nil {
		panic(err)
	}
	var speakerWorks = false
	var micWorks = false
	eb := vsh.NewEventBroker(resultChan)
	eb.Register(intent.Vocabulary.Name, intent.ShutDown, "shutdown", vsh.MakeDefaultAction(beforeShutdown))
	eb.Register(intent.Vocabulary.Name, intent.SkipSong, "skip", vsh.MakeDefaultAction(skipTrack))
	eb.Register(intent.Vocabulary.Name, intent.PauseMusic, "pause", vsh.MakeDefaultAction(pauseAudio))
	eb.Register(intent.Vocabulary.Name, intent.PlayMusic, "play", vsh.MakeDefaultAction(playAudio))
	eb.Register(intent.Vocabulary.Name, intent.UploadClip, "upload", vsh.Action{
		MinActivationProb: 0.9,
		MaxResetProb:      0.5,
		CoolDownDuration:  10 * time.Second,
		TimeLastCalled:    time.Now(),
		HandlerFunction: func(prob float32) {
			logger.Println("uploading in 2 seconds")
			time.Sleep(2 * time.Second)
			clip := dump()
			saveClip(clip)
			logger.Println("saved clip:", clip.ID())
		},
	})
	eb.Register(word.Vocabulary.Name, word.Hello, "sound_test", vsh.MakeDefaultAction(func() {
		speakerWorks = true
		micWorks = true
		logger.Println("Sound works")
		eb.Unregister(word.Vocabulary.Name, word.Hello, "sound_test")
	}))

	go func() {
		time.Sleep(2 * time.Second)
		// now start playing audio.
		f, err := os.Open("webgui/static/hello.wav")
		if err != nil {
			logger.Println(err)
		}
		s, format, err := wav.Decode(f)
		if err != nil {
			logger.Println(err)
		}
		speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10))
		speaker.Play(beep.Seq(s, beep.Callback(func() {
			if !speakerWorks {
				logger.Println("no speaker")
			}
		})))
	}()
	return
}
