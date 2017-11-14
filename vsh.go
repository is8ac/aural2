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

func startVsh(saveClip func(*libaural2.AudioClip), client *mpd.Client, models map[libaural2.VocabName]tf.SavedModel) {
	resultChan, dump, err := vsh.Init(os.Stdin, models)
	if err != nil {
		panic(err)
	}
	var speakerWorks = false
	var micWorks = false
	eb := vsh.NewEventBroker(resultChan)
	eb.Register(intent.Vocabulary.Name, intent.SkipSong, "skip", vsh.MakeDefaultAction(func() { client.Next() }))
	eb.Register(intent.Vocabulary.Name, intent.PauseMusic, "pause", vsh.MakeDefaultAction(func() { client.Pause(true) }))
	eb.Register(intent.Vocabulary.Name, intent.PlayMusic, "play", vsh.MakeDefaultAction(func() { client.Pause(false) }))
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
		},
	})
	eb.Register(word.Vocabulary.Name, word.Hello, "sound_test", vsh.MakeDefaultAction(func() {
		speakerWorks = true
		micWorks = true
		logger.Println("Sound works")
		eb.Unregister(word.Vocabulary.Name, word.Hello, "sound_test")
	}))

	time.Sleep(2 * time.Second)
	// now start playing audio.
	f, err := os.Open("static/hello.wav")
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
	for {
		time.Sleep(time.Hour)
	}
}
