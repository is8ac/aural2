package main

import (
	"time"

	"os"

	"github.com/fhs/gompd/mpd"
	"github.com/open-horizon/self-go-sdk/self"
	"github.com/satori/go.uuid"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/vsh"
	"github.ibm.com/Blue-Horizon/aural2/vsh/intent"
)

func startVsh(
	saveClip func(*libaural2.AudioClip),
	stepInferenceFuncs map[libaural2.VocabName]func(*tf.Tensor) ([]float32, error),
	beforeShutdown func(),
	bbHost string,
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
		pauseAudio = func() { logger.Println("pause"); client.Pause(true) }
		playAudio = func() { logger.Println("play"); client.Pause(false) }
		skipTrack = func() { logger.Println("next"); client.Next() }
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
	//var speakerWorks = false
	//var micWorks = false
	eb := vsh.NewEventBroker(resultChan)
	// try to connect to the Intu blackboard.
	bb, err := self.Init(bbHost, "aural2")
	if err == nil {
		eb.Register(intent.Vocabulary.Name, intent.PlayMusic, "intu_play", vsh.Action{
			MinActivationProb: 0.9,
			MaxResetProb:      0.2,
			TimeLastCalled:    time.Now(),
			HandlerFunction: func(prob float32) {
				guid, err := uuid.NewV4()
				if err != nil {
					logger.Println(err)
					return
				}
				thing := self.Thing{
					GUID:        guid.String(),
					Type:        "IThing",
					DataType:    "voice_intent",
					CreateTime:  float64(time.Now().Unix()),
					Text:        "play_music",
					Confidence:  float64(prob),
					Info:        "play_music",
					Name:        "play_music",
					State:       "ADDED",
					ECategory:   self.ThingCategoryPERCEPTION,
					FImportance: 1,
					FLifeSpan:   3600,
					Data:        map[string]string{"intent": "play_music"},
				}
				logger.Println("publishing play")
				if err := bb.Pub(self.TargetBlackboard, thing); err != nil {
					logger.Println(err)
				}
			},
		})
		eb.Register(intent.Vocabulary.Name, intent.PauseMusic, "intu_pause", vsh.Action{
			MinActivationProb: 0.9,
			MaxResetProb:      0.2,
			TimeLastCalled:    time.Now(),
			HandlerFunction: func(prob float32) {
				guid, err := uuid.NewV4()
				if err != nil {
					logger.Println(err)
					return
				}
				thing := self.Thing{
					GUID:        guid.String(),
					Type:        "IThing",
					DataType:    "voice_intent",
					CreateTime:  float64(time.Now().Unix()),
					Text:        "pause_music",
					Confidence:  float64(prob),
					Info:        "pause_music",
					Name:        "pause_music",
					State:       "ADDED",
					ECategory:   self.ThingCategoryPERCEPTION,
					FImportance: 1,
					FLifeSpan:   3600,
					Data:        map[string]string{"intent": "pause_music"},
				}
				if err := bb.Pub(self.TargetBlackboard, thing); err != nil {
					logger.Println(err)
				}
			},
		})
		eb.Register(intent.Vocabulary.Name, intent.SkipSong, "intu_skip", vsh.Action{
			MinActivationProb: 0.9,
			MaxResetProb:      0.2,
			TimeLastCalled:    time.Now(),
			HandlerFunction: func(prob float32) {
				guid, err := uuid.NewV4()
				if err != nil {
					logger.Println(err)
					return
				}
				thing := self.Thing{
					GUID:        guid.String(),
					Type:        "IThing",
					DataType:    "voice_intent",
					CreateTime:  float64(time.Now().Unix()),
					Text:        "skip_music",
					Confidence:  float64(prob),
					Info:        "skip_music",
					Name:        "skip_music",
					State:       "ADDED",
					ECategory:   self.ThingCategoryPERCEPTION,
					FImportance: 1,
					FLifeSpan:   3600,
					Data:        map[string]string{"intent": "skip_music"},
				}
				if err := bb.Pub(self.TargetBlackboard, thing); err != nil {
					logger.Println(err)
				}
			},
		})
	} else {
		logger.Println("can't connect to Intu blackboard")
	}

	eb.Register(intent.Vocabulary.Name, intent.ShutDown, "shutdown", vsh.Action{
		MinActivationProb: 0.99,
		MaxResetProb:      0.5,
		HandlerFunction: func(prob float32) {
			beforeShutdown()
		},
	})
	eb.Register(intent.Vocabulary.Name, intent.SkipSong, "skip", vsh.MakeDefaultAction(skipTrack))
	eb.Register(intent.Vocabulary.Name, intent.PauseMusic, "pause", vsh.MakeDefaultAction(pauseAudio))
	eb.Register(intent.Vocabulary.Name, intent.PlayMusic, "play", vsh.MakeDefaultAction(playAudio))
	eb.Register(intent.Vocabulary.Name, intent.UploadClip, "upload", vsh.Action{
		MinActivationProb: 0.95,
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
	//eb.Register(word.Vocabulary.Name, word.Hello, "sound_test", vsh.MakeDefaultAction(func() {
	//	speakerWorks = true
	//	micWorks = true
	//	logger.Println("Sound works")
	//	eb.Unregister(word.Vocabulary.Name, word.Hello, "sound_test")
	//}))

	//go func() {
	//	time.Sleep(2 * time.Second)
	//	// now start playing audio.
	//	f, err := os.Open("webgui/static/hello.wav")
	//	if err != nil {
	//		logger.Println(err)
	//	}
	//	s, format, err := wav.Decode(f)
	//	if err != nil {
	//		logger.Println(err)
	//	}
	//	speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10))
	//	speaker.Play(beep.Seq(s, beep.Callback(func() {
	//		if !speakerWorks {
	//			logger.Println("no speaker")
	//		}
	//	})))
	//}()
	return
}
