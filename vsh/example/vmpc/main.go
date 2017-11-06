package main

import (
	"bytes"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/faiface/beep"
	"github.com/faiface/beep/speaker"
	"github.com/faiface/beep/wav"
	"github.com/fhs/gompd/mpd"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/vsh"
	"github.ibm.com/Blue-Horizon/aural2/vsh/intent"
	"github.ibm.com/Blue-Horizon/aural2/vsh/word"
)

var logger = log.New(os.Stdout, "vmpc: ", log.Lshortfile|log.LUTC|log.Ltime|log.Ldate)

func main() {
	var saveClip func(*libaural2.AudioClip) error
	basePath := os.Args[1]
	if basePath[:7] == "http://" { // if the base is a url
		saveClip = func(clip *libaural2.AudioClip) (err error) {
			_, err = http.Post(os.Args[2]+"/sample/upload", "application/octet-stream", bytes.NewReader(clip[:]))
			return
		}
	} else { // else assume it is a directory path
		saveClip = func(clip *libaural2.AudioClip) (err error) {
			err = ioutil.WriteFile(basePath+"/"+clip.ID().FSsafeString(), clip[:], 0644)
			return
		}
	}
	// Connect to MPD server
	client, err := mpd.Dial("tcp", "localhost:6600")
	if err != nil {
		log.Fatalln(err)
	}
	defer client.Close()
	go func() {
		for {
			client.Ping()
			time.Sleep(time.Second)
		}
	}()
	w, err := mpd.NewWatcher("tcp", ":6600", "")
	if err != nil {
		log.Fatalln(err)
	}
	defer w.Close()
	go func() {
		for err := range w.Error {
			logger.Println("Error:", err)
		}
	}()
	go func() {
		for subsystem := range w.Event {
			//log.Println("Changed subsystem:", subsystem)
		}
	}()

	intentGraphBytes, err := ioutil.ReadFile("models/intent.pb")
	if err != nil {
		panic(err)
	}
	wordGraphBytes, err := ioutil.ReadFile("models/word.pb")
	if err != nil {
		panic(err)
	}
	graphs := map[libaural2.VocabName][]byte{
		intent.Vocabulary.Name: intentGraphBytes,
		word.Vocabulary.Name:   wordGraphBytes,
	}
	resultChan, dump, err := vsh.Init(os.Stdin, graphs)
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
		MaxResetProb:      0.1,
		CoolDownDuration:  10 * time.Second,
		TimeLastCalled:    time.Now(),
		HandlerFunction: func(prob float32) {
			logger.Println("uploading in 2 seconds")
			time.Sleep(2 * time.Second)
			clip := dump()
			if err = saveClip(clip); err != nil {
				logger.Println(err)
			} else {
				logger.Println("audio saved")
			}
		},
	})
	eb.Register(word.Vocabulary.Name, word.Hello, "sound_test", vsh.MakeDefaultAction(func() {
		speakerWorks = true
		micWorks = true
		logger.Println("Sound works")
		eb.Unregister(word.Vocabulary.Name, word.Hello, "sound_test")
	}))

	time.Sleep(5 * time.Second)
	// now start playing audio.
	f, err := os.Open("audio/hello.wav")
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
	time.Sleep(time.Hour)
}
