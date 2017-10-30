package main

import (
	"bytes"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/fhs/gompd/mpd"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/vsh"
	"github.ibm.com/Blue-Horizon/aural2/vsh/intent"
)

var logger = log.New(os.Stdout, "vmpc: ", log.Lshortfile|log.LUTC|log.Ltime|log.Ldate)

func main() {
	var saveClip func(*libaural2.AudioClip) error
	basePath := os.Args[2]
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
	modelPath := os.Args[1]
	intentGraphBytes, err := ioutil.ReadFile(modelPath)
	if err != nil {
		panic(err)
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
			log.Println("Error:", err)
		}
	}()
	go func() {
		for subsystem := range w.Event {
			log.Println("Changed subsystem:", subsystem)
		}
	}()

	graphs := map[libaural2.VocabName][]byte{
		libaural2.VocabName("intent"): intentGraphBytes,
	}
	resultChan, dump, err := vsh.Init(os.Stdin, graphs)
	if err != nil {
		panic(err)
	}
	lastUploaded := time.Now()
	var hasSkipped bool
	for result := range resultChan {
		if result["intent"][intent.SkipSong] < 0.5 {
			hasSkipped = false
		}
		state, prob := vsh.Argmax(result["intent"])
		if state == intent.UploadClip && prob > 0.7 {
			logger.Println(intent.Vocabulary.Names[state], prob)
			if lastUploaded.Add(10 * time.Second).Before(time.Now()) {
				go func() {
					logger.Println("uploading in 2 seconds")
					time.Sleep(2 * time.Second)
					clip := dump()
					if err = saveClip(clip); err != nil {
						logger.Println(err)
					} else {
						logger.Println("audio saved")
					}
				}()
				lastUploaded = time.Now()
			}
		}
		if state == intent.PlayMusic && prob > 0.9 {
			logger.Println(intent.Vocabulary.Names[state], prob)
			client.Pause(false)
		}
		if state == intent.PauseMusic && prob > 0.9 {
			logger.Println(intent.Vocabulary.Names[state], prob)
			client.Pause(true)
		}
		if state == intent.SkipSong && prob > 0.9 && !hasSkipped {
			logger.Println(intent.Vocabulary.Names[state], prob)
			client.Next()
			hasSkipped = true
		}
	}

	//if cmd == libaural2.Vocaloid && prob > 0.95 {
	//  logger.Println(cmd, prob)
	//  client.Pause(true)
	//  client.Clear()
	//  client.PlaylistLoad("vocaloid", -1, -1)
	//  client.Play(0)
	//}
	//if cmd == libaural2.Classical && prob > 0.95 {
	//  logger.Println(cmd, prob)
	//  client.Pause(true)
	//  client.Clear()
	//  client.PlaylistLoad("classical", -1, -1)
	//  client.Play(0)
	//}
	//if cmd == libaural2.Shakuhachi && prob > 0.9 {
	//  logger.Println(cmd, prob)
	//  client.Pause(true)
	//  client.Clear()
	//  client.PlaylistLoad("shakuhachi", -1, -1)
	//  client.Play(0)
	//}
}
