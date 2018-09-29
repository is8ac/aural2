package main

import (
	"encoding/json"
	"fmt"
	"net"
	"strconv"
	"sync"
	"time"

	"os"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/vsh"
	"github.ibm.com/Blue-Horizon/aural2/vsh/intent"
)

type intentMsg struct {
	Name string    `json:"name"`
	Prob float32   `json:"prob"`
	TS   time.Time `json:"ts"`
}

func initMakeSendIntentMsgAction(intentChan chan intentMsg) func(string, float32) vsh.Action {
	return func(intentName string, minProb float32) (action vsh.Action) {
		action = vsh.Action{
			MinActivationProb: minProb,
			MaxResetProb:      0.2,
			HandlerFunction: func(prob float32) {
				fmt.Println(intentName, prob)
				intentChan <- intentMsg{
					Name: intentName,
					Prob: prob,
					TS:   time.Now(),
				}
			},
		}
		return
	}
}

func startVsh(
	saveClip func(*libaural2.AudioClip),
	stepInferenceFuncs map[libaural2.VocabName]func(*tf.Tensor) ([]float32, error),
	beforeShutdown func(),
) (
	dump func() *libaural2.AudioClip,
) {
	// connect to the audio stream
	conn, err := net.Dial("tcp", "microphone:48926")
	if err != nil {
		logger.Println("failed to connect to microphone:48926, trying localhost:48926")
		conn, err = net.Dial("tcp", "localhost:48926")
		if err != nil {
			panic(err)
		}
	}
	listenAddr := "aural2:49610"
	l, err := net.Listen("tcp", listenAddr)
	if err != nil {
		fmt.Println("Error aural2 on", listenAddr, err.Error(), ", trying localhost")
		listenAddr = "localhost:49610"
		l, err = net.Listen("tcp", listenAddr)
		if err != nil {
			fmt.Println("Error listening:", err.Error())
			os.Exit(1)
		}
	}

	fmt.Println("Listening for tcp connections on", listenAddr)
	resultChan, dump, err := vsh.Init(conn, stepInferenceFuncs)
	if err != nil {
		panic(err)
	}
	connsMap := map[int]*json.Encoder{}
	var connsIndex int
	connsMutex := sync.Mutex{}
	intentsChan := make(chan intentMsg)
	makeSendIntentMsgAction := initMakeSendIntentMsgAction(intentsChan)
	go func() {
		for {
			msg := <-intentsChan
			for i, encoder := range connsMap {
				logger.Print("conn", i)
				go func(writer *json.Encoder, connID int) {
					err := writer.Encode(msg)
					logger.Println("writing message to conn", i)
					if err != nil {
						logger.Println("got error:", err.Error(), "closing connection", i)
						delete(connsMap, i)
					}
				}(encoder, i)
			}
		}
	}()
	eb := vsh.NewEventBroker(resultChan)
	eb.Register(intent.Vocabulary.Name, intent.PlayMusic, "play0.5", makeSendIntentMsgAction("play0.5", 0.5))
	eb.Register(intent.Vocabulary.Name, intent.PlayMusic, "play0.8", makeSendIntentMsgAction("play0.8", 0.8))
	eb.Register(intent.Vocabulary.Name, intent.PlayMusic, "play0.9", makeSendIntentMsgAction("play0.9", 0.9))
	eb.Register(intent.Vocabulary.Name, intent.PlayMusic, "play0.95", makeSendIntentMsgAction("play0.95", 0.95))
	eb.Register(intent.Vocabulary.Name, intent.PlayMusic, "play0.99", makeSendIntentMsgAction("play0.99", 0.99))

	eb.Register(intent.Vocabulary.Name, intent.PauseMusic, "pause0.5", makeSendIntentMsgAction("pause0.5", 0.5))
	eb.Register(intent.Vocabulary.Name, intent.PauseMusic, "pause0.8", makeSendIntentMsgAction("pause0.8", 0.8))
	eb.Register(intent.Vocabulary.Name, intent.PauseMusic, "pause0.9", makeSendIntentMsgAction("pause0.9", 0.9))
	eb.Register(intent.Vocabulary.Name, intent.PauseMusic, "pause0.95", makeSendIntentMsgAction("pause0.95", 0.95))
	eb.Register(intent.Vocabulary.Name, intent.PauseMusic, "pause0.99", makeSendIntentMsgAction("pause0.99", 0.99))

	eb.Register(intent.Vocabulary.Name, intent.SkipSong, "skip0.5", makeSendIntentMsgAction("skip0.5", 0.5))
	eb.Register(intent.Vocabulary.Name, intent.SkipSong, "skip0.8", makeSendIntentMsgAction("skip0.8", 0.8))
	eb.Register(intent.Vocabulary.Name, intent.SkipSong, "skip0.9", makeSendIntentMsgAction("skip0.9", 0.9))
	eb.Register(intent.Vocabulary.Name, intent.SkipSong, "skip0.95", makeSendIntentMsgAction("skip0.95", 0.95))
	eb.Register(intent.Vocabulary.Name, intent.SkipSong, "skip0.99", makeSendIntentMsgAction("skip0.99", 0.99))

	eb.Register(intent.Vocabulary.Name, intent.Next, "next0.95", makeSendIntentMsgAction("next0.95", 0.95))
	eb.Register(intent.Vocabulary.Name, intent.Previous, "previous0.95", makeSendIntentMsgAction("previous0.95", 0.95))

	eb.Register(intent.Vocabulary.Name, intent.ShutDown, "shutdown", vsh.Action{
		MinActivationProb: 0.99,
		MaxResetProb:      0.5,
		HandlerFunction: func(prob float32) {
			beforeShutdown()
		},
	})
	var uploadMinActivationProb float32 = 0.98
	saveClipThreshold := os.Getenv("SAVE_CLIP_THRESHOLD")
	if saveClipThreshold != "" {
		parsedFloat, err := strconv.ParseFloat(saveClipThreshold, 64)
		if err != nil {
			logger.Println("Can't parse SAVE_CLIP_THRESHOLD:", err.Error())
		} else {
			uploadMinActivationProb = float32(parsedFloat)
		}
	}
	eb.Register(intent.Vocabulary.Name, intent.UploadClip, "upload", vsh.Action{
		MinActivationProb: uploadMinActivationProb,
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
	go func() {
		for {
			conn, err := l.Accept()
			if err != nil {
				fmt.Println("Error accepting: ", err.Error())
			} else {
				connsMutex.Lock()
				connsIndex++
				index := connsIndex
				fmt.Println("adding connection", index)
				encoder := json.NewEncoder(conn)
				connsMap[connsIndex] = encoder
				connsMutex.Unlock()
			}
		}
	}()
	return
}
