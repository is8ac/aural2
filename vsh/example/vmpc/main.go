package main

import (
	"github.com/fhs/gompd/mpd"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/vsh"
	"io/ioutil"
	"log"
	"os"
  "time"
)

var logger = log.New(os.Stdout, "vmpc: ", log.Lshortfile|log.LUTC|log.Ltime|log.Ldate)

func main() {
	modelPath := os.Args[1]
	graphBytes, err := ioutil.ReadFile(modelPath)
	if err != nil {
		panic(err)
	}
	// Connect to MPD server
	client, err := mpd.Dial("tcp", "localhost:6600")
	if err != nil {
		log.Fatalln(err)
	}
	defer client.Close()
  go func(){
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

	resultChan, err := vsh.Init(os.Stdin, graphBytes)
	if err != nil {
		panic(err)
	}
	for result := range resultChan {
		cmd, prob := vsh.Argmax(result)
		if cmd == libaural2.Play && prob > 0.9 {
			logger.Println(cmd)
      client.Pause(false)
		}
		if cmd == libaural2.Pause && prob > 0.9 {
			logger.Println(cmd)
			client.Pause(true)
		}
    if cmd == libaural2.Vocaloid && prob > 0.9 {
      logger.Println("Vocaloid")
      client.Pause(true)
      client.Clear()
      client.PlaylistLoad("vocaloid", -1, -1)
      client.Play(0)
    }
    if cmd == libaural2.Classical && prob > 0.9 {
      logger.Println("Classical")
      client.Pause(true)
      client.Clear()
      client.PlaylistLoad("classical", -1, -1)
      client.Play(0)
    }
    if cmd == libaural2.Shakuhachi && prob > 0.9 {
      logger.Println("Shakuhachi")
      client.Pause(true)
      client.Clear()
      client.PlaylistLoad("shakuhachi", -1, -1)
      client.Play(0)
    }
	}
}
