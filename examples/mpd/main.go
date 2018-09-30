package main

import (
	"encoding/json"
	"fmt"
	"net"
	"time"

	"github.com/fhs/gompd/mpd"
)

type intentMsg struct {
	Name string    `json:"name"`
	Prob float32   `json:"prob"`
	TS   time.Time `json:"ts"`
}

func main() {
	client, err := mpd.Dial("tcp", "localhost:6600")
	if err != nil {
		panic(err)
	}
	go func() {
		for {
			client.Ping()
			time.Sleep(time.Second)
		}
	}()
	// connect to the audio stream
	conn, err := net.Dial("tcp", "aural2:49610")
	fmt.Println("failed to connect to aural2:49610, trying localhost")
	if err != nil {
		conn, err = net.Dial("tcp", "localhost:49610")
		if err != nil {
			panic(err)
		}
	}
	var intent intentMsg
	decoder := json.NewDecoder(conn)
	for {
		err = decoder.Decode(&intent)
		if err != nil {
			panic(err)
		}
		//fmt.Println(intent)
		if intent.Name == "play0.9" {
			fmt.Println("playing")
			client.Pause(false)
		}
		if intent.Name == "pause0.9" {
			fmt.Println("pausing")
			client.Pause(true)
		}
		if intent.Name == "skip0.9" {
			fmt.Println("skipping")
			client.Next()
		}
	}
}
