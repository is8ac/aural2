package main

import (
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/go-humble/locstor"
	"honnef.co/go/js/dom"

	"github.com/gopherjs/gopherjs/js"
	"github.com/gopherjs/websocket/websocketjs"
)

type intentMsg struct {
	Name string    `json:"name"`
	Prob float32   `json:"prob"`
	TS   time.Time `json:"ts"`
}

func returnOnOpen(ws *websocketjs.WebSocket) func(*js.Object) {
	return func(ev *js.Object) {
		if err := ws.Send("browser has connected"); err != nil {
			log.Println(err)
		}
	}
}

var slideIDs = []string{
	"title",
	"problem",
	"traditional_approach",
	"causality",
	"pipeline",
	"training",
	"arch_diagram",
	"merits",
	"demerits",
	"end",
}

var currentSlide int

func onMsg(ev *js.Object) {
	msg := intentMsg{}
	err := json.Unmarshal([]byte(ev.Get("data").String()), &msg)
	if err != nil {
		fmt.Println("bad base64 msg:", err)
		return
	}
	fmt.Println(msg)
	if msg.Name == "next0.95" {
		nextSlide()
	}
	if msg.Name == "previous0.95" {
		previousSlide()
	}
}

func nextSlide() {
	if currentSlide < (len(slideIDs) - 1) {
		currentSlide++
		print("going forword to slide", currentSlide)
		d := dom.GetWindow().Document()
		slide := d.GetElementByID(slideIDs[currentSlide])
		slide.Underlying().Call("scrollIntoView", true)
	}
}

func previousSlide() {
	if currentSlide > 0 {
		currentSlide += -1
		print("going back to slide", currentSlide)
		d := dom.GetWindow().Document()
		slide := d.GetElementByID(slideIDs[currentSlide])
		slide.Underlying().Call("scrollIntoView", true)
	}
}

func onError(ev *js.Object) {
	fmt.Println("got error:", ev.String())
}

func main() {
	fmt.Println("aural2 slide deck web gui v0.0.1")
	host := js.Global.Get("location").Get("host").String()
	wsHost, err := locstor.GetItem("ws_host")
	if err != nil {
		fmt.Println(err)
	}
	if wsHost != "" {
		host = wsHost
		fmt.Println("connecting to ws server", wsHost)
	}
	ws, err := websocketjs.New("ws://" + host + "/ws") // Does not block.
	if err != nil {
		panic(err)
	}
	log.Println("state is:", ws.ReadyState)
	time.Sleep(1 * time.Second)
	log.Println("ws connected")
	if err = ws.Send("foo"); err != nil {
		log.Println(err)
	}
	log.Println("sent data")

	ws.AddEventListener("open", false, returnOnOpen(ws))
	ws.AddEventListener("message", false, onMsg)
	ws.AddEventListener("error", false, onError)
}
