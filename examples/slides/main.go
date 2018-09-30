package main

import (
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

type intentMsg struct {
	Name string    `json:"name"`
	Prob float32   `json:"prob"`
	TS   time.Time `json:"ts"`
}

func main() {
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
	msgChan := make(chan intentMsg)
	go serveGui(msgChan)
	for {
		err = decoder.Decode(&intent)
		if err != nil {
			panic(err)
		}
		if (intent.Name == "previous0.95") ||
			(intent.Name == "next0.95") {
			fmt.Println(intent)
			msgChan <- intent
		}
	}
}

func returnServeDeck() (serveIndex func(http.ResponseWriter, *http.Request)) {
	serveIndex = func(w http.ResponseWriter, r *http.Request) {
		var index = template.Must(template.ParseFiles("templates/deck.html"))
		params := struct{}{}
		err := index.Execute(w, params)
		if err != nil {
			fmt.Println("error serving index: ", err)
		}
	}
	return
}

func msgBridge(wsConns *wsConns, msgChan chan intentMsg) {
	for {
		msg := <-msgChan
		msgString, err := json.Marshal(msg)
		if err != nil {
			panic(err)
		}
		for _, c := range wsConns.Conns {
			c.WriteMessage(websocket.TextMessage, msgString)
		}
	}
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true }, // allow any client to connect.
}

func returnOnConnect(wsConns *wsConns) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		log.Println("ws connected")
		c, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Print("upgrade:", err)
			return
		}
		wsConns.Conns = append(wsConns.Conns, c)
	}
}

type wsConns struct {
	sync.Mutex
	Conns []*websocket.Conn
}

func serveGui(guiMsgChan chan intentMsg) {
	conns := wsConns{
		Mutex: sync.Mutex{},
		Conns: []*websocket.Conn{},
	}
	go msgBridge(&conns, guiMsgChan)
	http.HandleFunc("/ws", returnOnConnect(&conns))
	http.HandleFunc("/", returnServeDeck())
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))
	log.Fatal(http.ListenAndServe(":27637", nil))
}
