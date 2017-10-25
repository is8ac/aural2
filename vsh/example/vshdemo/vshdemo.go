package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"time"

	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/vsh"
	"github.ibm.com/Blue-Horizon/aural2/vsh/intent"
)

func uploadClip(clip *libaural2.AudioClip) (err error) {
	_, err = http.Post("http://localhost:48125/sample/upload", "application/octet-stream", bytes.NewReader(clip[:]))
	return
}

func main() {
	modelPath := os.Args[1]
	intentGraphBytes, err := ioutil.ReadFile(modelPath)
	if err != nil {
		panic(err)
	}
	graphs := map[libaural2.VocabName][]byte{
		libaural2.VocabName("intent"): wordGraphBytes,
	}
	resultChan, dump, err := vsh.Init(os.Stdin, graphs)
	if err != nil {
		panic(err)
	}
	lastUploaded := time.Now()
	for result := range resultChan {
		state, prob := vsh.Argmax(result["intent"])
		if state == intent.UploadClip && prob > 0.7 {
			fmt.Println(intent.Vocabulary.Names[state], prob)
			if lastUploaded.Add(10 * time.Second).Before(time.Now()) {
				fmt.Println("uploading")
				clip := dump()
				if err = uploadClip(clip); err != nil {
					fmt.Println(err)
				} else {
					fmt.Println("audio uploaded")
					lastUploaded = time.Now()
				}
			}
		}
	}
}
