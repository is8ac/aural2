package main

import (
	"fmt"
	"io/ioutil"
	"os"

	"github.ibm.com/Blue-Horizon/aural2/vsh"
)

func main() {
	modelPath := os.Args[1]
	graphBytes, err := ioutil.ReadFile(modelPath)
	if err != nil {
		panic(err)
	}
	resultChan, err := vsh.Init(os.Stdin, graphBytes)
	if err != nil {
		panic(err)
	}
	for result := range resultChan {
		fmt.Println(vsh.Argmax(result))
	}
}
