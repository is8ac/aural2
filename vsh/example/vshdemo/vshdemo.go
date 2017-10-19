package main

import (
  "io/ioutil"
  "fmt"
  "os"
  "github.ibm.com/Blue-Horizon/aural2/vsh"
  "github.ibm.com/Blue-Horizon/aural2/libaural2"
)

// argmax returns the index of the largest elements of the list.
func argmax(probs []float32) (cmd libaural2.Cmd, prob float32) {
	for i, val := range probs {
		if val > prob {
			prob = val
			cmd = libaural2.Cmd(int32(i))
		}
	}
	return
}

func main(){
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
    fmt.Println(argmax(result))
  }
}
