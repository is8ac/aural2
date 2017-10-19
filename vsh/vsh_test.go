package vsh

import (
  "testing"
  "io/ioutil"
  "fmt"
  "os"
)

func TestInit(t *testing.T){
  reader, err := os.Open("testaudio.raw")
  if err != nil {
    t.Fatal(err)
  }
  graphBytes, err := ioutil.ReadFile("cmd_rnn.pb")
  if err != nil {
    t.Fatal(err)
  }
  resultChan, err := Init(reader, graphBytes)
  if err != nil {
    t.Fatal(err)
  }
  fmt.Println("reading")
  for result := range resultChan {
    fmt.Println(result)
  }
}
