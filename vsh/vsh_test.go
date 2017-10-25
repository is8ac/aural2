package vsh

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	"github.ibm.com/Blue-Horizon/aural2/libaural2"
)

func TestInit(t *testing.T) {
	reader, err := os.Open("testaudio.raw")
	if err != nil {
		t.Fatal(err)
	}
	wordGraphBytes, err := ioutil.ReadFile("cmd_rnn.pb")
	if err != nil {
		t.Fatal(err)
	}
	resultChan, _, err := Init(reader, map[libaural2.VocabName][]byte{libaural2.VocabName("word"): wordGraphBytes})
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println("reading")
	for result := range resultChan {
		_ = result
		//fmt.Println(result)
	}
}

func TestRing(t *testing.T) {
	reader, err := os.Open("testaudio.raw")
	if err != nil {
		t.Fatal(err)
	}
	rb := makeRing()
	buf := make([]byte, libaural2.StrideWidth*2)
	for {
		if _, err = reader.Read(buf); err != nil {
			break
		}
		rb.write(buf)
	}
	clip := rb.dump()
	fmt.Println(len(clip))
	ioutil.WriteFile("outclip.raw", clip[:], 0644)
}
