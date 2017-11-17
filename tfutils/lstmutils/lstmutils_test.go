package lstmutils

import (
	"io/ioutil"
	"os"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/tfutils"
)

func TestLoadGraphStep(t *testing.T) {
	reader, err := os.Open("testaudio.raw")
	if err != nil {
		t.Fatal(err)
	}
	_ = reader
	graphBytes, err := ioutil.ReadFile("cmd_rnn.pb")
	if err != nil {
		t.Fatal(err)
	}
	graph := tf.NewGraph()
	if err := graph.Import(graphBytes, ""); err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, _, _, _, _, err = LoadGraph(graph, sess, "step_inference")
	if err != nil {
		t.Fatal(err)
	}
}

func TestMakeSeqInference(t *testing.T) {
	rawBytes, err := ioutil.ReadFile("testaudio.raw")
	if err != nil {
		t.Fatal(err)
	}
	audioClip := &libaural2.AudioClip{}
	copy(audioClip[:], rawBytes) // convert the slice of bytes to an array of bytes.
	graphBytes, err := ioutil.ReadFile("cmd_rnn.pb")
	if err != nil {
		t.Fatal(err)
	}
	audioClipToMFCCtensor, err := tfutils.MakeAudioClipToMFCCtensor()
	if err != nil {
		t.Fatal(err)
	}
	graph := tf.NewGraph()
	if err := graph.Import(graphBytes, ""); err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}

	seqInference, err := MakeSeqInference(tf.SavedModel{Graph: graph, Session: sess})
	if err != nil {
		t.Fatal(err)
	}
	probsTensorToImage, err := tfutils.MakeProbsTensorToImage()
	if err != nil {
		t.Fatal(err)
	}
	mfccTensor, err := audioClipToMFCCtensor(audioClip)
	if err != nil {
		t.Fatal(err)
	}
	probs, err := seqInference(mfccTensor)
	if err != nil {
		t.Fatal(err)
	}
	imageBytes, err := probsTensorToImage(probs)
	if err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile("probs.jpeg", imageBytes, 0644); err != nil {
		t.Fail()
	}
}
