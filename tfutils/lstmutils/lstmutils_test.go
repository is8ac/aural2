package lstmutils

import (
	"io/ioutil"
	"os"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/tftrain"
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

func TestRenderLSTMstate(t *testing.T) {
	rawBytes, err := ioutil.ReadFile("testaudio2.raw")
	if err != nil {
		t.Fatal(err)
	}
	audioClip := &libaural2.AudioClip{}
	copy(audioClip[:], rawBytes) // convert the slice of bytes to an array of bytes.
	graphBytes, err := ioutil.ReadFile("intent.pb")
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
	requiredOutputs := []string{
		"step_inference/softmax/output",
		"step_inference/initial_state_names",
		"step_inference/final_state_names",
		"step_inference/loss_monitor/count",
		"seq_inference/loss_monitor/count",
		"seq_inference/loss_monitor/sum_mean_loss",
		"step_inference/loss_monitor/sum_mean_loss",
		"zeros",
	}

	oSess, err := tftrain.NewOnlineSess(graph,
		"training/inputs",  // placeholder for batch training inputs
		"training/targets", // placeholder for batch training targets
		"training/Adam",    // training operation
		"init",             // OP to initalise the variables
		"training/loss_monitor/div",    // the loss of the graph when training
		"seq_inference/softmax/output", // output for live inference
		"seq_inference/inputs",         // input for live inference
		requiredOutputs,                // any other ops which need to be preserved when freezing
	)
	if err != nil {
		t.Fatal(err)
	}
	renderLSTMstate, err := makeRenderLSTMstate(oSess)
	if err != nil {
		t.Fatal(err)
	}
	mfccTensor, err := audioClipToMFCCtensor(audioClip)
	if err != nil {
		t.Fatal(err)
	}
	logger.Println(mfccTensor.Shape())
	imageBytes, err := renderLSTMstate(mfccTensor)
	if err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile("states.jpeg", imageBytes, 0644); err != nil {
		t.Fail()
	}
}
