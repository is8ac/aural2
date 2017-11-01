package tftrain

import (
	"io/ioutil"
	"os"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func TestLoad(t *testing.T) {
	graph, err := loadTrainGraph("models/linear_train.pb")
	if err != nil {
		t.Fatal(err)
	}
	_ = graph
}

func getTrainingData(int) (inputTensor *tf.Tensor, targetTensor *tf.Tensor, err error) {
	inputTensor, err = tf.NewTensor([]float32{1, 2, 3, 4})
	if err != nil {
		return
	}
	targetTensor, err = tf.NewTensor([]float32{0, -1, -2, -3})
	if err != nil {
		return
	}
	return
}

func TestTrainLinear(t *testing.T) {
	graph, err := loadTrainGraph("models/linear_train.pb")
	if err != nil {
		t.Fatal(err)
	}
	frozen, err := BatchTrain(graph, 1000, "x", "y", "train", "init", "loss", []string{"output"}, printLoss, getTrainingData)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Create("linear_frozen.pb")
	if err != nil {
		t.Fatal(err)
	}
	frozen.WriteTo(f)
}

func makeGetLSTMtrainingData() (getTrainingData func(int) (*tf.Tensor, *tf.Tensor, error), err error) {
	graphBytes, err := ioutil.ReadFile("lstm_trainingdata.pb")
	if err != nil {
		return
	}
	graph := tf.NewGraph() // make a new graph,
	// and import the graphdef into it.
	if err = graph.Import(graphBytes, ""); err != nil {
		return
	}
	inputOP, err := getOP(graph, "inputs/Identity")
	if err != nil {
		return
	}
	targetOP, err := getOP(graph, "outputs/Identity")
	if err != nil {
		return
	}
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return
	}
	getTrainingData = func(i int) (inputTensor *tf.Tensor, targetTensor *tf.Tensor, err error) {
		result, err := session.Run(
			map[tf.Output]*tf.Tensor{},
			[]tf.Output{
				inputOP.Output(0),
				targetOP.Output(0),
			},
			nil,
		)
		if err != nil {
			return
		}
		inputTensor = result[0]
		targetTensor = result[1]
		return
	}
	return
}

func printLoss(i int, loss float32) {
	//logger.Println(i, loss)
}

func TestBatchTrainLstm(t *testing.T) {
	graph, err := loadTrainGraph("models/lstm_train.pb")
	if err != nil {
		t.Fatal(err)
	}
	getLSTMtrainingData, err := makeGetLSTMtrainingData()
	if err != nil {
		t.Fatal(err)
	}
	requiredOutputs := []string{
		"step_inference/softmax/output",
		"step_inference/initial_state_names",
		"step_inference/final_state_names",
		"zeros",
	}
	frozen, err := BatchTrain(graph, 100, "training/inputs", "training/targets", "training/Adam", "init", "training/loss_monitor/Exp", requiredOutputs, printLoss, getLSTMtrainingData)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Create("lstm_frozen.pb")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = frozen.WriteTo(f); err != nil {
		t.Fatal(err)
	}
}

func TestStepTrainLinear(t *testing.T) {
	graph, err := loadTrainGraph("models/linear_train.pb")
	if err != nil {
		t.Fatal(err)
	}
	oSess, err := NewOnlineSess(graph, "x", "y", "train", "init", "loss", "output", "x", []string{"output"})
	if err != nil {
		t.Fatal(err)
	}
	inputTensor, targetTensor, _ := getTrainingData(0)
	loss1, err := oSess.Train(inputTensor, targetTensor)
	if err != nil {
		t.Fail()
	}
	inputTensor, targetTensor, _ = getTrainingData(1)
	loss2, err := oSess.Train(inputTensor, targetTensor)
	if err != nil {
		t.Fail()
	}
	if !(loss2 < loss1) {
		t.Fatal("loss did not decrease")
	}
	output, err := oSess.Infer(inputTensor)
	if err != nil {
		t.Fatal(err)
	}
	if output.Shape()[0] != 4 {
		t.Fatal("wrong shape")
	}
	weightVar := graph.Operation("weight")
	results, err := oSess.Run(map[tf.Output]*tf.Tensor{}, []tf.Output{weightVar.Output(0)})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Fatal("wrong len")
	}
}

func TestStepTrainLSTM(t *testing.T) {
	graph, err := loadTrainGraph("models/lstm_train.pb")
	if err != nil {
		t.Fatal(err)
	}
	getTrainingData, err := makeGetLSTMtrainingData()
	if err != nil {
		t.Fatal(err)
	}
	requiredOutputs := []string{
		"step_inference/softmax/output",
		"step_inference/initial_state_names",
		"step_inference/final_state_names",
		"zeros",
	}
	oSess, err := NewOnlineSess(graph, "training/inputs", "training/targets", "training/Adam", "init", "training/loss_monitor/truediv", "step_inference/softmax/output", "step_inference/inputs", requiredOutputs)
	if err != nil {
		t.Fatal(err)
	}
	inputTensor, targetTensor, _ := getTrainingData(0)
	loss1, err := oSess.Train(inputTensor, targetTensor)
	if err != nil {
		t.Fail()
	}
	var loss2 float32
	for i := 0; i < 30; i++ {
		inputTensor, targetTensor, _ = getTrainingData(1)
		loss2, err = oSess.Train(inputTensor, targetTensor)
		if err != nil {
			t.Fail()
		}
	}
	if !(loss2 < loss1) {
		t.Fatal("loss did not decrease")
	}
	inputTensor, err = tf.NewTensor([][][]float32{[][]float32{[]float32{0.3, 0.7, 0.8, 0.2, 0.5, 0.1, 0.6, 0.2, 0.5, 0.1, 0.6, 0.1, 0.6}}})
	if err != nil {
		t.Fatal(err)
	}
	logger.Println(inputTensor.Shape())
	output, err := oSess.Infer(inputTensor)
	if err != nil {
		t.Fatal(err)
	}
	logger.Println(output.Shape())
	if len(output.Shape()) != 2 {
		t.Fatal("wrong dim")
	}
	weightVar := graph.Operation("softmax/softmax_w")
	if weightVar == nil {
		t.Fatal("can't find softmax/softmax_w")
	}
	results, err := oSess.Run(map[tf.Output]*tf.Tensor{}, []tf.Output{weightVar.Output(0)})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Fatal("wrong len")
	}
	logger.Println(results[0].Value())
}
