package main

import (
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strconv"

	"github.com/tensorflow/tensorflow/tensorflow/go/op"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/tfutils"
)

var logger = log.New(os.Stdout, "sc: ", log.Lshortfile)

//"evaluation/input", "evaluation/softmax/output", "initial_state_names", "final_state_names", "zeros"]
const outputname = "evaluation/softmax/output"
const inputname = "evaluation/input"

// loadGraph loads the graphDef file from the fs, and returns a tf.SavedModel
func loadGraph() (session *tf.Session, input, output tf.Output, statePlaceholders, stateFetches []tf.Output, stateFeeds map[tf.Output]*tf.Tensor, err error) {
	// read the serialised graphDef file.
	graphBytes, err := ioutil.ReadFile("models/cmd_rnn.pb")
	if err != nil {
		logger.Println(err)
		return
	}
	graph := tf.NewGraph() // make a new graph,
	// and import the graphdef into it.
	if err = graph.Import(graphBytes, ""); err != nil {
		logger.Println(err)
		return
	} // Do we have a non nil graph now?
	if graph == nil {
		err = errors.New("graph is nil")
		return
	}
	// read in the two lists of state OP names
	initialStateNames, finalStateNames, err := readStateNames(graph)
	if err != nil {
		return
	}
	// init the placeholders, fetches, and feeds for state.
	stateFeeds = make(map[tf.Output]*tf.Tensor)                   // feeds is the map of placeholders to the tensors of initial state that should be fed into them.
	stateFetches = make([]tf.Output, len(finalStateNames))        // fetches are the outputs we pull on to get the tensors of final state.
	statePlaceholders = make([]tf.Output, len(initialStateNames)) // placeholders is a list of input placeholders. It contains the same elements as stateFeeds, but is needed becouse maps do not have a stable ordering.
	// Create a session for inference over graph.
	session, err = tf.NewSession(graph, nil)
	if err != nil {
		logger.Println(err)
		return
	}

	// zerosOP is an OP that returns a tensor of zeros of shape [1, `n`] where `n` is the size of the LSTM state.
	zerosOP := graph.Operation("zeros")
	if zerosOP == nil {
		logger.Println(err)
		return
	}
	// pull on the OP to get the zerosTensor
	result, err := session.Run(map[tf.Output]*tf.Tensor{}, []tf.Output{zerosOP.Output(0)}, nil)
	if err != nil {
		logger.Println(err)
		return
	}
	zeroTensor := result[0]
	for i, name := range initialStateNames { // for each name in the initial state outputs,
		operation := graph.Operation(name[:len(name)-2]) // Get the operation out of the graph.
		// The names are names of outputs, they all have ":0" on the end. Go TF wants names of operations, not outputs.
		// Therefore, we remove the last two chars of the names. This is hacky, TODO: do better.
		if operation == nil {
			err = errors.New("can't find op '" + name + "' in graph")
			return
		}
		statePlaceholders[i] = operation.Output(0) // Put the output in the list of placeholders,
		if err != nil {
			logger.Fatalln(err)
		}
		stateFeeds[statePlaceholders[i]] = zeroTensor // and populate the feeds with the zeroTensor.
	}
	// Now we do somthing similar for the state outputs
	for i, name := range finalStateNames { // for each name in the list of final state names,
		operation := graph.Operation(name[:len(name)-2]) // again, as Go TF can't access outputs directly, we strip ":0" from the end. This is, of course, inelegant.
		if operation == nil {
			err = errors.New("can't find op '" + name + "' in graph")
			return
		}
		stateFetches[i] = operation.Output(0) // put the output of the OP in the list of state outputs to be pulled on.
	}
	// We are now done with state, we just need to get input and outputs.
	operation := graph.Operation(inputname) // get the input placeholder
	if operation == nil {
		err = errors.New("can't find op '" + inputname + "' in graph")
		return
	}
	input = operation.Output(0) // assign the OPs output to `input`.

	operation = graph.Operation(outputname) // get the output OP
	if operation == nil {
		err = errors.New("can't find op '" + outputname + "' in graph")
		return
	}
	output = operation.Output(0) // assign the OPs output to `output`.
	return
}

// readStateNames fishes out the names of the initial_state placeholders, and the final_state outputs.
func readStateNames(graph *tf.Graph) (initialStateNames, finalStateNames []string, err error) {
	// The initial_state_names OP is a constant which will give us a 1D tensor of strings.
	initialStateNamesOP := graph.Operation("initial_state_names")
	if initialStateNamesOP == nil {
		err = errors.New("can't find op initial_state_names in graph")
		return
	}
	finalStateNamesOP := graph.Operation("final_state_names") // final_state_names will likewise give us a 1D tensor of strings.
	if finalStateNamesOP == nil {
		err = errors.New("can't find op final_state_names in graph")
		return
	}
	session, err := tf.NewSession(graph, nil) // let us make a new TF session just for reading the state OP names.
	if err != nil {
		logger.Println(err)
		return
	}
	// now run the two OPs in the session. No feeds are needed.
	result, err := session.Run(map[tf.Output]*tf.Tensor{}, []tf.Output{initialStateNamesOP.Output(0), finalStateNamesOP.Output(0)}, nil)
	if err != nil {
		return
	}
	// take the two outputs and assert them both to be []string.
	initialStateNames = result[0].Value().([]string)
	finalStateNames = result[1].Value().([]string)
	// sanity check. There is a one to one relationship between the elements of the two lists, so they had better be the same length.
	if len(initialStateNames) != len(finalStateNames) {
		err = errors.New("len(initialStateNames) != len(finalStateNames)")
		return
	}
	return
}

func oneHotEncode(char string, vocab map[string]int) (oneHot []int32) {
	oneHot = make([]int32, len(vocab))
	oneHot[vocab[char]] = 1
	return
}

// argmax returns the index of the largest elements of the list.
func argmax(probs []float32) (index int32) {
	var max float32
	for i, val := range probs {
		if val > max {
			max = val
			index = int32(i)
		}
	}
	return
}

func computeMFCC() (mfcc [][]float32, err error) {
	s := op.NewScope()
	bytesPH, pcm := tfutils.ParseRawBytesToPCM(s)
	mfccOP, sampleRatePH := tfutils.ComputeMFCC(s.SubScope("spectrogram"), pcm)
	sampleRateTensor, err := tf.NewTensor(int32(libaural2.SampleRate))
	if err != nil {
		logger.Fatalln(err)
	}
	graph, err := s.Finalize() // finalize the scope to get the graph
	if err != nil {
		logger.Fatalln(err)
	}
	sess, err := tf.NewSession(graph, nil) // start a new TF session
	if err != nil {
		logger.Fatalln(err)
	}
	rawBytes, err := ioutil.ReadFile("label_serve/audio/QAKZAMTDHFVNZ6QLPIBFYYSRQN2Y43IHU76ZQN6L6C22OWLR66RQ====.raw")
	if err != nil {
		return
	}
	if len(rawBytes) != libaural2.AudioClipLen {
		err = errors.New("Got " + strconv.Itoa(len(rawBytes)) + " bytes, expected" + strconv.Itoa(libaural2.AudioClipLen))
		return
	}
	clipTensor, err := tf.NewTensor(string(rawBytes)) // create a string tensor from the input bytes
	if err != nil {
		logger.Println(err)
		return
	}
	result, err := sess.Run(map[tf.Output]*tf.Tensor{bytesPH: clipTensor, sampleRatePH: sampleRateTensor}, []tf.Output{mfccOP}, nil)
	if err != nil {
		logger.Println(err)
		return
	}

	mfcc = result[0].Value().([][]float32)
	return
}

func main() {
	sess, input, output, placeholders, fetches, feeds, err := loadGraph() // load in the graph
	// sess: the TF session
	// input: the placeholder to be populated with the input data each iteration
	// output: the output from which can be pulled on to give the result each iteration
	// placeholders: an ordered list of the state placeholders
	// fetches: a list of outputs to be pulled on to get the final state each iteration.
	// feeds: a map of outputs to the tensors of initial state they are to be populated with. Currently they are populated with zero tensors of the appropriate dimensions. Later we will replace them with the final state tensors each iteration.
	// err: did anything go wrong?
	if err != nil {
		logger.Fatalln(err)
	}
	fetches = append(fetches, output) // also pull on output
	mfccs, err := computeMFCC()
	if err != nil {
		logger.Fatalln(err)
	}
	// Run
	for _, mfcc := range mfccs {
		inputTensor, err := tf.NewTensor([][][]float32{[][]float32{mfcc}})
		if err != nil {
			logger.Fatalln(err)
		}
		feeds[input] = inputTensor // feed the input
		results, err := sess.Run(
			feeds,
			fetches,
			nil,
		)
		if err != nil {
			logger.Fatalln(err)
		}
		probs := results[len(fetches)-1].Value().([][]float32)[0]
		//logger.Println(probs)
		cmd := libaural2.Cmd(argmax(probs))
		for i, ph := range placeholders {
			feeds[ph] = results[i]
		}
		_ = cmd
		fmt.Println(cmd)
	}
	println("")
}
