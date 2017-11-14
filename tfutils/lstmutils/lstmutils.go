// Package lstmutils contains some useful functions for processing trained LSTM models in the particular format used by aural2.
package lstmutils

import (
	"errors"
	"log"
	"os"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var logger = log.New(os.Stdout, "", log.Lshortfile)

const outputname = "/softmax/output"
const inputname = "/inputs"

// LoadGraph loads the TF graph from the bytes of an LSTM graphdef, and returns a
func LoadGraph(savedModel tf.SavedModel, scopeName string) (session *tf.Session, input, output tf.Output, statePlaceholders, stateFetches []tf.Output, stateFeeds map[tf.Output]*tf.Tensor, err error) {
	// read in the two lists of state OP names
	initialStateNames, finalStateNames, err := readStateNames(savedModel.Graph, scopeName)
	if err != nil {
		return
	}
	// init the placeholders, fetches, and feeds for state.
	stateFeeds = make(map[tf.Output]*tf.Tensor)                   // feeds is the map of placeholders to the tensors of initial state that should be fed into them.
	stateFetches = make([]tf.Output, len(finalStateNames))        // fetches are the outputs we pull on to get the tensors of final state.
	statePlaceholders = make([]tf.Output, len(initialStateNames)) // placeholders is a list of input placeholders. It contains the same elements as stateFeeds, but is needed becouse maps do not have a stable ordering.
	// Create a session for inference over graph.
	session, err = tf.NewSession(savedModel.Graph, nil)
	if err != nil {
		logger.Println(err)
		return
	}

	// zerosOP is an OP that returns a tensor of zeros of shape [1, `n`] where `n` is the size of the LSTM state.
	zerosOP := savedModel.Graph.Operation("zeros")
	if zerosOP == nil {
		err = errors.New("can't find zeros op")
		return
	}
	// pull on the OP to get the zerosTensor
	result, err := savedModel.Session.Run(map[tf.Output]*tf.Tensor{}, []tf.Output{zerosOP.Output(0)}, nil)
	if err != nil {
		logger.Println(err)
		return
	}
	zeroTensor := result[0]
	for i, name := range initialStateNames { // for each name in the initial state outputs,
		operation := savedModel.Graph.Operation(name) // Get the operation out of the graph.
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
		operation := savedModel.Graph.Operation(name)
		if operation == nil {
			err = errors.New("can't find op '" + name + "' in graph")
			return
		}
		stateFetches[i] = operation.Output(0) // put the output of the OP in the list of state outputs to be pulled on.
	}
	// We are now done with state, we just need to get the input and output.
	operation := savedModel.Graph.Operation(scopeName + inputname) // get the input placeholder
	if operation == nil {
		err = errors.New("can't find op '" + scopeName + outputname + "' in graph")
		return
	}
	input = operation.Output(0) // assign the OPs output to `input`.

	operation = savedModel.Graph.Operation(scopeName + outputname) // get the output OP
	if operation == nil {
		err = errors.New("can't find op '" + scopeName + outputname + "' in graph")
		return
	}
	output = operation.Output(0) // assign the OPs output to `output`.
	return
}

// readStateNames fishes out the names of the initial_state placeholders, and the final_state outputs.
func readStateNames(graph *tf.Graph, scopeName string) (initialStateNames, finalStateNames []string, err error) {
	// The initial_state_names OP is a constant which will give us a 1D tensor of strings.
	initialStateNamesOP := graph.Operation(scopeName + "/initial_state_names")
	if initialStateNamesOP == nil {
		err = errors.New("can't find op " + scopeName + "/initial_state_names in graph")
		return
	}
	finalStateNamesOP := graph.Operation(scopeName + "/final_state_names") // final_state_names will likewise give us a 1D tensor of strings.
	if finalStateNamesOP == nil {
		err = errors.New("can't find op " + scopeName + "/final_state_names in graph")
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

// MakeSeqInference returns a function that takes an output of a slice of mfccs of one clip, and returns a [][]float32 labels.
func MakeSeqInference(savedModel tf.SavedModel) (seqInference func(*tf.Tensor) (*tf.Tensor, error), err error) {
	sess, input, output, _, _, _, err := LoadGraph(savedModel, "seq_inference")
	if err != nil {
		return
	}
	seqInference = func(mfccs *tf.Tensor) (probs *tf.Tensor, err error) {
		results, err := sess.Run(map[tf.Output]*tf.Tensor{input: mfccs}, []tf.Output{output}, nil)
		if err != nil {
			return
		}
		probs = results[0]
		return
	}
	return
}

// MakeStepInference returns a function that takes an output of a slice of mfccs of one clip, and returns a [][]float32 labels.
func MakeStepInference(savedModel tf.SavedModel) (stepInference func(*tf.Tensor) ([]float32, error), err error) {
	sess, input, output, placeholders, fetches, feeds, err := LoadGraph(savedModel, "step_inference")
	if err != nil {
		return
	}
	fetches = append(fetches, output) // also pull on output
	stepInference = func(mfccTensor *tf.Tensor) (probs []float32, err error) {
		feeds[input] = mfccTensor // feed the input
		results, err := sess.Run(
			feeds,
			fetches,
			nil,
		)
		if err != nil {
			return
		}
		for i, ph := range placeholders {
			feeds[ph] = results[i]
		}
		probs = results[len(fetches)-1].Value().([][]float32)[0]
		return
	}
	return
}
