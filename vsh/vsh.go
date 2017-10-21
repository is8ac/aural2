package vsh

import (
	"errors"
	"io"
	"log"
	"os"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/tfutils"
	"github.ibm.com/Blue-Horizon/aural2/tfutils/lstmutils"
)

const outputname = "evaluation/softmax/output"
const inputname = "evaluation/input"

var logger = log.New(os.Stdout, "vsh: ", log.Lshortfile)

// Argmax returns the index of the largest elements of the list.
func Argmax(probs []float32) (cmd libaural2.Cmd, prob float32) {
	for i, val := range probs {
		if val > prob {
			prob = val
			cmd = libaural2.Cmd(int32(i))
		}
	}
	return
}

func makeComputeMFCCgraph() (computeMFCC func([]byte) (*tf.Tensor, error), err error) {
	s := op.NewScope()
	rawBytesPH, pcm := tfutils.ParseRawBytesToPCM(s)
	mfccOP, sampleRatePH := tfutils.ComputeMFCC(s.SubScope("spectrogram"), pcm)
	dim := op.Const(s, int32(0))
	expanded := op.ExpandDims(s, mfccOP, dim)
	sampleRateTensor, err := tf.NewTensor(int32(libaural2.SampleRate))
	if err != nil {
		return
	}
	graph, err := s.Finalize() // finalize the scope to get the graph
	if err != nil {
		return
	}
	sess, err := tf.NewSession(graph, nil) // start a new TF session
	if err != nil {
		return
	}
	computeMFCC = func(rawBytes []byte) (mfccTensor *tf.Tensor, err error) {
		rawBytesTensor, err := tf.NewTensor(string(rawBytes[:])) // create a string tensor from the input bytes
		if err != nil {
			return
		}
		result, err := sess.Run(map[tf.Output]*tf.Tensor{rawBytesPH: rawBytesTensor, sampleRatePH: sampleRateTensor}, []tf.Output{expanded}, nil)
		if err != nil {
			return
		}
		//logger.Println(result[0].Shape())
		//logger.Println(result[0].Value())
		if result[0].Shape()[0] != 1 {
			err = errors.New("bad shape")
			return
		}
		mfccTensor = result[0]
		return
	}
	return
}

// Init takes a reader of raw audio, and returns a chan of outputs.
func Init(reader io.Reader, graphBytes []byte) (result chan []float32, err error) {
	result = make(chan []float32)
	stepInference, err := lstmutils.MakeStepInference(graphBytes)
	if err != nil {
		return
	}
	computeMFCC, err := makeComputeMFCCgraph()
	if err != nil {
		return
	}
	buff := make([]byte, libaural2.StrideWidth*2)
	go func() {
		for {
			n, err := reader.Read(buff)
			if err != nil {
				logger.Println(err)
				close(result)
				return
			}
			if n != libaural2.StrideWidth*2 {
				//logger.Println(n)
			}
			mfccTensor, err := computeMFCC(buff)
			if err != nil {
				logger.Println(err)
				close(result)
				return
			}
			probs, err := stepInference(mfccTensor)
			if err != nil {
				logger.Println(err)
				close(result)
				return
			}
			result <- probs
		}
		close(result)
	}()
	return
}
