package vsh

import (
	"bytes"
	"errors"
	"io"
	"log"
	"net/http"
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

type ringBuf struct {
	data [(libaural2.StridesPerClip + 1) * libaural2.StrideWidth * 2]byte
	end  int
}

func makeRing() (rb *ringBuf) {
	rb = new(ringBuf)
	return
}

func (rb *ringBuf) write(stride []byte) {
	copy(rb.data[rb.end*libaural2.StrideWidth*2:(rb.end)*libaural2.StrideWidth*2+libaural2.StrideWidth*2], stride)
	rb.end++
	if rb.end > libaural2.StridesPerClip {
		rb.end = 0
	}
}

func (rb *ringBuf) dump() (clip *libaural2.AudioClip) {
	whole := append(rb.data[rb.end*libaural2.StrideWidth*2:], rb.data[:rb.end*libaural2.StrideWidth*2]...)
	clip = &libaural2.AudioClip{}
	copy(clip[:], whole)
	return
}

// Argmax returns the index of the largest elements of the list.
func Argmax(probs []float32) (state libaural2.State, prob float32) {
	for i, val := range probs {
		if val > prob {
			prob = val
			state = libaural2.State(int32(i))
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

func uploadClip(clip *libaural2.AudioClip) (err error) {
	_, err = http.Post("http://localhost:48125/sample/upload", "application/octet-stream", bytes.NewReader(clip[:]))
	return
}

// Init takes a reader of raw audio, and returns a chan of outputs.
func Init(reader io.Reader, graphs map[libaural2.VocabName][]byte) (result chan map[libaural2.VocabName][]float32, dump func() *libaural2.AudioClip, err error) {
	rb := makeRing()
	dump = rb.dump
	result = make(chan map[libaural2.VocabName][]float32)
	stepInferenceFuncs := map[libaural2.VocabName]func(*tf.Tensor) ([]float32, error){}
	for vocabName, graphBytes := range graphs {
		stepInferenceFuncs[vocabName], err = lstmutils.MakeStepInference(graphBytes)
		if err != nil {
			return
		}
	}
	computeMFCC, err := makeComputeMFCCgraph()
	if err != nil {
		return
	}
	buf := make([]byte, libaural2.StrideWidth*2)
	go func() {
		for {
			n, err := reader.Read(buf)
			if err != nil {
				logger.Println(err)
				close(result)
				return
			}
			rb.write(buf)
			//logger.Println(n)
			if n != libaural2.StrideWidth*2 {
				//logger.Println(n)
			}
			mfccTensor, err := computeMFCC(buf)
			if err != nil {
				logger.Println(err)
				close(result)
				return
			}
			probsMap := map[libaural2.VocabName][]float32{}
			for vocabName, stepInference := range stepInferenceFuncs {
				probsMap[vocabName], err = stepInference(mfccTensor)
				if err != nil {
					logger.Println(err)
					close(result)
					return
				}
			}
			result <- probsMap
		}
		close(result)
	}()
	return
}
