package tfutils

import (
	"log"
	"os"

	"sync"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

var logger = log.New(os.Stdout, "tfutils: ", log.Lshortfile)

// ReadWaveToPCM returns a placeholder for a filepath to an int16le wav file, and an output for float PCM
func ReadWaveToPCM(s *op.Scope) (filePath, pcm tf.Output) {
	filePath = op.Placeholder(s.SubScope("file_path"), tf.String)                    // placeholder to be filled with the file path at run time
	wavBytes := op.ReadFile(s.SubScope("read_file"), filePath)                       // read in the wav file
	multiChanPCM, _ := op.DecodeWav(s.SubScope("decode_wav"), wavBytes)              // decode the wav file into PCM. The first dimension is time, the second is channels.
	channels := op.Unpack(s.SubScope("channels"), multiChanPCM, 1, op.UnpackAxis(1)) // unpack the channels.
	pcm = channels[0]                                                                // we only want the first channel
	return
}

// ParseWavBytesToPCM returns a placeholder for []byte of an int16le wav file, and an output for float PCM
func ParseWavBytesToPCM(s *op.Scope) (wavBytes, pcm tf.Output) {
	wavBytes = op.Placeholder(s.SubScope("file_path"), tf.String)                    // placeholder to be filled with the bytes of a wav file
	multiChanPCM, _ := op.DecodeWav(s.SubScope("decode_wav"), wavBytes)              // decode the wav file into PCM. The first dimension is time, the second is channels.
	channels := op.Unpack(s.SubScope("channels"), multiChanPCM, 1, op.UnpackAxis(1)) // unpack the channels.
	pcm = channels[0]                                                                // we only want the first channel
	return
}

// ComputeMFCC compute the Mel-frequency cepstrum coefficients of the PCM audio
func ComputeMFCC(s *op.Scope, pcm tf.Output) (mfcc, sampleRatePH tf.Output) {
	dim := op.Const(s.SubScope("dim"), int32(1))
	expanded := op.ExpandDims(s.SubScope("expand_dims"), pcm, dim)     // AudioSpectrogram wants 2D input, so add another dimension to make it happy.
	sampleRatePH = op.Placeholder(s.SubScope("sample_rate"), tf.Int32) // MFCC need to know the sample rate.
	spectrogram := op.AudioSpectrogram(s.SubScope("spectrogram"),      // Compute the spectrogram
		expanded,
		1024, // window size
		1024, // stride size
		op.AudioSpectrogramMagnitudeSquared(true), // square the magnitude
	)
	mfccs := op.Mfcc(s.SubScope("mfcc"), spectrogram, sampleRatePH)         // compute the mfcc
	mfcc = op.Unpack(s.SubScope("channels"), mfccs, 1, op.UnpackAxis(0))[0] // remove the unnecessary dimension
	return
}

// ComputeSpectrogram computes the spectrogram of the given audio
func ComputeSpectrogram(s *op.Scope, pcm tf.Output, freqMin, freqBuf int) (slice tf.Output) {
	dim := op.Const(s.SubScope("dim"), int32(1))
	expanded := op.ExpandDims(s.SubScope("expand_dims"), pcm, dim) // again, make AudioSpectrogram happy.
	spectrograms := op.AudioSpectrogram(s.SubScope("spectrogram"), expanded, 1024, 1024)
	invertedSpectrogram := op.Unpack(s.SubScope("channels"), spectrograms, 1, op.UnpackAxis(0))[0] // and remove the unnecessary dimension
	reverse := op.Reverse(s.SubScope("reverse"), invertedSpectrogram, op.Const(s.SubScope("reverse_dim"), []bool{false, true}))

	shape := op.Shape(s.SubScope("shape"), reverse)                                // read the shape of the tensor
	change := op.Const(s.SubScope("reduce"), []int32{0, int32(freqBuf + freqMin)}) // how much to reduce the top of freq range by.
	begin := op.Const(s.SubScope("slice_begin"), []int32{0, int32(freqBuf)})       // begin at 0 time and 0 space
	size := op.Sub(s.SubScope("size"), shape, change)                              // subtract the change from the shape to get the size.
	slice = op.Slice(s.SubScope("slice"), reverse, begin, size)                    // slice out the desired freq range.
	return
}

// RenderImage takes an operation of shape [time, freq], and returns an operation of the bytes in JPEG image.
func RenderImage(s *op.Scope, values tf.Output) (jpegBytes tf.Output) {
	max := op.Max(s.SubScope("max"), values, op.Const(s.SubScope("reduction_indices"), []int32{0})) // compute the max
	normalized := op.Div(s.SubScope("normalise"), values, max)                                      // normalise to betwene 0 and 1
	permutations := op.Const(s.SubScope("permutations"), []int32{1, 0})                             // the dims to be permuted, dim 0 is swiched with dim 1
	transposed := op.Transpose(s.SubScope("rotate"), normalized, permutations)                      // switch vertical and horizontal axis
	ones := op.OnesLike(s.SubScope("ones"), transposed)                                             // make some ones of same shape as the input
	hsv := op.Pack(s.SubScope("pack"), []tf.Output{transposed, ones, ones}, op.PackAxis(2))         // stack the values and the ones up into hsv. Hue is the normalized values, saturation and value are 1
	rgb := op.HSVToRGB(s.SubScope("rgb"), hsv)                                                      // convert the hsv values into rgb floats from 0 to 1
	rescaled := op.Mul(s.SubScope("rescale"), rgb, op.Const(s.SubScope("256"), float32(256)))       // convert the floats from 0-1, to 0-255
	int8RGB := op.Cast(s.SubScope("cast"), rescaled, tf.Uint8)                                      // cast them to uint8
	jpegBytes = op.EncodeJpeg(s.SubScope("jpeg"), int8RGB)                                          // encode to jpeg
	return
}

// BytesToBytes takes a scope, a placeholder for a []byte, and an output of []byte and returns a `func([]byte)[]byte`.
// feeds may be an empty map, or it may be populated with whatever special feeds your graph needs.
func BytesToBytes(s *op.Scope, inputPH, outputOP tf.Output, feeds map[tf.Output]*tf.Tensor) (conversionFunc func([]byte) ([]byte, error)) {
	graph, err := s.Finalize() // finalize the scope to get the graph
	if err != nil {
		logger.Println(err)
		return
	}
	sess, err := tf.NewSession(graph, nil) // start a new TF session
	if err != nil {
		logger.Println(err)
		return
	}
	var mutex = &sync.Mutex{}
	conversionFunc = func(inputBytes []byte) (outputBytes []byte, err error) { // make a function to convert a []byte into another []byte using whatever TF graph was given
		inputTensor, err := tf.NewTensor(string(inputBytes)) // create a string tensor from the input bytes
		if err != nil {
			return
		}
		mutex.Lock()
		feeds[inputPH] = inputTensor // add inputPH and inputTensor to feeds.
		result, err := sess.Run(feeds, []tf.Output{outputOP}, nil)
		if err != nil {
			logger.Println(err)
			return
		}
		mutex.Unlock()
		outputBytes = []byte(result[0].Value().(string))
		return
	}
	return
}
