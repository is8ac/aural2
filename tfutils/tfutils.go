// Package tfutils provides various functions usefull when constructing TensorFlow compute graphs.
package tfutils

import (
	"errors"
	"log"
	"math/rand"
	"os"

	"sync"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"

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

// ParseRawBytesToPCM returns a placeholder for []byte of an int16le raw file, and an output for float PCM
func ParseRawBytesToPCM(s *op.Scope) (rawBytes, pcm tf.Output) {
	rawBytes = op.Placeholder(s.SubScope("raw_bytes"), tf.String)
	int16PCM := op.DecodeRaw(s.SubScope("decode_raw"), rawBytes, tf.Int16)
	floats := op.Cast(s.SubScope("cast"), int16PCM, tf.Float)
	pcm = op.Div(s.SubScope("rescale"), floats, op.Const(s.SubScope("16pow"), float32(65536)))
	return
}

// EncodeWav encodes pcm to wav file
func EncodeWav(s *op.Scope, pcmOutput tf.Output) (wavBytesOutput, sampleRatePH tf.Output) {
	dim := op.Const(s.SubScope("dim"), int32(1))
	sampleRatePH = op.Placeholder(s.SubScope("sample_rate"), tf.Int32)
	expanded := op.ExpandDims(s.SubScope("expand_dims"), pcmOutput, dim)
	wavBytesOutput = op.EncodeWav(s.SubScope("encode_wav"), expanded, sampleRatePH)
	return
}

// MakeCleanWav returns a function which takes the bytes of a wav file, converts to PCM, checks that it is good and reconstructs a wav file from the PCM.
// If the input is malformed, it will return an error. The output may be slighty different from the input. Use the output.
func MakeCleanWav(sampleRate int) (cleanWav func([]byte) ([]byte, error), err error) {
	s := op.NewScope()
	dim := op.Const(s.SubScope("dim"), int32(1))
	inputPH := op.Placeholder(s.SubScope("wav_bytes"), tf.String, op.PlaceholderShape(tf.ScalarShape()))
	sampleRatePH := op.Placeholder(s.SubScope("sample_rate"), tf.Int32)
	pcmChansOP, sampleRateOP := op.DecodeWav(s.SubScope("decode_wav"), inputPH, op.DecodeWavDesiredChannels(1))
	_ = sampleRateOP
	channels := op.Unpack(s.SubScope("channels"), pcmChansOP, 1, op.UnpackAxis(1)) // unpack the channels.
	shape := op.Shape(s.SubScope("shape"), channels[0])
	expanded := op.ExpandDims(s.SubScope("expand_dims"), channels[0], dim) // again, make AudioSpectrogram happy.
	wavBytes := op.EncodeWav(s.SubScope("encode_wav"), expanded, sampleRatePH)
	graph, err := s.Finalize()
	if err != nil {
		logger.Println(err)
		return
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		logger.Println(err)
		return
	}

	cleanWav = func(inputBytes []byte) (outputBytes []byte, err error) {
		inputTensor, err := tf.NewTensor(string(inputBytes)) // create a string tensor from the input bytes
		if err != nil {
			return
		}
		sampleRateTensor, err := tf.NewTensor(int32(sampleRate)) // create a string tensor from the input bytes
		if err != nil {
			return
		}
		feeds := map[tf.Output]*tf.Tensor{
			inputPH:      inputTensor,
			sampleRatePH: sampleRateTensor,
		}
		result, err := sess.Run(feeds, []tf.Output{wavBytes, sampleRateOP, shape}, nil)
		if err != nil {
			return
		}
		shape := result[2].Value().([]int32)
		if len(shape) != 1 {
			err = errors.New("bad shape")
			return
		}
		actualSampleRate := result[1].Value().(int32)
		if actualSampleRate != int32(sampleRate) {
			err = errors.New("wrong sample rate")
			return
		}
		outputBytes = []byte(result[0].Value().(string))
		return
	}
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
		int64(libaural2.StrideWidth),              // stride size
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
	spectrograms := op.AudioSpectrogram(s.SubScope("spectrogram"), expanded, 1024, int64(libaural2.StrideWidth))
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
func BytesToBytes(s *op.Scope, inputPH, outputOP tf.Output, feeds map[tf.Output]*tf.Tensor) (conversionFunc func([]byte) ([]byte, error), err error) {
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
		mutex.Lock()                 // tf.Session is thread safe, the mutex is only needed to keep the feeds safe.
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

// SplitInputSeqs splits long seqs into shorter seqs for training.
func SplitInputSeqs(inputSet [][][]float32) (splitSet [][][]float32) {
	numSubSeqs := libaural2.StridesPerClip / libaural2.SeqLen
	logger.Println(numSubSeqs)
	splitSet = make([][][]float32, len(inputSet)*numSubSeqs)
	for i, seq := range inputSet {
		for part := 0; part < numSubSeqs; part++ {
			logger.Println(i*numSubSeqs+part, part*libaural2.SeqLen, (part+1)*libaural2.SeqLen)
			splitSet[i*numSubSeqs+part] = seq[part*libaural2.SeqLen : (part+1)*libaural2.SeqLen]
		}
	}
	return
}

// EmbedTrainingData returns a GrapDef with the inputs and outputs embeded
// inputs must be of shape [len, libaural2.StridesPerClip, libaural2.InputSize]
// outputs must be of shape [len, libaural2.StridesPerClip]
// where len is the same for inputs, outputs, and ids.
func EmbedTrainingData(inputs [][][]float32, outputs [][libaural2.StridesPerClip]int32, ids []libaural2.ClipID, numSubSeqs int, batchSize int) (graph *tf.Graph, err error) {
	if len(inputs) != len(outputs) || len(ids) != len(inputs) {
		err = errors.New("input, output, or ids len do not match")
		return
	}
	if len(inputs) == 0 {
		err = errors.New("must be given more then 0 clips")
		return
	}
	if len(inputs[0]) != libaural2.StridesPerClip {
		err = errors.New("input has wrong StridesPerClip")
		return
	}
	if len(outputs[0]) != libaural2.StridesPerClip {
		err = errors.New("output has wrong StridesPerClip")
		return
	}
	if len(inputs[0][0]) != libaural2.InputSize {
		err = errors.New("bad InputSize")
		return
	}
	s := op.NewScope()
	is := s.SubScope("inputs")
	os := s.SubScope("outputs")
	inputsConst := op.Const(is, inputs)
	outputsConst := op.Const(os, outputs)
	inputsSubSeqs := make([]tf.Output, numSubSeqs)
	outputsSubSeqs := make([]tf.Output, numSubSeqs)
	for i := 0; i < numSubSeqs; i++ {
		start := rand.Intn(libaural2.StridesPerClip - libaural2.SeqLen)

		inputsBegin := op.Const(is.SubScope("begin"), []int32{0, int32(start), 0})
		inputsSize := op.Const(is.SubScope("size"), []int32{int32(len(inputs)), int32(libaural2.SeqLen), int32(libaural2.InputSize)})
		inputSubSeq := op.Slice(is.SubScope("slice"), inputsConst, inputsBegin, inputsSize)
		inputsSubSeqs[i] = inputSubSeq

		outputsBegin := op.Const(os.SubScope("begin"), []int32{0, int32(start)})
		outputsSize := op.Const(os.SubScope("size"), []int32{int32(len(inputs)), int32(libaural2.SeqLen)})
		outputSubSeq := op.Slice(os.SubScope("slice"), outputsConst, outputsBegin, outputsSize)
		outputsSubSeqs[i] = outputSubSeq
	}
	concatDim := op.Const(s.SubScope("concat_dims"), int32(0))
	concatInputs := op.Concat(is, concatDim, inputsSubSeqs)
	concatOutputs := op.Concat(os, concatDim, outputsSubSeqs)

	indicesShape := op.Const(s.SubScope("indices_shape"), []int32{int32(batchSize)})
	min := op.Const(s.SubScope("min"), int32(0))
	max := op.Const(s.SubScope("max"), int32(len(inputs)*numSubSeqs))
	indices := op.RandomUniformInt(s, indicesShape, min, max)
	inputBatch := op.Gather(is, concatInputs, indices)
	outputBatch := op.Gather(os, concatOutputs, indices)

	idsConst := op.Const(s.SubScope("clip_hashes"), ids)
	inputIdent := op.Identity(is, inputBatch) // we use identity so that output names stay constant when changing the inner workings.
	outputIdent := op.Identity(os, outputBatch)
	logger.Println(inputIdent.Shape())
	logger.Println(inputIdent.Op.Name())
	_ = inputIdent
	_ = outputIdent
	//logger.Println(concatInputs)
	//logger.Println(outputIdent.Op.Name())
	_ = idsConst
	graph, err = s.Finalize()
	return
}
