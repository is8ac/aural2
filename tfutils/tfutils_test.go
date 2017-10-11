package tfutils

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"io/ioutil"
	"testing"
)

func TestReadWaveToPCM(t *testing.T) {
	s := op.NewScope()
	filePathTensor, err := tf.NewTensor("sox_sample_16k.wav")
	if err != nil {
		t.Fail()
	}
	filePathPH, pcmOP := ReadWaveToPCM(s)
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

	result, err := sess.Run(map[tf.Output]*tf.Tensor{filePathPH: filePathTensor}, []tf.Output{pcmOP}, nil)
	if err != nil {
		logger.Println(err)
		return
	}
	shape := result[0].Shape()
	if len(shape) != 1 {
		t.Fail()
	}
	if shape[0] < 1000 {
		t.Fail()
	}
	value := result[0].Value().([]float32)
	if len(value) < 1000 {
		t.Fail()
	}
	err = sess.Close()
	if err != nil {
		t.Fatal(err)
	}
}

func TestComputeMFCC(t *testing.T) {
	s := op.NewScope()
	filePathTensor, err := tf.NewTensor("sox_sample_16k.wav")
	if err != nil {
		t.Fail()
	}
	sampleRateTensor, err := tf.NewTensor(int32(16000))
	if err != nil {
		t.Fail()
	}
	filePathPH, pcmOP := ReadWaveToPCM(s.SubScope("read_pcm"))
	mfccOP, sampleRatePH := ComputeMFCC(s.SubScope("mfcc"), pcmOP)

	graph, err := s.Finalize()
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}

	result, err := sess.Run(map[tf.Output]*tf.Tensor{filePathPH: filePathTensor, sampleRatePH: sampleRateTensor}, []tf.Output{mfccOP}, nil)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	shape := result[0].Shape()
	if len(shape) != 2 {
		t.Fail()
	}
	if shape[1] != 13 {
		t.Fail()
	}
	values := result[0].Value().([][]float32)
	if len(values) < 100 {
		logger.Println("wrong len")
		t.Fail()
	}
	err = sess.Close()
	if err != nil {
		t.Fatal(err)
	}
}

func TestComputeMFCCinputSet(t *testing.T) {
	rawBytes, err := ioutil.ReadFile("10s.raw")
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	if len(rawBytes) != libaural2.AudioClipLen {
		logger.Println(err)
		t.Fail()
	}

	bytesTensor, err := tf.NewTensor(string(rawBytes))
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	sampleRateTensor, err := tf.NewTensor(int32(16000))
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	s := op.NewScope()
	bytesPH, pcmOutput := ParseRawBytesToPCM(s.SubScope("parse_raw"))
	mfccOP, sampleRatePH := ComputeMFCC(s.SubScope("mfcc"), pcmOutput)

	graph, err := s.Finalize()
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}

	result, err := sess.Run(map[tf.Output]*tf.Tensor{bytesPH: bytesTensor, sampleRatePH: sampleRateTensor}, []tf.Output{mfccOP}, nil)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	_ = result[0].Value().([][]float32)
	logger.Println(result[0].Shape())
	err = sess.Close()
	if err != nil {
		t.Fatal(err)
	}
}

func TestComputeSpectrogram(t *testing.T) {
	s := op.NewScope()
	filePathTensor, err := tf.NewTensor("sox_sample_16k.wav")
	if err != nil {
		t.Fail()
	}
	filePathPH, pcmOP := ReadWaveToPCM(s.SubScope("read_pcm"))
	specgramOP := ComputeSpectrogram(s.SubScope("spectrogram"), pcmOP, 0, 0)

	graph, err := s.Finalize()
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	result, err := sess.Run(map[tf.Output]*tf.Tensor{filePathPH: filePathTensor}, []tf.Output{specgramOP}, nil)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	shape := result[0].Shape()
	if len(shape) != 2 {
		t.Fail()
	}
	if shape[1] < 20 {
		logger.Println(shape[1])
		t.Fail()
	}
	values := result[0].Value().([][]float32)
	if len(values) < 100 {
		logger.Println("len:", len(values))
		t.Fail()
	}
	err = sess.Close()
	if err != nil {
		t.Fatal(err)
	}
}

func TestComputeSpectrogram2(t *testing.T) {
	rawBytes, err := ioutil.ReadFile("10s.raw")
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	if len(rawBytes) != libaural2.AudioClipLen {
		logger.Println(err)
		t.Fail()
	}

	bytesTensor, err := tf.NewTensor(string(rawBytes))
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	s := op.NewScope()
	bytesPH, pcmOutput := ParseRawBytesToPCM(s.SubScope("parse_raw"))
	spectrogramOP := ComputeSpectrogram(s.SubScope("spectrogram"), pcmOutput, 0, 20)

	graph, err := s.Finalize()
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}

	result, err := sess.Run(map[tf.Output]*tf.Tensor{bytesPH: bytesTensor}, []tf.Output{spectrogramOP}, nil)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	_ = result[0].Value().([][]float32)
	logger.Println(result[0].Shape())

	err = sess.Close()
	if err != nil {
		t.Fatal(err)
	}
}

func TestRenderImage(t *testing.T) {
	s := op.NewScope()
	filePathTensor, err := tf.NewTensor("sox_up_16k.wav")
	if err != nil {
		t.Fail()
	}
	sampleRateTensor, err := tf.NewTensor(int32(16000))
	if err != nil {
		t.Fail()
	}
	filePathPH, pcmOP := ReadWaveToPCM(s.SubScope("read_pcm"))                  // read in the wav file and convert to float32 pcm
	mfccOP, sampleRatePH := ComputeMFCC(s, pcmOP)                               // compute MFCC
	specgramOP := ComputeSpectrogram(s.SubScope("spectrogram"), pcmOP, 10, 400) // compute spectrogram
	mfccJpegBytesOP := RenderImage(s.SubScope("jpeg_bytes"), mfccOP)            // render image of mfcc
	specgramJpegBytesOP := RenderImage(s.SubScope("jpeg_bytes"), specgramOP)    // render image of spectrogram.
	graph, err := s.Finalize()
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}

	result, err := sess.Run(map[tf.Output]*tf.Tensor{filePathPH: filePathTensor, sampleRatePH: sampleRateTensor}, []tf.Output{mfccJpegBytesOP, specgramJpegBytesOP}, nil)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	mfccJpegBytes := result[0].Value().(string)
	if len(mfccJpegBytes) < 100 {
		t.Fail()
	}
	specgramJpegBytes := result[1].Value().(string)
	if len(specgramJpegBytes) < 100 {
		t.Fail()
	}
	if err = ioutil.WriteFile("mfcc.jpeg", []byte(mfccJpegBytes), 0644); err != nil {
		logger.Println(err)
	}
	if err = ioutil.WriteFile("spectrogram.jpeg", []byte(specgramJpegBytes), 0644); err != nil {
		logger.Println(err)
	}
	err = sess.Close()
	if err != nil {
		t.Fatal(err)
	}
}

func TestRenderWav(t *testing.T) {
	s := op.NewScope()
	rawBytes, err := ioutil.ReadFile("cmd_16k.raw")
	if err != nil {
		logger.Println(err)
		return
	}
	inputTensor, err := tf.NewTensor(string(rawBytes)) // create a string tensor from the input bytes
	if err != nil {
		return
	}

	sampleRateTensor, err := tf.NewTensor(int32(libaural2.SampleRate)) // create a string tensor from the input bytes
	if err != nil {
		return
	}

	bytesPH, pcmOutput := ParseRawBytesToPCM(s.SubScope("parseRawBytes"))
	wavBytesOutput, sampleRatePH := EncodeWav(s.SubScope("encodeWav"), pcmOutput)
	graph, err := s.Finalize()
	if err != nil {
		logger.Println(err)
		t.Fail()
		return
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}

	result, err := sess.Run(map[tf.Output]*tf.Tensor{bytesPH: inputTensor, sampleRatePH: sampleRateTensor}, []tf.Output{wavBytesOutput}, nil)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	logger.Println(result[0].Shape())
	wavBytes := result[0].Value().(string)
	if len(wavBytes) < 100 {
		t.Fail()
	}
	if err = ioutil.WriteFile("encoded.wav", []byte(wavBytes), 0644); err != nil {
		logger.Println(err)
	}
	err = sess.Close()
	if err != nil {
		t.Fatal(err)
	}
}

func TestParseRawToPCM(t *testing.T) {
	s := op.NewScope()
	rawBytesPH, pcmOutput := ParseRawBytesToPCM(s)
	graph, err := s.Finalize()
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}

	rawBytes, err := ioutil.ReadFile("cmd_16k.raw")
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	rawBytesTensor, err := tf.NewTensor(string(rawBytes))
	if err != nil {
		logger.Println(err)
		t.Fail()
	}

	result, err := sess.Run(map[tf.Output]*tf.Tensor{rawBytesPH: rawBytesTensor}, []tf.Output{pcmOutput}, nil)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	floats := result[0].Value().([]float32)
	if len(floats) != 160000 {
		t.Fail()
	}
	err = sess.Close()
	if err != nil {
		t.Fatal(err)
	}
}

func TestBytesToBytes1(t *testing.T) {
	s := op.NewScope()
	wavBytesPH, pcm := ParseWavBytesToPCM(s)
	specgramOP := ComputeSpectrogram(s.SubScope("spectrogram"), pcm, 10, 400) // compute spectrogram
	specgramJpegBytesOP := RenderImage(s.SubScope("jpeg_bytes"), specgramOP)  // render image of spectrogram.
	feeds := map[tf.Output]*tf.Tensor{}
	renderImage, err := BytesToBytes(s, wavBytesPH, specgramJpegBytesOP, feeds)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}

	wavBytes, err := ioutil.ReadFile("sox_sample_16k.wav")
	if err != nil {
		logger.Fatalln(err)
	}
	imageBytes, err := renderImage(wavBytes)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	if len(imageBytes) < 100 {
		t.Fail()
	}

	wavBytes, err = ioutil.ReadFile("sox_up_16k.wav")
	if err != nil {
		logger.Fatalln(err)
	}
	imageBytes, err = renderImage(wavBytes)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	if len(imageBytes) < 100 {
		t.Fail()
	}
}

func TestBytesToBytes2(t *testing.T) {
	s := op.NewScope()
	wavBytesPH, pcm := ParseWavBytesToPCM(s)
	specgramOP, sampleRatePH := ComputeMFCC(s, pcm)
	jpegBytesOP := RenderImage(s.SubScope("jpeg_bytes"), specgramOP) // render image

	sampleRateTensor, err := tf.NewTensor(int32(16000))
	if err != nil {
		t.Fail()
	}
	feeds := map[tf.Output]*tf.Tensor{sampleRatePH: sampleRateTensor}
	renderImage, err := BytesToBytes(s, wavBytesPH, jpegBytesOP, feeds)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}

	wavBytes, err := ioutil.ReadFile("sox_sample_16k.wav")
	if err != nil {
		logger.Fatalln(err)
	}
	imageBytes, err := renderImage(wavBytes)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	if len(imageBytes) < 100 {
		t.Fail()
	}

	wavBytes, err = ioutil.ReadFile("sox_up_16k.wav")
	if err != nil {
		logger.Fatalln(err)
	}
	imageBytes, err = renderImage(wavBytes)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	if len(imageBytes) < 100 {
		t.Fail()
	}
}

func TestBytesToBytesConcurrent(t *testing.T) {
	s := op.NewScope()
	wavBytesPH, pcm := ParseWavBytesToPCM(s)
	specgramOP, sampleRatePH := ComputeMFCC(s, pcm)
	specgramJpegBytesOP := RenderImage(s.SubScope("jpeg_bytes"), specgramOP) // render image of spectrogram.

	sampleRateTensor, err := tf.NewTensor(int32(16000))
	if err != nil {
		t.Fail()
	}
	feeds := map[tf.Output]*tf.Tensor{sampleRatePH: sampleRateTensor}
	renderImage, err := BytesToBytes(s, wavBytesPH, specgramJpegBytesOP, feeds)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}

	wavBytes, err := ioutil.ReadFile("sox_sample_16k.wav")
	if err != nil {
		logger.Fatalln(err)
	}
	doneChan := make(chan bool)
	total := 100
	for i := 0; i < total; i++ {
		go func() {
			imageBytes, err := renderImage(wavBytes)
			if err != nil {
				t.Fail()
			}
			if len(imageBytes) < 100 {
				t.Fail()
			}
			doneChan <- true
		}()
	}
	for i := 0; i < total; i++ {
		successful := <-doneChan
		if !successful {
			t.Fail()
		}
	}
}

func TestMakeCleanWav(t *testing.T) {
	cleanWav, err := MakeCleanWav(16000)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	wavBytes, err := ioutil.ReadFile("sox_sample_16k.wav")
	if err != nil {
		logger.Fatalln(err)
		t.Fail()
	}
	cleanedWav, err := cleanWav(wavBytes)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	if len(cleanedWav) < 10 {
		t.Fail()
	}
	_, err = cleanWav([]byte("some byte string that is not wav audio"))
	if err == nil {
		logger.Println("did not fail")
		t.Fail()
	}
	cleanWav, err = MakeCleanWav(12345)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	_, err = cleanWav(wavBytes)
	if err == nil {
		logger.Println("did not fail")
		t.Fail()
	}
}


func TestEmbedTrainingData(t *testing.T) {
	hash := libaural2.ClipID{}
	labelSets := []libaural2.LabelSet{
		libaural2.LabelSet{
			ID: hash,
			Labels: []libaural2.Label{
				libaural2.Label{
					Cmd:  libaural2.Who,
					Time: 1.23,
				},
				libaural2.Label{
					Cmd:  libaural2.What,
					Time: 4.03,
				},
				libaural2.Label{
					Cmd:  libaural2.When,
					Time: 9.20,
				},
			},
		},
		libaural2.LabelSet{
			ID: hash,
			Labels: []libaural2.Label{
				libaural2.Label{
					Cmd:  libaural2.Yes,
					Time: 0.93,
				},
				libaural2.Label{
					Cmd:  libaural2.No,
					Time: 9.02,
				},
			},
		},
		libaural2.LabelSet{
			ID: hash,
			Labels: []libaural2.Label{
				libaural2.Label{
					Cmd:  libaural2.OKgoogle,
					Time: 5.723,
				},
			},
		},
		libaural2.LabelSet{
			ID: hash,
			Labels: []libaural2.Label{
				libaural2.Label{
					Cmd:  libaural2.Alexa,
					Time: 9.53,
				},
				libaural2.Label{
					Cmd:  libaural2.CtrlC,
					Time: 2.7,
				},
			},
		},
	}
	var inputs [][][]float32
	var outputs [][libaural2.StridesPerClip]int32
	var ids []libaural2.ClipID
	// iterate over the labelSets
	for _, labelSet := range labelSets {
		mfcc := make([]float32, libaural2.InputSize)
		input := make([][]float32, libaural2.StridesPerClip)
		for i := range input {
			input[i] = mfcc
		}
		inputs = append(inputs, input)
		outputs = append(outputs, labelSet.ToCmdArray())
		ids = append(ids, labelSet.ID)
	}
	graph, err := EmbedTrainingData(inputs, outputs, ids)
	if err != nil {
		t.Fatal(err)
	}
	// now check that the graph is good.
	inputOP := graph.Operation("inputs/Identity")
	if inputOP == nil {
		t.Fatal("input OP nil")
	}
	outputOP := graph.Operation("outputs/Identity")
	if inputOP == nil {
		t.Fatal("output OP nil")
	}
	clipHashesOP := graph.Operation("clip_hashes/Const")
	if inputOP == nil {
		t.Fatal("clip_hashes OP nil")
	}
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	result, err := session.Run(
		map[tf.Output]*tf.Tensor{},
		[]tf.Output{
			inputOP.Output(0),
			outputOP.Output(0),
			clipHashesOP.Output(0),
		},
		nil,
	)
	if err != nil {
		t.Fail()
	}
	inputShape := result[0].Shape()
	outputShape := result[1].Shape()
	clipHashesShape := result[2].Shape()
	logger.Println(clipHashesShape)
	if len(inputShape) != 3 {
		t.Fatal("input wrong dims")
	}
	if len(outputShape) != 2 {
		t.Fatal("output wrong dims")
	}
	if len(clipHashesShape) != 2 {
		t.Fatal("clip shapes wrong dims")
	}
	outerDimLen := int64(len(labelSets))
	if inputShape[0] != outerDimLen || outputShape[0] != outerDimLen || clipHashesShape[0] != outerDimLen {
		t.Fatal("outerDim is wrong")
	}
	secondDimLen := int64(libaural2.StridesPerClip)
	if inputShape[1] != secondDimLen || outputShape[1] != secondDimLen {
		t.Fatal("second dim is wrong")
	}
	if inputShape[2] != int64(libaural2.InputSize) {
		t.Fatal("inputSize is wrong")
	}
	err = session.Close()
	if err != nil {
		t.Fatal(err)
	}
}
