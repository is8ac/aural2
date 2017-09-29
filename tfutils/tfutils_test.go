package tfutils

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"io/ioutil"
	"testing"
)

func TestReadWaveToPCM(t *testing.T) {
	s := op.NewScope()
	filePathTensor, err := tf.NewTensor("sox_sample.wav")
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
}

func TestComputeMFCC(t *testing.T) {
	s := op.NewScope()
	filePathTensor, err := tf.NewTensor("sox_sample.wav")
	if err != nil {
		t.Fail()
	}
	sampleRateTensor, err := tf.NewTensor(int32(44100))
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
		logger.Println("too short")
		t.Fail()
	}
}

func TestComputeSpectrogram(t *testing.T) {
	s := op.NewScope()
	filePathTensor, err := tf.NewTensor("sox_sample.wav")
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
		logger.Println("too short")
		t.Fail()
	}
}

func TestRenderImage(t *testing.T) {
	s := op.NewScope()
	filePathTensor, err := tf.NewTensor("sox_up.wav")
	if err != nil {
		t.Fail()
	}
	sampleRateTensor, err := tf.NewTensor(int32(44100))
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
}

func TestBytesToBytes1(t *testing.T) {
	s := op.NewScope()
	wavBytesPH, pcm := ParseWavBytesToPCM(s)
	specgramOP := ComputeSpectrogram(s.SubScope("spectrogram"), pcm, 10, 400) // compute spectrogram
	specgramJpegBytesOP := RenderImage(s.SubScope("jpeg_bytes"), specgramOP)  // render image of spectrogram.
	feeds := map[tf.Output]*tf.Tensor{}
	renderImage := BytesToBytes(s, wavBytesPH, specgramJpegBytesOP, feeds)

	wavBytes, err := ioutil.ReadFile("sox_sample.wav")
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

	wavBytes, err = ioutil.ReadFile("sox_up.wav")
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

	sampleRateTensor, err := tf.NewTensor(int32(44100))
	if err != nil {
		t.Fail()
	}
	feeds := map[tf.Output]*tf.Tensor{sampleRatePH: sampleRateTensor}
	renderImage := BytesToBytes(s, wavBytesPH, jpegBytesOP, feeds)

	wavBytes, err := ioutil.ReadFile("sox_sample.wav")
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

	wavBytes, err = ioutil.ReadFile("sox_up.wav")
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

	sampleRateTensor, err := tf.NewTensor(int32(44100))
	if err != nil {
		t.Fail()
	}
	feeds := map[tf.Output]*tf.Tensor{sampleRatePH: sampleRateTensor}
	renderImage := BytesToBytes(s, wavBytesPH, specgramJpegBytesOP, feeds)

	wavBytes, err := ioutil.ReadFile("sox_sample.wav")
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

func TestMakeCleanWav(t *testing.T){
	cleanWav, err := MakeCleanWav(44100)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	wavBytes, err := ioutil.ReadFile("sox_sample.wav")
	if err != nil {
		logger.Fatalln(err)
		t.Fail()
	}
	cleanedWav, err := cleanWav(wavBytes)
	if err != nil {
		logger.Println(err)
		t.Fail()
	}
	if len(cleanedWav) < 100 {
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
		logger.Println(err)
		t.Fail()
	}
	if err == nil {
		logger.Println("did not fail")
		t.Fail()
	}
}
