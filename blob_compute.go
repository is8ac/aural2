package main

import (
	"encoding/hex"
	"errors"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/tfutils"
	"image"
	"bytes"
	"image/png"
	"io/ioutil"
	"github.com/lucasb-eyer/go-colorful"
	"strconv"
)

const modelPath = "../python/models/intent_rnn.pb"

func getAudioClipFromFS(id libaural2.ClipID) (audioClip *libaural2.AudioClip, err error) {
	rawBytes, err := ioutil.ReadFile("audio/" + id.FSsafeString() + ".raw")
	if err != nil {
		return
	}
	if len(rawBytes) != len(audioClip) {
		err = errors.New("Got " + strconv.Itoa(len(rawBytes)) + " bytes, expected" + strconv.Itoa(len(audioClip)))
		return
	}
	audioClip = &libaural2.AudioClip{}
	copy(audioClip[:], rawBytes) // convert the slice of bytes to an array of bytes.
	return
}

func makeAddRIFF() (addRIFF clipToBlob, err error) {
	headerString := "5249464624e2040057415645666d74201000000001000100803e0000007d0000020010006461746100e20400"
	header, err := hex.DecodeString(headerString)
	if err != nil {
		return
	}
	addRIFF = func(audioClip *libaural2.AudioClip, vocabName libaural2.VocabName) ([]byte, error) {
		return append(header, audioClip[:]...), nil
	}
	return
}

func makeRenderSpectrogram() (renderSpectrogram clipToBlob, err error) {
	s := op.NewScope()
	bytesPH, pcm := tfutils.ParseRawBytesToPCM(s)
	specgramOP := tfutils.ComputeSpectrogram(s.SubScope("spectrogram"), pcm, 0, 0)
	specgramJpegBytesOP := tfutils.RenderImage(s.SubScope("jpeg_bytes"), specgramOP)
	feeds := map[tf.Output]*tf.Tensor{}
	renderImage, err := tfutils.BytesToBytes(s, bytesPH, specgramJpegBytesOP, feeds)
	if err != nil {
		return
	}

	renderSpectrogram = func(raw *libaural2.AudioClip, vocabName libaural2.VocabName) (imageBytes []byte, err error) {
		if raw == nil {
			err = errors.New("raw is nil")
			return
		}
		imageBytes, err = renderImage(raw[:])
		if err != nil {
			logger.Println(err)
			return
		}
		return
	}
	return
}

func makeRenderMFCC() (renderMFCC clipToBlob, err error) {
	s := op.NewScope()
	bytesPH, pcm := tfutils.ParseRawBytesToPCM(s)
	mfccOP, sampleRatePH := tfutils.ComputeMFCC(s.SubScope("spectrogram"), pcm)
	jpegBytesOP := tfutils.RenderImage(s.SubScope("jpeg_bytes"), mfccOP)
	sampleRateTensor, err := tf.NewTensor(int32(libaural2.SampleRate))
	if err != nil {
		return
	}
	feeds := map[tf.Output]*tf.Tensor{sampleRatePH: sampleRateTensor}
	renderImage, err := tfutils.BytesToBytes(s, bytesPH, jpegBytesOP, feeds)
	if err != nil {
		return
	}
	renderMFCC = func(raw *libaural2.AudioClip, vocabName libaural2.VocabName) (imageBytes []byte, err error) {
		if raw == nil {
			err = errors.New("raw is nil")
			return
		}
		imageBytes, err = renderImage(raw[:])
		if err != nil {
			logger.Println(err)
			return
		}
		return
	}
	return
}

func makeRenderProbs(
	seqInferenceFuncs map[libaural2.VocabName]func(*tf.Tensor)(*tf.Tensor, error), // takes a map of savedModels,
	) (
		renderProbs func(*libaural2.AudioClip, libaural2.VocabName, // returns a func that takes a clip and a vocabName
			) ([]byte, error),
			err error,
			) {
	audioClipToMFCCtensor, err := tfutils.MakeAudioClipToMFCCtensor()
	if err != nil {
		return
	}
	probsTensorToImage, err := tfutils.MakeProbsTensorToImage()
	if err != nil {
		return
	}
	renderProbs = func(clip *libaural2.AudioClip, vocabName libaural2.VocabName) (imageBytes []byte, err error) {
		seqInference, prs := seqInferenceFuncs[vocabName]
		if !prs {
			err = errors.New("don't have seqInferenceFunc for " + string(vocabName))
			return
		}
		mfccTensor, err := audioClipToMFCCtensor(clip)
		if err != nil {
			return
		}
		probs, err := seqInference(mfccTensor)
		if err != nil {
			return
		}
		imageBytes, err = probsTensorToImage(probs)
		if err != nil {
			return
		}
		return
	}
	return
}


func makeRenderArgmaxedStates(
	seqInferenceFuncs map[libaural2.VocabName]func(*tf.Tensor)(*tf.Tensor, error),
	) (
		renderProbs func(*libaural2.AudioClip, libaural2.VocabName) ([]byte, error),
		err error,
		) {
	audioClipToMFCCtensor, err := tfutils.MakeAudioClipToMFCCtensor()
	if err != nil {
		return
	}
	renderProbs = func(clip *libaural2.AudioClip, vocabName libaural2.VocabName) (imageBytes []byte, err error) {
		seqInference, prs := seqInferenceFuncs[vocabName]
		if !prs {
			err = errors.New("don't have seqInferenceFunc for " + string(vocabName))
			return
		}

		mfccTensor, err := audioClipToMFCCtensor(clip)
		if err != nil {
			return
		}
		probsTensor, err := seqInference(mfccTensor)
		if err != nil {
			return
		}
		probsList := probsTensor.Value().([][]float32)
		image := image.NewRGBA(image.Rect(0, 0, libaural2.StridesPerClip, 1))
		for x, probs := range probsList {
			cmd, prob := argmax(probs)
			color := colorful.Hsv(cmd.Hue(), 1, float64(prob))
			image.Set(x, 0, color)
		}
		buff := bytes.Buffer{}
		if err = png.Encode(&buff, image); err != nil {
			return
		}
		imageBytes = buff.Bytes()
		return
	}
	return
}

// argmax returns the index and prob of the largest elements of the list.
func argmax(probs []float32) (cmd libaural2.State, prob float32) {
	for i, val := range probs {
		if val > prob {
			prob = val
			cmd = libaural2.State(int32(i))
		}
	}
	return
}
