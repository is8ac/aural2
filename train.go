package main

import (
	"errors"
	"math/rand"
	"sync"
	"time"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"github.ibm.com/Blue-Horizon/aural2/boltstore"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/tftrain"
	"github.ibm.com/Blue-Horizon/aural2/tfutils"
)

type trainParams struct {
	sync.Mutex
	Loss      float32
	SleepTime time.Duration
}

type miniBatch struct {
	Input  *tf.Tensor
	Target *tf.Tensor
}

func newTrainingDataMap(
	getAudioClip func(libaural2.ClipID) (*libaural2.AudioClip, error),
	getLabelSet func(libaural2.ClipID, libaural2.VocabName) (libaural2.LabelSet, error),
	vocabName libaural2.VocabName,
) (
	td *trainingDataMaps,
	err error,
) {
	clipToMFCC, err := makeClipToMFCC()
	td = &trainingDataMaps{
		rand:         rand.New(rand.NewSource(time.Now().UnixNano())),
		inputs:       map[libaural2.ClipID][][]float32{},
		targets:      map[libaural2.ClipID][libaural2.StridesPerClip]int32{},
		clipToMFCC:   clipToMFCC,
		getAudioClip: getAudioClip,
		getLabelSet:  getLabelSet,
		vocabName:    vocabName,
	}
	return
}

type trainingDataMaps struct {
	sync.Mutex
	rand         *rand.Rand
	ids          []libaural2.ClipID
	inputs       map[libaural2.ClipID][][]float32
	targets      map[libaural2.ClipID][libaural2.StridesPerClip]int32
	clipToMFCC   func(*libaural2.AudioClip) ([][]float32, error)
	getAudioClip func(libaural2.ClipID) (*libaural2.AudioClip, error)
	getLabelSet  func(libaural2.ClipID, libaural2.VocabName) (libaural2.LabelSet, error)
	vocabName    libaural2.VocabName
}

func (td *trainingDataMaps) addClip(clipID libaural2.ClipID) (err error) {
	audioClip, err := td.getAudioClip(clipID)
	if err != nil {
		return
	}
	labelSet, err := td.getLabelSet(clipID, td.vocabName)
	if err != nil {
		return
	}
	mfcc, err := td.clipToMFCC(audioClip)
	if err != nil {
		return
	}
	stateIDArray := labelSet.ToStateIDArray()
	td.Lock()
	defer td.Unlock()
	td.inputs[clipID] = mfcc
	td.targets[clipID] = stateIDArray
	td.ids = append(td.ids, clipID)
	return
}

func (td *trainingDataMaps) makeMiniBatch() (mb miniBatch, err error) {
	inputs := make([][][]float32, libaural2.BatchSize)
	targets := make([][]int32, libaural2.BatchSize)
	for i := range inputs {
		start := td.rand.Intn(libaural2.StridesPerClip - libaural2.SeqLen)
		end := start + libaural2.SeqLen
		id := td.ids[td.rand.Intn(len(td.ids))]

		input := td.inputs[id]
		inputs[i] = input[start:end]

		target := td.targets[id]
		targets[i] = target[start:end]
	}
	mb.Input, err = tf.NewTensor(inputs)
	if err != nil {
		return
	}
	mb.Target, err = tf.NewTensor(targets)
	if err != nil {
		return
	}
	return
}

func startTrainingLoops(db boltstore.DB, onlineSessions map[libaural2.VocabName]*tftrain.OnlineSess, sleepms *int32) (tdmMap map[libaural2.VocabName]*trainingDataMaps, err error) {
	tdmMap = map[libaural2.VocabName]*trainingDataMaps{}
	for vocabName, oSess := range onlineSessions {
		tdm := &trainingDataMaps{}
		tdm, err = newTrainingDataMap(getAudioClipFromFS, db.GetLabelSet, vocabName)
		if err != nil {
			return
		}
		tdmMap[vocabName] = tdm
		mbChan := startTrainingDataLoop(vocabName, tdm)
		go trainLoop(vocabName, oSess, mbChan, sleepms)
		labelSets, err := db.GetAllLabelSets(vocabName)
		if err != nil {
			logger.Fatalln(err)
		}
		for _, labelSet := range labelSets {
			tdm.addClip(labelSet.ID)
		}
	}
	return
}

func trainLoop(vocabName libaural2.VocabName, oSess *tftrain.OnlineSess, miniBatchChan chan miniBatch, sleepms *int32) {
	if vocabName == libaural2.VocabName("word") {
		logger.Println("not training word vocab")
		return
	}
	var i int
	for {
		mb := <-miniBatchChan
		loss, err := oSess.Train(mb.Input, mb.Target)
		if err != nil {
			logger.Fatal(err)
		}
		if i%100 == 0 {
			logger.Println(vocabName, loss)
		}
		time.Sleep(time.Duration(*sleepms) * time.Millisecond)
		i++
	}
}

func startTrainingDataLoop(vocabName libaural2.VocabName, tdm *trainingDataMaps) (miniBatchChan chan miniBatch) {
	miniBatchChan = make(chan miniBatch, 3)
	go func() {
		for {
			numClips := len(tdm.ids)
			if numClips == 0 { // if there is no little data,
				time.Sleep(time.Second) // wait for some data to be added
				continue
			}
			mb, err := tdm.makeMiniBatch()
			if err != nil {
				logger.Println(err)
				continue
			}
			miniBatchChan <- mb
		}
	}()
	return
}

// makeClipToMFCC returns a function to turn a clip into tensor
func makeClipToMFCC() (clipToMFCC func(*libaural2.AudioClip) ([][]float32, error), err error) {
	s := op.NewScope()
	bytesPH, pcm := tfutils.ParseRawBytesToPCM(s)
	mfccOP, sampleRatePH := tfutils.ComputeMFCC(s.SubScope("spectrogram"), pcm)
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
	clipToMFCC = func(clip *libaural2.AudioClip) (mfccs [][]float32, err error) {
		clipTensor, err := tf.NewTensor(string(clip[:])) // create a string tensor from the input bytes
		if err != nil {
			logger.Println(err)
			return
		}
		result, err := sess.Run(map[tf.Output]*tf.Tensor{bytesPH: clipTensor, sampleRatePH: sampleRateTensor}, []tf.Output{mfccOP}, nil)
		if err != nil {
			logger.Println(err)
			return
		}
		shape := result[0].Shape()
		if shape[0] != int64(libaural2.StridesPerClip) || shape[1] != int64(libaural2.InputSize) {
			err = errors.New("bad shape")
			return
		}
		mfccs = result[0].Value().([][]float32)
		return
	}
	return
}
