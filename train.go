package main

import (
	"io/ioutil"
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
  Loss float32
  SleepTime time.Duration
}

func startTrainingLoops(db boltstore.DB, vocabNames []*libaural2.Vocabulary){
  trainGraphs := map[libaural2.VocabName]*tf.Graph{}
  for _, vocab := range vocabNames {
    params := new(trainParams)
    go trainLoop(vocab.Name, params)
  }
}

func trainLoop(vocabName libaural2.VocabName, params *trainParams){
  graphBytes, err := ioutil.ReadFile("model/" + string(vocabName) + "_train.pb")
  if err != nil {
    logger.Fatal(err)
  }
  graph := tf.NewGraph() // make a new graph,
  // and import the graphdef into it.
  if err = graph.Import(graphBytes, ""); err != nil {
    logger.Fatal(err)
  }
  getTrainingData, err := makeGetLSTMtrainingData(vocabName)
  if err != nil {
    logger.Fatal(err)
  }
  requiredOutputs := []string{
    "step_inference/softmax/output",
    "step_inference/initial_state_names",
    "step_inference/final_state_names",
    "zeros",
  }
  oSess, err := tftrain.NewOnlineSess(graph, "training/inputs", "training/targets", "training/Adam", "init", "training/loss_monitor/truediv", "step_inference/softmax/output", "step_inference/inputs", requiredOutputs)
  if err != nil {
    logger.Fatal(err)
  }
  var loss2 float32
  for {
    inputTensor, targetTensor, _ := getTrainingData(1) // [7, 100, 13], [7, 100]
    loss2, err = oSess.Train(inputTensor, targetTensor)
    if err != nil {
      logger.Fatal(err)
    }
    time.Sleep(100 * time.Millisecond)
  }
}

// makeTrainingDataGraphdef makes a graphdef that, when pulled on, will give training data.
func makeTrainingDataGraphdef(getAllLabelSets func(libaural2.VocabName) (map[libaural2.ClipID]libaural2.LabelSet, error)) (graph tf.Graph) {
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

	return func(w http.ResponseWriter, r *http.Request) {
		vocabName := libaural2.VocabName(mux.Vars(r)["vocab"])
		if !vocabPrs[vocabName] {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		labelSets, err := getAllLabelSets(vocabName)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		logger.Println(len(labelSets))
		var inputs [][][]float32
		var outputs [][libaural2.StridesPerClip]int32
		var ids []libaural2.ClipID
		// iterate over the clips
		for id, labelSet := range labelSets {
			audioClip, err := getAudioClipFromFS(id)
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
				logger.Println(shape, "is not", libaural2.StridesPerClip)
				return
			}
			input := result[0].Value().([][]float32)
			//input = libaural2.GenFakeInput(labelSet.ToStateIDArray())
			inputs = append(inputs, input)
			outputs = append(outputs, labelSet.ToStateIDArray())
			ids = append(ids, id)
		}
		logger.Println(len(inputs), len(outputs))
		graph, err := tfutils.EmbedTrainingData(inputs, outputs, ids, 8, libaural2.BatchSize) // take 8 sub seqs, and batch size of 10
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		if _, err = graph.WriteTo(w); err != nil {
			logger.Println(err)
		}
	}
}

// makeTrainingDataGraphdef makes a graphdef that, when pulled on, will give training data.
func makeClipToMFCCTensor(getAllLabelSets func(libaural2.VocabName) (map[libaural2.ClipID]libaural2.LabelSet, error)) (graph tf.Graph) {
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

	return func(w http.ResponseWriter, r *http.Request) {
		vocabName := libaural2.VocabName(mux.Vars(r)["vocab"])
		if !vocabPrs[vocabName] {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		labelSets, err := getAllLabelSets(vocabName)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		logger.Println(len(labelSets))
		var inputs [][][]float32
		var outputs [][libaural2.StridesPerClip]int32
		var ids []libaural2.ClipID
		// iterate over the clips
		for id, labelSet := range labelSets {
			audioClip, err := getAudioClipFromFS(id)
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
				logger.Println(shape, "is not", libaural2.StridesPerClip)
				return
			}
			input := result[0].Value().([][]float32)
			//input = libaural2.GenFakeInput(labelSet.ToStateIDArray())
			inputs = append(inputs, input)
			outputs = append(outputs, labelSet.ToStateIDArray())
			ids = append(ids, id)
		}
		logger.Println(len(inputs), len(outputs))
		graph, err := tfutils.EmbedTrainingData(inputs, outputs, ids, 8, libaural2.BatchSize) // take 8 sub seqs, and batch size of 10
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		if _, err = graph.WriteTo(w); err != nil {
			logger.Println(err)
		}
	}
}


func makeGetLSTMtrainingData(vocabName libaural2.VocabName) (getTrainingData func(int) (*tf.Tensor, *tf.Tensor, error), err error) {
	graphBytes, err := ioutil.ReadFile("lstm_trainingdata.pb")
	if err != nil {
		return
	}
	graph := tf.NewGraph() // make a new graph,
	// and import the graphdef into it.
	if err = graph.Import(graphBytes, ""); err != nil {
		return
	}
	inputOP, err := getOP(graph, "inputs/Identity")
	if err != nil {
		return
	}
	targetOP, err := getOP(graph, "outputs/Identity")
	if err != nil {
		return
	}
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return
	}
	getTrainingData = func(i int) (inputTensor *tf.Tensor, targetTensor *tf.Tensor, err error) {
		result, err := session.Run(
			map[tf.Output]*tf.Tensor{},
			[]tf.Output{
				inputOP.Output(0),
				targetOP.Output(0),
			},
			nil,
		)
		if err != nil {
			return
		}
		inputTensor = result[0]
		targetTensor = result[1]
		return
	}
	return
}
