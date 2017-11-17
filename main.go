package main

import (
	"io/ioutil"
	"log"
	"os"

	"time"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.ibm.com/Blue-Horizon/aural2/boltstore"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/tftrain"
	"github.ibm.com/Blue-Horizon/aural2/tfutils/lstmutils"
	"github.ibm.com/Blue-Horizon/aural2/vsh/intent"
	"github.ibm.com/Blue-Horizon/aural2/vsh/word"
)

var logger = log.New(os.Stdout, "arl2: ", log.Lshortfile)

func main() {
	vocabList := []*libaural2.Vocabulary{
		&word.Vocabulary,
		&intent.Vocabulary,
	}
	vocabs := map[libaural2.VocabName]*libaural2.Vocabulary{} // map to get the vocabulary struct
	namesPrs := map[libaural2.VocabName]bool{}                // map to check if the vocab name exists
	onlineSessions := map[libaural2.VocabName]*tftrain.OnlineSess{}
	stepInferenceFuncs := map[libaural2.VocabName]func(*tf.Tensor) ([]float32, error){}
	graphBytes, err := ioutil.ReadFile("target/train_graph.pb") // save graph for each vocab
	if err != nil {
		logger.Fatalln(err)
	}
	for _, vocab := range vocabList { // for each vocab,
		vocabs[vocab.Name] = vocab
		namesPrs[vocab.Name] = true
		graph := tf.NewGraph()
		err = graph.Import(graphBytes, "")
		if err != nil {
			logger.Fatalln(err)
		}
		oSess, err := tftrain.NewOnlineSess(graph, "training/inputs", "training/targets", "training/Adam", "init", "training/loss_monitor/div", "seq_inference/softmax/output", "seq_inference/inputs", []string{})
		if err != nil {
			logger.Fatalln(err)
		}
		onlineSessions[vocab.Name] = &oSess
		stepInfFunc, err := lstmutils.MakeStepInference(oSess)
		if err != nil {
			logger.Fatalln(err)
		}
		stepInferenceFuncs[vocab.Name] = stepInfFunc
	}
	db, err := boltstore.Init("label_store.db", []libaural2.VocabName{"word", "intent"})
	if err != nil {
		logger.Fatalln(err)
	}
	// given an audio clip, save it.
	saveFunc := func(clip *libaural2.AudioClip) {
		if err = db.PutClipID(clip.ID()); err != nil {
			logger.Println(err)
			return
		}
		if err = ioutil.WriteFile("audio/"+clip.ID().FSsafeString()+".raw", clip[:], 0644); err != nil {
			logger.Println(err)
			return
		}
	}
	tdmMap, err := startTrainingLoops(db, onlineSessions)
	if err != nil {
		logger.Fatalln(err)
	}
	dumpClip := startVsh(saveFunc, stepInferenceFuncs)
	go serve(db, onlineSessions, namesPrs, dumpClip, tdmMap)
	for {
		time.Sleep(time.Hour)
	}
}
