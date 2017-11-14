package main

import (
	"io/ioutil"
	"log"
	"os"

	"github.com/fhs/gompd/mpd"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.ibm.com/Blue-Horizon/aural2/boltstore"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/tfutils/lstmutils"
	"github.ibm.com/Blue-Horizon/aural2/vsh/intent"
	"github.ibm.com/Blue-Horizon/aural2/vsh/word"
	"time"
)

var logger = log.New(os.Stdout, "arl2: ", log.Lshortfile)

func main() {
	vocabList := []*libaural2.Vocabulary{
		&word.Vocabulary,
		&intent.Vocabulary,
	}
	models := map[libaural2.VocabName]tf.SavedModel{}                                   // get the savedModel
	seqInferenceFuncs := map[libaural2.VocabName]func(*tf.Tensor) (*tf.Tensor, error){} // get the function to do inference on a whole clip
	vocabs := map[libaural2.VocabName]*libaural2.Vocabulary{}                           // get the vocabulary struct
	namesPrs := map[libaural2.VocabName]bool{}                                          // check if the vocab name exists.
	for _, vocab := range vocabList {
		vocabs[vocab.Name] = vocab
		namesPrs[vocab.Name] = true
		graphBytes, err := ioutil.ReadFile("models/" + string(vocab.Name) + ".pb")
		if err != nil {
			logger.Fatalln(err)
		}
    graph := tf.NewGraph()
		err = graph.Import(graphBytes, "")
		if err != nil {
			logger.Fatalln(err)
		}
		sess, err := tf.NewSession(graph, nil)
    savedModel := tf.SavedModel{
      Graph: graph,
      Session: sess,
    }
		if err != nil {
			logger.Fatalln(err)
		}
		models[vocab.Name] = savedModel
		seqFunc, err := lstmutils.MakeSeqInference(savedModel)
		if err != nil {
			logger.Fatalln(err)
		}
		seqInferenceFuncs[vocab.Name] = seqFunc
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
	client, err := mpd.Dial("tcp", "localhost:6600")
	if err != nil {
		log.Fatalln(err)
	}
	defer client.Close()
	go func() {
		for {
			client.Ping()
			time.Sleep(time.Second)
		}
	}()
	go startVsh(saveFunc, client, models)
	go serve(db, seqInferenceFuncs, namesPrs)
  go startTrainingLoops(db, vocabList)
  for {
		time.Sleep(time.Hour)
	}
}
