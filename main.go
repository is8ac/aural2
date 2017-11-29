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
	vocabs := map[libaural2.VocabName]*libaural2.Vocabulary{}                           // map to get the vocabulary struct
	namesPrs := map[libaural2.VocabName]bool{}                                          // map to check if the vocab name exists
	onlineSessions := map[libaural2.VocabName]*tftrain.OnlineSess{}                     // map of online sessions
	stepInferenceFuncs := map[libaural2.VocabName]func(*tf.Tensor) ([]float32, error){} // map of functions to run statefull inference on individual MFCCs.
	untrainedGraphBytes, err := ioutil.ReadFile("target/train_graph.pb")                // load the untrained training graph
	if err != nil {
		logger.Fatalln(err)
	}
	for _, vocab := range vocabList { // for each vocab,
		vocabs[vocab.Name] = vocab
		namesPrs[vocab.Name] = true
		graph := tf.NewGraph()
		trainedGraphBytes, err := ioutil.ReadFile("models/" + string(vocab.Name) + ".pb") // try to read the trained graph for that vocab
		if err != nil {                                                                   // if the graph could not be loaded,
			logger.Println("Using untrained graph for", vocab.Name)
			err = graph.Import(untrainedGraphBytes, "") // then fall back to the untrained graph
			if err != nil {
				logger.Fatalln(err)
			}
		} else {
			logger.Println("Using trained graph for", vocab.Name)
			err = graph.Import(trainedGraphBytes, "") // but if it could be loaded, use the trained graph for that vocab.
			if err != nil {
				logger.Fatalln(err)
			}
		}
		requiredOutputs := []string{
			"step_inference/softmax/output",
			"step_inference/initial_state_names",
			"step_inference/final_state_names",
			"step_inference/loss_monitor/count",
			"seq_inference/loss_monitor/count",
			"seq_inference/loss_monitor/sum_mean_loss",
			"step_inference/loss_monitor/sum_mean_loss",
			"zeros",
		}

		// we need to create an online session so we can train and infer at the same time. It takes vareus operation names.
		oSess, err := tftrain.NewOnlineSess(graph,
			"training/inputs",  // placeholder for batch training inputs
			"training/targets", // placeholder for batch training targets
			"training/Adam",    // training operation
			"init",             // OP to initalise the variables
			"training/loss_monitor/div",    // the loss of the graph when training
			"seq_inference/softmax/output", // output for live inference
			"seq_inference/inputs",         // input for live inference
			requiredOutputs,                // any other ops which need to be preserved when freezing
		)
		if err != nil {
			logger.Fatalln(err)
		}
		onlineSessions[vocab.Name] = &oSess
		stepInfFunc, err := lstmutils.MakeStepInference(oSess) // make the func to do statefull step inference
		if err != nil {
			logger.Fatalln(err)
		}
		stepInferenceFuncs[vocab.Name] = stepInfFunc // and put in the map.
	}
	db, err := boltstore.Init("label_store.db", []libaural2.VocabName{"word", "intent"}) // open the bolt DB
	if err != nil {
		logger.Fatalln(err)
	}
	// func to save a 10 second audio clip
	saveFunc := func(clip *libaural2.AudioClip) {
		// write the file to disk
		if err = ioutil.WriteFile("audio/"+clip.ID().FSsafeString()+".raw", clip[:], 0644); err != nil {
			logger.Println(err)
			return
		}
		// add it to the DB
		if err = db.PutClipID(clip.ID()); err != nil {
			logger.Println(err)
			return
		}
	}
	tdmMap, err := startTrainingLoops(db, onlineSessions)
	if err != nil {
		logger.Fatalln(err)
	}
	// func to be run on shutdown.
	shutdownFunc := func() {
		for vocabName, oSess := range onlineSessions { // for each model,
			logger.Println("writing", vocabName, "model to disk")
			frozenGraph, err := oSess.Save() // freeze it,
			if err != nil {
				logger.Println(err)
				continue
			}
			f, err := os.Create("models/" + string(vocabName) + ".pb")
			if err != nil {
				logger.Println(err)
				continue
			}
			if _, err = frozenGraph.WriteTo(f); err != nil { // and write to disk.
				logger.Println(err)
			}
		}
		logger.Println("models saved, shutting down now.")
		os.Exit(0)
	}
	// start vsh, passing it the step
	dumpClip := startVsh(saveFunc, stepInferenceFuncs, shutdownFunc)
	// start the http server and REST API.
	go serve(db, onlineSessions, namesPrs, dumpClip, tdmMap)
	for { // endless loop of saving the models every 10 minutes.
		time.Sleep(10 * time.Minute)
		for vocabName, oSess := range onlineSessions {
			logger.Println("writing", vocabName, "model to disk")
			frozenGraph, err := oSess.Save()
			if err != nil {
				logger.Println(err)
				continue
			}
			f, err := os.Create("models/" + string(vocabName) + ".pb")
			if err != nil {
				logger.Println(err)
				continue
			}
			if _, err = frozenGraph.WriteTo(f); err != nil {
				logger.Println(err)
			}
		}
	}
}
