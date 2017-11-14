package main

import (
	"bytes"
	"errors"
	"image"
	"image/png"
	"io/ioutil"
	"net/http"
	"strconv"

	"html/template"

	"encoding/base32"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"

	"github.com/gorilla/mux"
	"github.ibm.com/Blue-Horizon/aural2/boltstore"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/tfutils"
	"repo.hovitos.engineering/MTN/wave/cloud/application/aural/urbitname"
)

func parseURLvar(urlVar string) (clipID libaural2.ClipID, err error) {
	fileHash, err := base32.StdEncoding.DecodeString(urlVar)
	if err != nil {
		return
	}
	if len(fileHash) != 32 {
		err = errors.New("hash length must be 32 bytes")
		return
	}
	emptyBytes := make([]byte, 32)
	if bytes.Equal(fileHash, emptyBytes) {
		err = errors.New("hash is nil")
	}
	copy(clipID[:], fileHash)
	return
}

// makeServeAudioDerivedBlob makes a handler func to serve a []byte derived from an AudioClip.
func makeMakeServeAudioDerivedBlob(vocabPrs map[libaural2.VocabName]bool) func(clipToBlob) func(w http.ResponseWriter, r *http.Request) {
	return func(toBlob clipToBlob) func(http.ResponseWriter, *http.Request) {
		return func(w http.ResponseWriter, r *http.Request) {
			vocabName := libaural2.VocabName(mux.Vars(r)["vocab"])
			if !vocabPrs[vocabName] {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			audioIDstring := mux.Vars(r)["sampleID"]
			clipID, err := parseURLvar(audioIDstring)
			if err != nil {
				logger.Println(err)
				http.Error(w, "", http.StatusBadRequest)
				return
			}
			audioClip, err := getAudioClipFromFS(clipID)
			if err != nil {
				logger.Println(err)
				http.Error(w, "", http.StatusInternalServerError)
				return
			}
			blobBytes, err := toBlob(audioClip, vocabName)
			if err != nil {
				logger.Println(err)
				http.Error(w, "", http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Length", strconv.Itoa(len(blobBytes)))
			w.Header().Set("Accept-Ranges", "bytes")
			w.Write(blobBytes)
		}
	}
}

// makeServeLabelsSetDerivedBlob makes a handler func to serve a []byte derived from a labelSet.
func makeServeLabelsSetDerivedBlob(
	vocabPrs map[libaural2.VocabName]bool,
	getLabelsSet func(libaural2.ClipID, libaural2.VocabName) (libaural2.LabelSet, error),
	setToBlob func(libaural2.LabelSet) ([]byte, error),
) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		vocabName := libaural2.VocabName(mux.Vars(r)["vocab"])
		if !vocabPrs[vocabName] {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		audioIDstring := mux.Vars(r)["sampleID"]
		clipID, err := parseURLvar(audioIDstring)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		labelSet, err := getLabelsSet(clipID, vocabName)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		serialized, err := setToBlob(labelSet)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(serialized)))
		w.Header().Set("Accept-Ranges", "bytes")
		w.Write(serialized)
	}
}

// makeServeTrainingDataGraphdef makes a handler func to serve a graphdef that, when pulled on, will give training data.
func makeServeTrainingDataGraphdef(getAllLabelSets func(libaural2.VocabName) (map[libaural2.ClipID]libaural2.LabelSet, error), vocabPrs map[libaural2.VocabName]bool) func(http.ResponseWriter, *http.Request) {
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
				http.Error(w, "", http.StatusInternalServerError)
				return
			}
			clipTensor, err := tf.NewTensor(string(audioClip[:])) // create a string tensor from the input bytes
			if err != nil {
				logger.Println(err)
				http.Error(w, "", http.StatusInternalServerError)
				return
			}
			result, err := sess.Run(map[tf.Output]*tf.Tensor{bytesPH: clipTensor, sampleRatePH: sampleRateTensor}, []tf.Output{mfccOP}, nil)
			if err != nil {
				logger.Println(err)
				http.Error(w, "", http.StatusInternalServerError)
				return
			}
			shape := result[0].Shape()
			if shape[0] != int64(libaural2.StridesPerClip) || shape[1] != int64(libaural2.InputSize) {
				logger.Println(shape, "is not", libaural2.StridesPerClip)
				http.Error(w, "", http.StatusInternalServerError)
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

func makeWriteLabelsSet(put func(libaural2.LabelSet) error, vocabPrs map[libaural2.VocabName]bool) func(http.ResponseWriter, *http.Request) {
	nilID := libaural2.ClipID{}
	return func(w http.ResponseWriter, r *http.Request) {
		audioIDstring := mux.Vars(r)["sampleID"]
		vocabName := libaural2.VocabName(mux.Vars(r)["vocab"])
		if !vocabPrs[vocabName] {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		sampleID, err := parseURLvar(audioIDstring)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		logger.Println("putting", sampleID.String(), "to", vocabName)
		serialized, err := ioutil.ReadAll(r.Body)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		if err := r.Body.Close(); err != nil {
			logger.Println(err)
			return
		}
		labelsSet, err := libaural2.DeserializeLabelSet(serialized)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		if !labelsSet.IsGood() {
			logger.Println(sampleID, "bad labelSet", labelsSet.ID)
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		if vocabName != labelsSet.VocabName {
			logger.Println(vocabName, "!=", labelsSet.VocabName)
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		if sampleID != labelsSet.ID {
			logger.Println(sampleID, "!=", labelsSet.ID)
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		if sampleID == nilID {
			logger.Println("clip ID is empty")
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		if err := put(labelsSet); err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
	}
}

func makeServeIndex(list func() []libaural2.ClipID, vocabPrs map[libaural2.VocabName]bool) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		vocabName := libaural2.VocabName(mux.Vars(r)["vocab"])
		if !vocabPrs[vocabName] {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var indexTemplate = template.Must(template.ParseFiles("templates/index.html"))
		ids := list()
		params := struct {
			IDs       []libaural2.ClipID
			VocabName libaural2.VocabName
		}{
			IDs:       ids,
			VocabName: vocabName,
		}
		err := indexTemplate.Execute(w, params)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
	}
}

func makeServeTagUI(vocabPrs map[libaural2.VocabName]bool) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		audioIDstring := mux.Vars(r)["sampleID"]
		vocabName := libaural2.VocabName(mux.Vars(r)["vocab"])
		if !vocabPrs[vocabName] {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		hash, err := parseURLvar(audioIDstring)
		if err != nil {
			logger.Println(err)
			logger.Println(audioIDstring)
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		var uiTemplate = template.Must(template.ParseFiles("templates/tag.html"))
		params := struct {
			Base32ID      string
			UrbitSampleID string
			VocabName     libaural2.VocabName
		}{
			UrbitSampleID: urbitname.Encode(hash[:4]),
			Base32ID:      base32.StdEncoding.EncodeToString(hash[:]),
			VocabName:     vocabName,
		}
		err = uiTemplate.Execute(w, params)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
	}
}

func makeServeVocabUI(vocabPrs map[libaural2.VocabName]bool) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		vocabName := libaural2.VocabName(mux.Vars(r)["vocab"])
		if !vocabPrs[vocabName] {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var tmpl = template.Must(template.ParseFiles("templates/vocab.html"))
		params := struct {
			VocabName     libaural2.VocabName
		}{
			VocabName:     vocabName,
		}
		err := tmpl.Execute(w, params)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
	}
}


func makeSampleHandler(putClipID func(libaural2.ClipID) error) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		rawBytes, err := ioutil.ReadAll(r.Body)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		if len(rawBytes) != libaural2.AudioClipLen {
			logger.Println("wrong length")
			http.Error(w, "wrong length", http.StatusBadRequest)
			return
		}
		var audioClip libaural2.AudioClip
		copy(audioClip[:], rawBytes) // convert the slice of bytes to an array of bytes.
		if err != nil {
			logger.Println(err)
			http.Error(w, "malformed audio", http.StatusBadRequest)
			return
		}
		id := audioClip.ID()
		logger.Println("putting clipID")
		if err = putClipID(id); err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		if err = ioutil.WriteFile("audio/"+id.FSsafeString()+".raw", rawBytes, 0644); err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		w.Write([]byte(id.FSsafeString()))
		return
	}
}

func renderColorLabelSetImage(labelSet libaural2.LabelSet) (pngBytes []byte, err error) {
	image := image.NewRGBA(image.Rect(0, 0, libaural2.StridesPerClip, 1))
	for x, state := range labelSet.ToStateArray() {
		state.Hue()
		image.Set(x, 0, state)
	}
	buff := bytes.Buffer{}
	if err = png.Encode(&buff, image); err != nil {
		return
	}
	pngBytes = buff.Bytes()
	return
}

type clipToBlob func(*libaural2.AudioClip, libaural2.VocabName) ([]byte, error)

func serve(
	db boltstore.DB,
	seqInferenceFuncs map[libaural2.VocabName]func(*tf.Tensor) (*tf.Tensor, error),
	namesPrs map[libaural2.VocabName]bool,
) {
	defer db.Close()
	makeServeAudioDerivedBlob := makeMakeServeAudioDerivedBlob(namesPrs)
	// make some function that take *libaural2.AudioClip and return a []byte
	computeWav, err := makeAddRIFF()
	if err != nil {
		logger.Fatalln(err)
	}
	renderMFCC, err := makeRenderMFCC()
	if err != nil {
		logger.Fatalln(err)
	}
	renderSpectrogram, err := makeRenderSpectrogram()
	if err != nil {
		logger.Fatalln(err)
	}
	renderProbs, err := makeRenderProbs(seqInferenceFuncs)
	if err != nil {
		logger.Fatalln(err)
	}
	renderArgmaxedStates, err := makeRenderArgmaxedStates(seqInferenceFuncs)
	if err != nil {
		logger.Fatalln(err)
	}
	serializeLabelSet := func(labelSet libaural2.LabelSet) (serialized []byte, err error) {
		serialized, err = labelSet.Serialize()
		return
	}
	r := mux.NewRouter()
	// with makeServeAudioDerivedBlob(), we convert the blob conversion func into a request handler.
	r.HandleFunc("/images/spectrogram/{vocab}/{sampleID}.jpeg", makeServeAudioDerivedBlob(renderSpectrogram))
	r.HandleFunc("/images/mfcc/{vocab}/{sampleID}.jpeg", makeServeAudioDerivedBlob(renderMFCC))
	r.HandleFunc("/images/probs/{vocab}/{sampleID}.jpeg", makeServeAudioDerivedBlob(renderProbs))
	r.HandleFunc("/images/argmax/{vocab}/{sampleID}.png", makeServeAudioDerivedBlob(renderArgmaxedStates))
	r.HandleFunc("/images/labelset/{vocab}/{sampleID}.png", makeServeLabelsSetDerivedBlob(namesPrs, db.GetLabelSet, renderColorLabelSetImage))
	r.HandleFunc("/audio/{sampleID}.wav", makeServeAudioDerivedBlob(computeWav))
	r.HandleFunc("/tagui/{vocab}/{sampleID}", makeServeTagUI(namesPrs))
	r.HandleFunc("/{vocab}/index", makeServeIndex(db.ListAudioClips, namesPrs))
	r.HandleFunc("/trainingdata/{vocab}.pb", makeServeTrainingDataGraphdef(db.GetAllLabelSets, namesPrs))
	r.HandleFunc("/labelsset/{vocab}/{sampleID}", makeWriteLabelsSet(db.PutLabelSet, namesPrs)).Methods("POST")
	r.HandleFunc("/labelsset/{vocab}/{sampleID}", makeServeLabelsSetDerivedBlob(namesPrs, db.GetLabelSet, serializeLabelSet)).Methods("GET")
	r.HandleFunc("/sample/upload", makeSampleHandler(db.PutClipID))
	fs := http.FileServer(http.Dir("static"))
	http.Handle("/static/", http.StripPrefix("/static/", fs))
	http.Handle("/", r)
	http.ListenAndServe(":48125", nil)
}
