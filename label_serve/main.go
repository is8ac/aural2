package main

import (
	"bytes"
	"errors"
	"image"
	"image/png"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strconv"

	"html/template"

	"encoding/base32"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"

	"github.com/gorilla/mux"
	"github.ibm.com/Blue-Horizon/aural2/label_serve/boltstore"
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
func makeServeAudioDerivedBlob(clipToBlob func(*libaural2.AudioClip) ([]byte, error)) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
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
		blobBytes, err := clipToBlob(audioClip)
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

// makeServeLabelsSetDerivedBlob makes a handler func to serve a []byte derived from a labelSet.
func makeServeLabelsSetDerivedBlob(getLabelsSet func(libaural2.ClipID) (libaural2.LabelSet, error), setToBlob func(libaural2.LabelSet) ([]byte, error)) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		audioIDstring := mux.Vars(r)["sampleID"]
		clipID, err := parseURLvar(audioIDstring)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		labelSet, err := getLabelsSet(clipID)
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
func makeServeTrainingDataGraphdef(getAllLabelSets func() (map[libaural2.ClipID]libaural2.LabelSet, error)) func(http.ResponseWriter, *http.Request) {
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
		labelSets, err := getAllLabelSets()
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
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
			//input = libaural2.GenFakeInput(labelSet.ToCmdIDArray())
			inputs = append(inputs, input)
			outputs = append(outputs, labelSet.ToCmdIDArray())
			ids = append(ids, id)
		}
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

func makeWriteLabelsSet(put func(libaural2.LabelSet) error) func(http.ResponseWriter, *http.Request) {
	nilID := libaural2.ClipID{}
	return func(w http.ResponseWriter, r *http.Request) {
		audioIDstring := mux.Vars(r)["sampleID"]
		sampleID, err := parseURLvar(audioIDstring)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusBadRequest)
			return
		}
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

func makeServeIndex(list func() []libaural2.ClipID) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		var indexTemplate = template.Must(template.ParseFiles("templates/index.html"))
		ids := list()
		params := struct {
			IDs []libaural2.ClipID
		}{
			IDs: ids,
		}
		err := indexTemplate.Execute(w, params)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
	}
}

func makeServeTagUI() func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		audioIDstring := mux.Vars(r)["sampleID"]
		hash, err := parseURLvar(audioIDstring)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		var uiTemplate = template.Must(template.ParseFiles("templates/tag.html"))
		params := struct {
			Base32ID      string
			UrbitSampleID string
		}{
			UrbitSampleID: urbitname.Encode(hash[:4]),
			Base32ID:      base32.StdEncoding.EncodeToString(hash[:]),
		}
		err = uiTemplate.Execute(w, params)
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
	for x, cmd := range labelSet.ToCmdArray() {
		image.Set(x, 0, cmd)
	}
	buff := bytes.Buffer{}
	if err = png.Encode(&buff, image); err != nil {
		return
	}
	pngBytes = buff.Bytes()
	return
}

var logger = log.New(os.Stdout, "ts_vis: ", log.Lshortfile|log.LUTC|log.Ltime|log.Ldate)

const version = "0.3.1"

func main() {
	logger.Println("Audio viz server version " + version)
	db, err := boltstore.Init("trainingdata.db")
	if err != nil {
		logger.Fatalln(err)
	}
	defer db.Close()
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
	renderProbs, err := makeRenderProbs()
	if err != nil {
		logger.Fatalln(err)
	}
	serializeLabelSet := func(labelSet libaural2.LabelSet) (serialized []byte, err error) {
		serialized, err = labelSet.Serialize()
		return
	}
	r := mux.NewRouter()
	// with makeServeAudioDerivedBlob(), we convert the blob conversion func into a request handler.
	r.HandleFunc("/images/spectrogram/{sampleID}.jpeg", makeServeAudioDerivedBlob(renderSpectrogram))
	r.HandleFunc("/images/mfcc/{sampleID}.jpeg", makeServeAudioDerivedBlob(renderMFCC))
	r.HandleFunc("/images/probs/{sampleID}.jpeg", makeServeAudioDerivedBlob(renderProbs))
	r.HandleFunc("/images/labelset/{sampleID}.png", makeServeLabelsSetDerivedBlob(db.GetLabelSet, renderColorLabelSetImage))
	r.HandleFunc("/audio/{sampleID}.wav", makeServeAudioDerivedBlob(computeWav))
	r.HandleFunc("/tagui/{sampleID}", makeServeTagUI())
	r.HandleFunc("/index", makeServeIndex(db.ListAudioClips))
	r.HandleFunc("/trainingdata.pb", makeServeTrainingDataGraphdef(db.GetAllLabelSets))
	r.HandleFunc("/labelsset/{sampleID}", makeWriteLabelsSet(db.PutLabelSet)).Methods("POST")
	r.HandleFunc("/labelsset/{sampleID}", makeServeLabelsSetDerivedBlob(db.GetLabelSet, serializeLabelSet)).Methods("GET")
	r.HandleFunc("/sample/upload", makeSampleHandler(db.PutClipID))
	fs := http.FileServer(http.Dir("static"))
	http.Handle("/static/", http.StripPrefix("/static/", fs))
	http.Handle("/", r)
	http.ListenAndServe(":48125", nil)
}
