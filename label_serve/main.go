package main

import (
	"crypto/sha256"
	"errors"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"

	"html/template"

	"encoding/base32"

	"github.com/gorilla/mux"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"github.ibm.com/Blue-Horizon/aural2/tfutils"
	"repo.hovitos.engineering/MTN/wave/cloud/application/aural/urbitname"
)

func parseURLvar(urlVar string) (fileHash []byte, err error) {
	fileHash, err = base32.StdEncoding.DecodeString(urlVar)
	if err != nil {
		return
	}
	if len(fileHash) != 32 {
		err = errors.New("hash length must be 32 bytes")
		return
	}
	return
}

func makeServeMFCC() func(http.ResponseWriter, *http.Request) {
	s := op.NewScope()
	wavBytesPH, pcm := tfutils.ParseWavBytesToPCM(s)
	specgramOP, sampleRatePH := tfutils.ComputeMFCC(s.SubScope("mfcc"), pcm)
	jpegBytesOP := tfutils.RenderImage(s.SubScope("jpeg_bytes"), specgramOP)
	sampleRateTensor, err := tf.NewTensor(int32(44100))
	if err != nil {
		logger.Fatalln(err)
	}
	feeds := map[tf.Output]*tf.Tensor{sampleRatePH: sampleRateTensor}
	renderImage := tfutils.BytesToBytes(s, wavBytesPH, jpegBytesOP, feeds)

	return func(w http.ResponseWriter, r *http.Request) {
		audioIDstring := mux.Vars(r)["sampleID"]
		hash, err := parseURLvar(audioIDstring)
		_ = hash
		wavBytes, err := ioutil.ReadFile(path.Clean("audio/" + audioIDstring))
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusNotFound)
			return
		}
		imageBytes, err := renderImage(wavBytes)
		if err != nil {
			logger.Println(err)
			http.Error(w, "TF error", http.StatusInternalServerError)
			return
		}
		w.Write([]byte(imageBytes))
	}
}

func makeServeSpectrogram() func(http.ResponseWriter, *http.Request) {
	s := op.NewScope()
	wavBytesPH, pcm := tfutils.ParseWavBytesToPCM(s)
	specgramOP := tfutils.ComputeSpectrogram(s.SubScope("spectrogram"), pcm, 0, 350)
	specgramJpegBytesOP := tfutils.RenderImage(s.SubScope("jpeg_bytes"), specgramOP)
	feeds := map[tf.Output]*tf.Tensor{}
	renderImage := tfutils.BytesToBytes(s, wavBytesPH, specgramJpegBytesOP, feeds)

	return func(w http.ResponseWriter, r *http.Request) {
		audioIDstring := mux.Vars(r)["sampleID"]
		hash, err := parseURLvar(audioIDstring)
		_ = hash
		wavBytes, err := ioutil.ReadFile(path.Clean("audio/" + audioIDstring))
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusNotFound)
			return
		}
		imageBytes, err := renderImage(wavBytes)
		if err != nil {
			logger.Println(err)
			http.Error(w, "TF error", http.StatusInternalServerError)
			return
		}
		w.Write([]byte(imageBytes))
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
		var deviceTemplate = template.Must(template.ParseFiles("templates/tag.html"))
		params := struct {
			Base32ID      string
			UrbitSampleID string
		}{
			UrbitSampleID: urbitname.Encode(hash[:8]),
			Base32ID:      base32.StdEncoding.EncodeToString(hash[:]),
		}
		err = deviceTemplate.Execute(w, params)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
	}
}

func makeSampleHandler() func(http.ResponseWriter, *http.Request) {
	cleanWav, err := tfutils.MakeCleanWav(44100)
	if err != nil {
		logger.Fatalln(err)
	}
	return func(w http.ResponseWriter, r *http.Request) {
		wavBytes, err := ioutil.ReadAll(r.Body)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		if len(wavBytes) == 0 {
			http.Error(w, "empty", http.StatusBadRequest)
			return
		}
		if len(wavBytes) > 100000000 {
			http.Error(w, "too large", http.StatusRequestEntityTooLarge)
			return
		}
		cleanBytes, err := cleanWav(wavBytes)
		if err != nil {
			http.Error(w, "malformed audio", http.StatusBadRequest)
			return
		}
		hash := sha256.Sum256(cleanBytes)
		cleanedWavIDstring := base32.StdEncoding.EncodeToString(hash[:])
		if err = ioutil.WriteFile("audio/"+cleanedWavIDstring, cleanBytes, 0644); err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		w.Write([]byte(cleanedWavIDstring))
		return
	}
}

var logger = log.New(os.Stdout, "ts_vis: ", log.Lshortfile|log.LUTC|log.Ltime|log.Ldate)

const version = "0.1.0"

func main() {
	logger.Println("Audio viz server version " + version)

	r := mux.NewRouter()
	r.HandleFunc("/images/spectrogram/{sampleID}", makeServeSpectrogram())
	r.HandleFunc("/audio/{sampleID}", makeServeSpectrogram())
	r.HandleFunc("/images/mfcc/{sampleID}", makeServeMFCC())
	r.HandleFunc("/tagui/{sampleID}", makeServeTagUI())
	r.HandleFunc("/sample/upload", makeSampleHandler())
	fs := http.FileServer(http.Dir("static"))
	http.Handle("/static/", http.StripPrefix("/static/", fs))
	audiofs := http.FileServer(http.Dir("audio"))
	http.Handle("/audio/", http.StripPrefix("/audio/", audiofs))
	http.Handle("/", r)
	http.ListenAndServe(":48125", nil)
}
