package main

import (
	"errors"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strconv"

	"html/template"

	"encoding/base32"

	"github.com/gorilla/mux"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
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
	copy(clipID[:], fileHash)
	return
}

func makeServeMFCC() func(http.ResponseWriter, *http.Request) {
	renderImage, err := makeComputeMFCC()
	if err != nil {
		logger.Fatal(err)
	}
	return func(w http.ResponseWriter, r *http.Request) {
		audioIDstring := mux.Vars(r)["sampleID"]
		clipID, err := parseURLvar(audioIDstring)
		audioClip, err := getAudioClipFromFS(clipID)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		imageBytes, err := renderImage(audioClip)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}

		w.Write([]byte(imageBytes))
	}
}

func makeServeSpectrogram() func(http.ResponseWriter, *http.Request) {
	renderImage, err := makeComputeSpectrogram()
	if err != nil {
		logger.Fatalln(err)
	}
	return func(w http.ResponseWriter, r *http.Request) {
		audioIDstring := mux.Vars(r)["sampleID"]
		clipID, err := parseURLvar(audioIDstring)
		audioClip, err := getAudioClipFromFS(clipID)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		imageBytes, err := renderImage(audioClip)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		w.Write([]byte(imageBytes))
	}
}

func makeServeWav() func(http.ResponseWriter, *http.Request) {
	addRiff, err := makeAddRIFF()
	if err != nil {
		logger.Fatalln(err)
	}
	return func(w http.ResponseWriter, r *http.Request) {
		audioIDstring := mux.Vars(r)["sampleID"]
		clipID, err := parseURLvar(audioIDstring)
		audioClip, err := getAudioClipFromFS(clipID)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		wavBytes := addRiff(audioClip)
		logger.Println(len(wavBytes))
		w.Header().Set("Content-Type", "audio/x-wav")
		w.Header().Set("Content-Length", strconv.Itoa(len(wavBytes)))
		w.Header().Set("Accept-Ranges", "bytes")
		w.Write(wavBytes)
	}
}

func makeWriteLabelsSet(put func(libaural2.LabelSet) error) func(http.ResponseWriter, *http.Request) {
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
		if sampleID != labelsSet.ID {
			logger.Println(sampleID, "!=", labelsSet.ID)
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

func makeServeLabelsSet(getLabelsSet func(libaural2.ClipID) (libaural2.LabelSet, error)) func(http.ResponseWriter, *http.Request) {
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
		serialized, err := labelSet.Serialize()
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		w.Write(serialized)
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
			UrbitSampleID: urbitname.Encode(hash[:4]),
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
	return func(w http.ResponseWriter, r *http.Request) {
		rawBytes, err := ioutil.ReadAll(r.Body)
		if err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		if len(rawBytes) != libaural2.AudioClipLen {
			http.Error(w, "wrong length", http.StatusBadRequest)
			return
		}
		var audioClip libaural2.AudioClip
		copy(audioClip[:], rawBytes) // convert the slice of bytes to an array of bytes.
		if err != nil {
			http.Error(w, "malformed audio", http.StatusBadRequest)
			return
		}
		id := audioClip.ID()
		if err = ioutil.WriteFile("audio/"+id.FSsafeString()+".raw", rawBytes, 0644); err != nil {
			logger.Println(err)
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
		w.Write([]byte(id.FSsafeString()))
		return
	}
}

var logger = log.New(os.Stdout, "ts_vis: ", log.Lshortfile|log.LUTC|log.Ltime|log.Ldate)

const version = "0.2.1"

func main() {
	logger.Println("Audio viz server version " + version)
	put, get, close, err := initDB()
	if err != nil {
		logger.Fatalln(err)
	}
	defer close()
	r := mux.NewRouter()
	r.HandleFunc("/images/spectrogram/{sampleID}", makeServeSpectrogram())
	r.HandleFunc("/audio/{sampleID}.wav", makeServeWav())
	r.HandleFunc("/images/mfcc/{sampleID}", makeServeMFCC())
	r.HandleFunc("/tagui/{sampleID}", makeServeTagUI())
	r.HandleFunc("/labelsset/{sampleID}", makeWriteLabelsSet(put)).Methods("POST")
	r.HandleFunc("/labelsset/{sampleID}", makeServeLabelsSet(get)).Methods("GET")
	r.HandleFunc("/sample/upload", makeSampleHandler())
	fs := http.FileServer(http.Dir("static"))
	http.Handle("/static/", http.StripPrefix("/static/", fs))
	http.Handle("/", r)
	http.ListenAndServe(":48125", nil)
}
