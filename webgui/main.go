package main

import (
	"encoding/base32"
	"fmt"
	"strconv"
	"time"

	"image/color"

	la "github.ibm.com/Blue-Horizon/aural2/libaural2"
	"github.ibm.com/Blue-Horizon/aural2/vsh/emotion"
	"github.ibm.com/Blue-Horizon/aural2/vsh/intent"
	"github.ibm.com/Blue-Horizon/aural2/vsh/speaker"
	"github.ibm.com/Blue-Horizon/aural2/vsh/word"
	"honnef.co/go/js/dom"
	"honnef.co/go/js/xhr"
)

var vocabs = map[string]*la.Vocabulary{
	"word":    &word.Vocabulary,
	"intent":  &intent.Vocabulary,
	"speaker": &speaker.Vocabulary,
	"emotion": &emotion.Vocabulary,
}

func colorToCSSstring(colorObj color.Color) (colorString string) {
	r, g, b, _ := colorObj.RGBA()
	colorString = "rgb(" + strconv.Itoa(int(r/256)) + ", " + strconv.Itoa(int(g/256)) + ", " + strconv.Itoa(int(b/256)) + ")"
	return
}

// setCurser takes the fraction of the audio file duration as a float betwene 0 and 1.
func setCurser(position float64) {
	if position > 1 {
		print("position:", position)
		return
	}
	curser := dom.GetWindow().Document().GetElementByID("curser").(*dom.HTMLDivElement)
	curser.Style().Set("left", strconv.FormatFloat(position*100, 'f', 8, 64)+"%")
}

func createLabelMarker(label la.Label) (setEnd func(float64)) {
	d := dom.GetWindow().Document()
	labelsContainer := d.GetElementByID("labels").(*dom.HTMLDivElement)
	labelDiv := d.CreateElement("div").(*dom.HTMLDivElement)
	left := label.Start / float64(la.Duration)
	labelDiv.Style().Set("left", strconv.FormatFloat(left*100, 'f', 8, 64)+"%")
	labelDiv.SetClass("label")
	labelDiv.SetInnerHTML("<p class='state_label'>" + vocab.Names[label.State] + "</p>")
	labelDiv.Style().SetProperty("background-color", colorToCSSstring(label.State), "")
	labelsContainer.AppendChild(labelDiv)
	labelDiv.AddEventListener("click", false, func(arg3 dom.Event) {
		labelsContainer.RemoveChild(labelDiv)
		for i, otherLabel := range labelsSet.Labels {
			if otherLabel.Start == label.Start {
				labelsSet.Labels = append(labelsSet.Labels[:i], labelsSet.Labels[i+1:]...)
			}
		}
	})
	setEnd = func(end float64) {
		width := (end - label.Start) / float64(la.Duration)
		labelDiv.Style().Set("width", strconv.FormatFloat(width*100, 'f', 8, 64)+"%")
	}
	if label.End != 0 {
		setEnd(label.End)
	}
	return
}

func postLabelsSet(labels la.LabelSet) (err error) {
	print("posting")
	serialised, err := labels.Serialize()
	if err != nil {
		return
	}
	_, err = xhr.Send("POST", "/labelsset/"+string(vocab.Name)+"/"+clipID.FSsafeString(), serialised)
	if err != nil {
		print(err)
		return
	}
	return
}

func getLabelsSet() {
	resp, err := xhr.Send("GET", "/labelsset/"+string(vocab.Name)+"/"+clipID.FSsafeString(), nil)
	if err != nil {
		print(err)
		return
	}
	labelsSet, err = la.DeserializeLabelSet(resp)
	if err != nil {
		print(err)
		return
	}
	for _, label := range labelsSet.Labels {
		createLabelMarker(label)
	}
	return
}

func reloadProbs() {
	d := dom.GetWindow().Document()
	probs := d.GetElementByID("probs").(*dom.HTMLImageElement)
	probsSrc := probs.Src
	argmax := d.GetElementByID("argmax").(*dom.HTMLImageElement)
	argmaxSrc := argmax.Src
	for {
		time.Sleep(500 * time.Millisecond)
		probs.Set("src", probsSrc+"?"+time.Now().String())
		argmax.Set("src", argmaxSrc+"?"+time.Now().String())
		print("loading")
	}
}

var clipID la.ClipID
var vocab *la.Vocabulary
var labelsSet la.LabelSet

func start() {
	go reloadProbs()
	var err error
	serialisedData := dom.GetWindow().Document().GetElementByID("data").(*dom.HTMLDivElement).Dataset()
	clipIDbytes, err := base32.StdEncoding.DecodeString(serialisedData["b32sampleid"])
	if err != nil {
		panic(err)
	}
	if len(clipIDbytes) != 32 {
		panic("clip id is wrong len")
	}

	copy(clipID[:], clipIDbytes)
	print(clipID.String())

	vocabName := dom.GetWindow().Document().GetElementByID("data").(*dom.HTMLDivElement).Dataset()["vocabname"]
	vocab = vocabs[vocabName]
	if vocab == nil {
		print("can't find vocab name", vocabName)
	}
	go getLabelsSet()
	w := dom.GetWindow()
	d := w.Document()
	var currentlyDepressedKey string
	var curLabel la.Label
	var setEnd func(float64)
	audio := d.GetElementByID("audio").(*dom.HTMLAudioElement)
	//audio.Play()
	go func() {
		for {
			time.Sleep(time.Millisecond * 10)
			if !audio.Paused {
				currentTime := audio.Get("currentTime").Float()
				frac := currentTime / float64(la.Duration)
				setCurser(frac)
				if setEnd != nil {
					setEnd(currentTime)
				}
			}
		}
	}()
	audio.AddEventListener("timeupdate", false, func(event dom.Event) {
		currentTime := audio.Get("currentTime").Float()
		frac := currentTime / float64(la.Duration)
		setCurser(frac)
		if setEnd != nil {
			setEnd(currentTime)
		}
		//timelinePos += -0.1
		//setTimelinePos(timelinePos)
	})
	w.AddEventListener("keyup", false, func(event dom.Event) {
		key := event.(*dom.KeyboardEvent).Key
		if key == currentlyDepressedKey {
			curLabel.End = audio.Get("currentTime").Float()
			setEnd = nil
			labelsSet.Labels = append(labelsSet.Labels, curLabel)
			currentlyDepressedKey = ""
		}
	})
	w.AddEventListener("keydown", false, func(event dom.Event) {
		ke := event.(*dom.KeyboardEvent)
		state, prs := vocab.KeyMapping[ke.Key]
		if prs {
			if ke.Key != currentlyDepressedKey {
				currentlyDepressedKey = ke.Key
				label := la.Label{
					State: state,
					Start: audio.Get("currentTime").Float(),
				}
				curLabel = label
				setEnd = createLabelMarker(curLabel)
			}
		}
		//if ke.Key == "s" {
		//	go postLabelsSet(labelsSet)
		//}
		if ke.Key == "Delete" {
			labelsSet.Labels = []la.Label{}
			labelsElm := d.GetElementByID("labels").(*dom.HTMLDivElement)
			for labelsElm.Call("hasChildNodes").Bool() {
				labelsElm.Call("removeChild", labelsElm.Get("lastChild"))
			}
		}
		if ke.Key == "ArrowRight" {
			currentTime := audio.Get("currentTime").Float()
			newTime := currentTime + 0.04
			audio.Set("currentTime", strconv.FormatFloat(newTime, 'f', 8, 64))
		}
		if ke.Key == "ArrowLeft" {
			currentTime := audio.Get("currentTime").Float()
			newTime := currentTime - 0.04
			audio.Set("currentTime", strconv.FormatFloat(newTime, 'f', 8, 64))
		}
		if ke.Key == "ArrowDown" {
			currentTime := audio.Get("currentTime").Float()
			newTime := currentTime - 1
			audio.Set("currentTime", strconv.FormatFloat(newTime, 'f', 8, 64))
		}
		if ke.Key == "ArrowUp" {
			currentTime := audio.Get("currentTime").Float()
			newTime := currentTime + 1
			audio.Set("currentTime", strconv.FormatFloat(newTime, 'f', 8, 64))
		}
		if ke.Key == " " {
			if audio.Paused {
				print("playing")
				audio.Play()
			} else {
				audio.Pause()
			}
		}
	})
}

func main() {
	fmt.Println("Audio vis GUI version 0.1.3")
	w := dom.GetWindow()
	w.AddEventListener("DOMContentLoaded", true, func(event dom.Event) {
		go start()
	})
	w.AddEventListener("beforeunload", true, func(event dom.Event) {
		fmt.Println("unloading")
		postLabelsSet(labelsSet)
	})
}
