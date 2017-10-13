package main

import (
	"encoding/base32"
	"fmt"
	"strconv"
	"time"

	"image/color"

	la "github.ibm.com/Blue-Horizon/aural2/libaural2"
	"honnef.co/go/js/dom"
	"honnef.co/go/js/xhr"
)

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
	labelDiv.SetInnerHTML("<p class='cmd_label'>" + label.Cmd.String() + "</p>")
	labelDiv.Style().SetProperty("background-color", colorToCSSstring(label.Cmd), "")
	labelsContainer.AppendChild(labelDiv)

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
	serialised, err := labels.Serialize()
	if err != nil {
		return
	}
	_, err = xhr.Send("POST", "../labelsset/"+clipID.FSsafeString(), serialised)
	if err != nil {
		print(err)
		return
	}
	return
}

func getLabelsSet() {
	resp, err := xhr.Send("GET", "../labelsset/"+clipID.FSsafeString(), nil)
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

var clipID la.ClipID
var labelsSet la.LabelSet

var keyToCmd = map[string]la.Cmd{
	"u": la.Unknown,
	"y": la.Yes,
	"n": la.No,
	"0": la.Zero,
	"1": la.One,
	"2": la.Two,
	"3": la.Three,
	"4": la.Four,
	"5": la.Five,
	"6": la.Six,
	"7": la.Seven,
	"8": la.Eight,
	"9": la.Nine,
	"c": la.CtrlC,
	"t": la.True,
	"f": la.False,
}

func start() {
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
		for key, cmd := range keyToCmd {
			// if it is a cmd,
			if key == ke.Key && key != currentlyDepressedKey {
				currentlyDepressedKey = key
				label := la.Label{
					Cmd:   cmd,
					Start: audio.Get("currentTime").Float(),
				}
				curLabel = label
				setEnd = createLabelMarker(curLabel)
				continue
			}
		}
		if ke.Key == "s" {
			go postLabelsSet(labelsSet)
		}
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
				audio.Play()
			} else {
				audio.Pause()
			}
		}
	})
}

func main() {
	fmt.Println("Audio vis GUI version 0.1.1")
	w := dom.GetWindow()
	w.AddEventListener("DOMContentLoaded", true, func(event dom.Event) {
		go start()
	})
	w.AddEventListener("beforeunload", true, func(event dom.Event) {
		fmt.Println("unloading")
		postLabelsSet(labelsSet)
	})
}
