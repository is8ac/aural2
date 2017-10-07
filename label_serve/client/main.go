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
		fmt.Println("position:", position)
		return
	}
	curser := dom.GetWindow().Document().GetElementByID("curser").(*dom.HTMLDivElement)
	curser.Style().Set("left", strconv.FormatFloat(position*100, 'f', 8, 64)+"%")
}

func addLabel(label la.Label) {
	d := dom.GetWindow().Document()
	labelsContainer := d.GetElementByID("labels").(*dom.HTMLDivElement)
	labelDiv := d.CreateElement("div").(*dom.HTMLDivElement)
	position := label.Time / 10
	fmt.Println(position)
	labelDiv.Style().Set("left", strconv.FormatFloat(position*100, 'f', 8, 64)+"%")
	labelDiv.SetClass("label")
	labelDiv.SetInnerHTML("<p class='cmd_label'>" + label.Cmd.String() + "</p>")
	labelDiv.Style().SetProperty("background-color", colorToCSSstring(label.Cmd), "")
	labelsContainer.AppendChild(labelDiv)
}

func postLabelsSet(labels la.LabelSet) (err error) {
	serialised, err := labels.Serialize()
	if err != nil {
		return
	}
	_, err = xhr.Send("POST", "../labelsset/"+clipID.FSsafeString(), serialised)
	if err != nil {
		fmt.Println(err)
		return
	}
	return
}

func getLabelsSet() {
	resp, err := xhr.Send("GET", "../labelsset/"+clipID.FSsafeString(), nil)
	if err != nil {
		fmt.Println(err)
		return
	}
	labelsSet, err = la.DeserializeLabelSet(resp)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, label := range labelsSet.Labels {
		fmt.Println(label.Cmd)
		addLabel(label)
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
		fmt.Println(err)
		panic(err)
	}
	if len(clipIDbytes) != 32 {
		fmt.Println("clip id is bad")
		panic("clip id is wrong len")
	}

	copy(clipID[:], clipIDbytes)
	fmt.Println(clipID)
	go getLabelsSet()
	w := dom.GetWindow()
	d := w.Document()
	audio := d.GetElementByID("audio").(*dom.HTMLAudioElement)
	//audio.Play()
	go func() {
		for {
			time.Sleep(time.Millisecond * 100)
			if !audio.Paused {
				currentTime := audio.Get("currentTime").Float()
				frac := currentTime / float64(la.Duration)
				setCurser(frac)
			}
		}
	}()
	audio.AddEventListener("timeupdate", false, func(event dom.Event) {
		currentTime := audio.Get("currentTime").Float()
		frac := currentTime / float64(la.Duration)
		setCurser(frac)
		//timelinePos += -0.1
		//setTimelinePos(timelinePos)
	})
	w.AddEventListener("keydown", false, func(event dom.Event) {
		ke := event.(*dom.KeyboardEvent)
		for key, cmd := range keyToCmd {
			if key == ke.Key {
				label := la.Label{
					Cmd:  cmd,
					Time: audio.Get("currentTime").Float(),
				}
				addLabel(label)
				labelsSet.Labels = append(labelsSet.Labels, label)
				fmt.Println(label)
			}
		}
		if ke.Key == "s" {
			go postLabelsSet(labelsSet)
		}
		if ke.Key == "ArrowRight" {
			currentTime := audio.Get("currentTime").Float()
			newTime := currentTime + 0.04
			audio.Set("currentTime", strconv.FormatFloat(newTime, 'f', 8, 64))
			fmt.Println("newTime:", newTime)
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
