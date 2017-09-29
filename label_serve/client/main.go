package main

import (
	"fmt"
	"honnef.co/go/js/dom"
	"strconv"
)

func setCurser(position float64) {
	curser := dom.GetWindow().Document().GetElementByID("curser").(*dom.HTMLDivElement)
	curser.Style().Set("left", strconv.FormatFloat(position*100, 'f', 8, 64)+"%")
}

func setTimelinePos(position float64) {
	curser := dom.GetWindow().Document().GetElementByID("timeline").(*dom.HTMLDivElement)
	curser.Style().Set("left", strconv.FormatFloat(position*100, 'f', 8, 64)+"%")
}


func start() {
	var zoomLevel float64 = 1
	var timelinePos float64
	w := dom.GetWindow()
	d := w.Document()
	timeline := d.GetElementByID("timeline").(*dom.HTMLDivElement)
	w.AddEventListener("wheel", false, func(event dom.Event) {
		we := event.(*dom.WheelEvent)
		delta := zoomLevel * 0.005
		zoomLevel = zoomLevel - delta*we.DeltaY
		if zoomLevel < 1 {
			zoomLevel = 1
		}
		if zoomLevel > 20 {
			zoomLevel = 20
		}
		fmt.Println(zoomLevel, delta)
		timeline.Style().Set("width", strconv.FormatFloat(zoomLevel*100, 'f', 8, 64)+"%")
	})
	audio := d.GetElementByID("audio").(*dom.HTMLAudioElement)
	audio.Pause()
	audio.AddEventListener("timeupdate", false, func(event dom.Event) {
		duration := audio.Get("duration").Float()
		currentTime := audio.Get("currentTime").Float()
		frac := currentTime / duration
		setCurser(frac)
		timelinePos += -0.1
		setTimelinePos(timelinePos)
	})
	w.AddEventListener("keydown", false, func(event dom.Event) {
		ke := event.(*dom.KeyboardEvent)
		if ke.Key == "ArrowRight" {
			currentTime := audio.Get("currentTime").Float()
			newTime := currentTime + 0.03
			audio.Set("currentTime", strconv.FormatFloat(newTime, 'f', 8, 64))
		}
		if ke.Key == "ArrowLeft" {
			currentTime := audio.Get("currentTime").Float()
			newTime := currentTime - 0.03
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
	fmt.Println("Audio vis GUI version 0.1.0")
	dom.GetWindow().AddEventListener("DOMContentLoaded", true, func(event dom.Event) {
		go start()
	})

}
