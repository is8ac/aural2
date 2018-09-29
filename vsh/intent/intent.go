package intent

import (
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
)

// Vocabulary is the set of actions the machine can take at the users request.
var Vocabulary = libaural2.Vocabulary{
	Name: "intent",
	Size: 20,
	Names: map[libaural2.State]string{
		Nil:            "Nil",
		PlayMusic:      "PlayMusic",
		PauseMusic:     "PauseMusic",
		SkipSong:       "SkipSong",
		SayTime:        "SayTime",
		SayVersion:     "SayVersion",
		SayTemperature: "SayTemperature",
		SayAirQuality:  "SayAirQuality",
		TurnOff:        "TurnOff",
		EasterEgg:      "EasterEgg",
		DoIt:           "DoIt",
		DontDoIt:       "Don'tDoIt",
		UploadClip:     "UploadClip",
		ShutDown:       "ShutDown",
		Next:           "Next",
		Previous:       "Previous",
	},
	Hue: map[libaural2.State]float64{
		Nil: 0,
	},
	KeyMapping: map[string]libaural2.State{
		"n": Nil,
		"p": PlayMusic,
		"a": PauseMusic,
		"s": SkipSong,
		"t": SayTime,
		"v": SayVersion,
		"e": SayTemperature,
		"q": SayAirQuality,
		"c": TurnOff,
		"g": EasterEgg,
		"y": DoIt,
		"o": DontDoIt,
		"u": UploadClip,
		"d": ShutDown,
		".": Next,
		",": Previous,
	},
}

// Things vsh can do
const (
	Nil            libaural2.State = iota // the user doesn't want anything.
	PlayMusic                             // play the music
	PauseMusic                            // stop playing the music
	SkipSong                              // skip to next song in playlist
	SayTime                               // tell the user the current time
	SayVersion                            // tell the user the software version
	SayTemperature                        // tell the user the temp
	SayAirQuality
	TurnOff    // turn off
	EasterEgg  // easter egg
	DoIt       // the user agrees with the proposed action
	DontDoIt   // Don't do whatever it was that you asked the user if you could do.
	UploadClip // Upload the last 10 seconds of audio.
	ShutDown
	Next
	Previous
)
