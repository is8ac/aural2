package emotion

import (
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
)

// Vocabulary is the vocabulary of emotions
var Vocabulary = libaural2.Vocabulary{
	Name: "emotion",
	Size: 10,
	Names: map[libaural2.State]string{
		Nil:     "Nil",
		Neutral: "Neutral",
		Happy:   "Happy",
		Sad:     "Sad",
		Angry:   "Angry",
	},
	Hue: map[libaural2.State]float64{
		Nil:     0,
		Neutral: 0.1,
	},
	KeyMapping: map[string]libaural2.State{
		"n": Nil,
		"-": Neutral,
		"h": Happy,
		"s": Sad,
		"a": Angry,
	},
}

// emotional states of the user
const (
	Nil libaural2.State = iota
	Neutral
	Happy
	Sad
	Angry
)
