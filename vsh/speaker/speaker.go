package speaker

import (
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
)

// Vocabulary is the set of voices who talk to the machine
var Vocabulary = libaural2.Vocabulary{
	Name: "speaker",
	Size: 10,
	Names: map[libaural2.State]string{
		Nil:             "Nil",
		Isaac:           "Isaac",
		Chris:           "Chris",
		Igor:            "Igor",
		Egan:            "Egan",
		Mosquito:        "Glen",
		GoogleAssistant: "GoogleAssistant",
		Alexa:           "Alexa",
	},
	Hue: map[libaural2.State]float64{
		Nil: 0,
	},
	KeyMapping: map[string]libaural2.State{
		"n": Nil,
		"i": Isaac,
		"c": Chris,
    "o": Igor,
    "e": Egan,
    "m": Mosquito,
    "g": GoogleAssistant,
    "a": Alexa,
	},
}

// Standard people
const (
	Nil libaural2.State = iota
	Unknown
	Isaac
	Chris
	Igor
	Egan
  Mosquito
	GoogleAssistant
	Alexa
)
