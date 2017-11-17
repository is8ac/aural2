package word

import (
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
)

// Vocabulary is the set of words the user can say.
var Vocabulary = libaural2.Vocabulary{
	Name: "word",
	Size: 50,
	Names: map[libaural2.State]string{
		Nil:         "Nil",
		Unknown:     "Unknown",
		Yes:         "Yes",
		No:          "No",
		True:        "True",
		False:       "False",
		CtrlC:       "CtrlC",
		Sudo:        "Sudo",
		Mpc:         "Mpc",
		Play:        "Play",
		Pause:       "Pause",
		Stop:        "Stop",
		OK:          "OK",
		Set:         "Set",
		Is:          "Is",
		What:        "What",
		Same:        "Same",
		Different:   "Different",
		When:        "When",
		Who:         "Who",
		Where:       "Where",
		OKgoogle:    "OKgoogle",
		Alexa:       "Alexa",
		Music:       "Music",
		Genre:       "Genre",
		Classical:   "Classical",
		Plainsong:   "Plainsong",
		Vocaloid:    "Vocaloid",
		Reggae:      "Reggae",
		Rock:        "Rock",
		RockAndRoll: "RockAndRoll",
		Rap:         "Rap",
		HipHop:      "HipHop",
		Blues:       "Blues",
		Shakuhachi:  "Shakuhachi",
		Grep:        "Grep",
		Yotsugi:     "Yotsugi",
		Emo:         "Emo",
		GangstaRap:  "GangstaRap",
		Punk:        "Punk",
		Alternative: "Alternative",
		Welcome:     "Welcome",
		Hello:       "Hello",
	},
	Hue: map[libaural2.State]float64{
		Stop: 0.1,
	},
	KeyMapping: map[string]libaural2.State{
		"?": Unknown,
		"y": Yes,
		"n": No,
		"t": True,
		"f": False,
		"C": CtrlC,
		"S": Sudo,
		"M": Mpc,
		">": Play,
		"<": Pause,
		"|": Stop,
		"O": OK,
		"s": Set,
		"i": Is,
		"h": What,
		"=": Same,
		";": Different,
		"T": When,
		"w": Who,
		"L": Where,
		"G": OKgoogle,
		"A": Alexa,
		"m": Music,
		"g": Genre,
		"c": Classical,
		"p": Plainsong,
		"v": Vocaloid,
		"r": Reggae,
		"!": Rock,
		"N": RockAndRoll,
		"$": Rap,
		"d": HipHop,
		"b": Blues,
		"j": Shakuhachi,
		"x": Grep,
		"Y": Yotsugi,
		"e": Emo,
		"4": GangstaRap,
		"P": Punk,
		"a": Alternative,
		"W": Welcome,
		"H": Hello,
	},
}

// Standard Words
const (
	Nil libaural2.State = iota
	Unknown
	Yes
	No
	True
	False
	CtrlC
	Sudo
	Mpc
	Play
	Pause
	Stop
	OK
	Set
	Is
	What
	Same
	Different
	When
	Who
	Where
	OKgoogle
	Alexa
	Music
	Genre
	Classical
	Plainsong
	Vocaloid
	Reggae
	Rock
	RockAndRoll
	Rap
	HipHop
	Blues
	Shakuhachi
	Yotsugi
	Grep
	Emo
	GangstaRap
	Punk
	Alternative
	Welcome
	Hello
)
