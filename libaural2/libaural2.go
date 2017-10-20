// Package libaural2 provides libs share betwene edge, server and browser.
package libaural2

import (
	"bytes"
	"crypto/sha256"
	"encoding/base32"
	"encoding/binary"
	"encoding/gob"
	"fmt"

	"github.com/lucasb-eyer/go-colorful"
	"github.ibm.com/ifleonar/mu/urbitname"
)

// Duration of audio clip in seconds
const Duration int = 10

// SampleRate of audio
const SampleRate int = 16000

// StrideWidth is the number of samples in one stride
const StrideWidth int = 512

// SamplePerClip is the number of samples in each clip
const SamplePerClip int = SampleRate * Duration

// StridesPerClip is the number of strides per clip
const StridesPerClip int = SamplePerClip / StrideWidth

// SeqLen is the length of sequences to be feed to the LSTM for training.
const SeqLen int = 100

// AudioClipLen is the number of bytes in one audio clip
const AudioClipLen int = SamplePerClip * 2

// InputSize is the length of the input vector, currently one MFCC
const InputSize int = 13

// OutputSize is the number of commands. Increase when adding new Cmds!
const OutputSize int = 40

// BatchSize is the size of the one batch
const BatchSize int = 10

// CmdList is a list of Cmds
type CmdList [StridesPerClip]Cmd

// Input is the one input to the LSTM
type Input [InputSize]float32

// InputSet is the set of inputs for one clip.
type InputSet [StridesPerClip]Input

// Output is one output, the softmax array of Cmds.
type Output [OutputSize]float32

// OutputSet is the set of outputs for one clip.
type OutputSet [StridesPerClip]Output

// Serialize converts an outputSet to a []bytes
func (outputSet *OutputSet) Serialize() (serialized []byte) {
	buf := new(bytes.Buffer)
	var count int
	for _, output := range outputSet {
		for _, cmdVal := range output {
			binary.Write(buf, &binary.LittleEndian, cmdVal)
		}
	}
	fmt.Println("count", count*4)
	serialized = buf.Bytes()
	return
}

// AudioClip stores a `Duration` second clip of int16 raw audio
type AudioClip [AudioClipLen]byte

// ID computes the hash of the audio clip
func (rawBytes *AudioClip) ID() ClipID {
	return sha256.Sum256(rawBytes[:])
}

// ClipID is the hash of a clip of raw audio
type ClipID [32]byte

// FSsafeString returns an encoding of the ClipID safe to filesystems and URLs.
func (hash ClipID) FSsafeString() string {
	return base32.StdEncoding.EncodeToString(hash[:])
}

func (hash ClipID) String() string {
	return urbitname.Encode(hash[0:4])
}

// Cmd is one Cmd
type Cmd int

func (cmd Cmd) String() string {
	return CmdToString[cmd]
}

// ToOutput convert a Cmd to the onehot encoded output
func (cmd Cmd) ToOutput() (output Output) {
	output[cmd] = 1
	return
}

// Hue returns the hue as a float64
func (cmd Cmd) Hue() (hue float64) {
	hash := sha256.Sum256([]byte{uint8(cmd)})
	hue = float64(hash[0]) * float64(hash[1]) / (255 * 255) * 360
	return
}

// RGBA implements the color.Color interface
func (cmd Cmd) RGBA() (r, g, b, a uint32) {
	classColor := colorful.Hsv(cmd.Hue(), 1, 1)
	return classColor.RGBA()
}

// Label is one period of time.
type Label struct {
	Cmd   Cmd
	Start float64 // the duration since the start of the clip.
	End   float64
}

// LabelSet is the set of labels for one Clip
type LabelSet struct {
	ID     ClipID
	Labels []Label
}

// ToCmdIDArray converts the labelSet to a slice of Cmds IDs
func (labels *LabelSet) ToCmdIDArray() (cmdArray [StridesPerClip]int32) {
	for i := range cmdArray {
		loc := float64(i) / float64(StridesPerClip) * float64(Duration)
		for _, label := range labels.Labels {
			if loc > label.Start && loc < label.End {
				cmdArray[i] = int32(label.Cmd)
				continue
			}
		}
	}
	return
}

// ToCmdArray converts the labelSet to a slice of Cmds
func (labels *LabelSet) ToCmdArray() (cmdArray [StridesPerClip]Cmd) {
	for i := range cmdArray {
		loc := float64(i) / float64(StridesPerClip) * float64(Duration)
		for _, label := range labels.Labels {
			if loc > label.Start && loc < label.End {
				cmdArray[i] = label.Cmd
				continue
			}
		}
	}
	return
}

// IsGood returns true iff the labelsSet contains no overlaps or other bad things. Executes in O(n2) time.
func (labels *LabelSet) IsGood() bool {
	for _, label := range labels.Labels {
		if label.Start < 0 {
			return false
		}
		if label.End > float64(Duration) {
			return false
		}
		for _, otherLabel := range labels.Labels {
			if label.Start > otherLabel.Start && label.Start < otherLabel.End {
				return false
			}
			if label.End > otherLabel.Start && label.End < otherLabel.End {
				return false
			}
		}
	}
	return true
}

// Serialize converts a LabelSet to []byte
func (labels *LabelSet) Serialize() (serialized []byte, err error) {
	buf := bytes.Buffer{}
	gobEnc := gob.NewEncoder(&buf)
	if err = gobEnc.Encode(labels); err != nil {
		return
	}
	serialized = buf.Bytes()
	return
}

// DeserializeLabelSet converts a []byte back into a LabelSet.
func DeserializeLabelSet(serialized []byte) (labelSet LabelSet, err error) {
	dec := gob.NewDecoder(bytes.NewReader(serialized))
	if err = dec.Decode(&labelSet); err != nil {
		return
	}
	return
}

// Standard Cmds
const (
	Nil Cmd = iota
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
)

// CmdToString converts a cmd to the cmds name.
var CmdToString = map[Cmd]string{
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
}

// GenFakeLabelSet creates a fake LabelSet for testing.
func GenFakeLabelSet() (output LabelSet) {
	output.Labels = []Label{
		Label{
			Cmd:   Nil,
			Start: 0,
			End:   1,
		},
		Label{
			Cmd:   Yes,
			Start: 1,
			End:   2,
		},
		Label{
			Cmd:   No,
			Start: 2,
			End:   3,
		},
		Label{
			Cmd:   Nil,
			Start: 3,
			End:   4,
		},
		Label{
			Cmd:   Yes,
			Start: 4,
			End:   5,
		},
		Label{
			Cmd:   No,
			Start: 5,
			End:   6,
		},
		Label{
			Cmd:   Nil,
			Start: 6,
			End:   7,
		},
		Label{
			Cmd:   Yes,
			Start: 7,
			End:   8,
		},
		Label{
			Cmd:   No,
			Start: 8,
			End:   9,
		},
		Label{
			Cmd:   Nil,
			Start: 9,
			End:   10,
		},
	}
	return
}

//GenFakeInput produces fake a mfcc list exactly matching the given cmdIdArray
func GenFakeInput(cmds [StridesPerClip]int32) (fakeMFCCs [][]float32) {
	fakeMFCCs = make([][]float32, StridesPerClip)
	for i, cmd := range cmds {
		fakeMFCCs[i] = make([]float32, InputSize)
		if cmd == 0 {
			fakeMFCCs[i][0] = 1
		}
		if cmd == 1 {
			fakeMFCCs[i][1] = 1
		}
		if cmd == 2 {
			fakeMFCCs[i][2] = 1
		}
		if cmd == 3 {
			fakeMFCCs[i][3] = 1
		}
	}
	return
}
