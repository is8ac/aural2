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
const StrideWidth int = 500

// SamplePerClip is the number of samples in each clip
const SamplePerClip int = SampleRate * Duration

// StridesPerClip is the number of strides per clip
const StridesPerClip int = SamplePerClip / StrideWidth

// AudioClipLen is the number of bytes in one audio clip
const AudioClipLen int = SamplePerClip * 2

// InputSize is the length of the input vector, currently one MFCC
const InputSize int = 13

// OutputSize is the number of commands. Increase when adding new Cmds!
const OutputSize int = 40

// Input is the one input to the LSTM
type Input [InputSize]float32

// InputSet is the set of inputs for one clip.
type InputSet [StridesPerClip]Input

// Output is one output, the onehot array of Cmds.
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

// RGBA implements the color.Color interface
func (cmd Cmd) RGBA() (r, g, b, a uint32) {
	hash := sha256.Sum256([]byte{uint8(cmd)})
	hue := float64(hash[0]) * float64(hash[1]) / (255 * 255) * 360
	classColor := colorful.Hsv(hue, 1, 1)
	return classColor.RGBA()
}

// Label is one label in time
type Label struct {
	Cmd  Cmd
	Time float64 // the duration since the start of the clip.
}

// LabelSet is the set of labels for one Clip
type LabelSet struct {
	ID     ClipID
	Labels []Label
}

// ToOutputSet converts the labelSet to an OutputSet
func (labels *LabelSet) ToOutputSet() (output *OutputSet) {
	output = &OutputSet{}
	for _, label := range labels.Labels {
		output[int(label.Time*float64(SampleRate/StrideWidth))] = label.Cmd.ToOutput()
	}
	return
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
	Silence Cmd = iota
	Unknown
	Yes
	No
	Zero
	One
	Two
	Three
	Four
	Five
	Six
	Seven
	Eight
	Nine
	Ten
	CtrlC
	Sudo
	Tab
	Return
	Train
	Mpc
	Play
	Pause
	True
	False
	Wrong
	Grep
	What
	Same
	Different
	When
	Who
)

// CmdToString converts a cmd to the cmds name.
var CmdToString = map[Cmd]string{
	Silence:   "Silence",
	Unknown:   "Unknown",
	Yes:       "Yes",
	No:        "No",
	Zero:      "Zero",
	One:       "One",
	Two:       "Two",
	Three:     "Three",
	Four:      "Four",
	Five:      "Five",
	Six:       "Six",
	Seven:     "Seven",
	Eight:     "Eight",
	Nine:      "Nine",
	Ten:       "Ten",
	CtrlC:     "CtrlC",
	Sudo:      "Sudo",
	Tab:       "Tab",
	Return:    "Return",
	Train:     "Train",
	Mpc:       "Mpc",
	Play:      "Play",
	Pause:     "Pause",
	True:      "True",
	False:     "False",
	Wrong:     "Wrong",
	Grep:      "Grep",
	What:      "What",
	Same:      "Same",
	Different: "Different",
	When:      "When",
	Who:       "Who",
}

// IsNumeric is a map of Cmds that are numeric.
var IsNumeric = map[Cmd]bool{
	Zero:  true,
	One:   true,
	Two:   true,
	Three: true,
	Four:  true,
	Five:  true,
	Six:   true,
	Seven: true,
	Eight: true,
	Nine:  true,
	Ten:   true,
}

// NumericToInt converts a numeric Cmd to an int.
var NumericToInt = map[Cmd]int{
	Zero:  0,
	One:   1,
	Two:   2,
	Three: 3,
	Four:  4,
	Five:  5,
	Six:   6,
	Seven: 7,
	Eight: 8,
	Nine:  9,
	Ten:   10,
}
