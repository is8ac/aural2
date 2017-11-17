// Package libaural2 provides libs share betwene edge, server and browser.
package libaural2

import (
	"bytes"
	"crypto/sha256"
	"encoding/base32"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"image/color"

	"github.com/lucasb-eyer/go-colorful"

	"github.ibm.com/Blue-Horizon/aural2/urbitname"
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

// BatchSize is the size of the one batch
const BatchSize int = 7

// StateList is a list of States
type StateList [StridesPerClip]State

// Input is the one input to the LSTM
type Input [InputSize]float32

// InputSet is the set of inputs for one clip.
type InputSet [StridesPerClip]Input

// Output is one output, the softmax array of States.
type Output []float32

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

// FSsafeString returns an encoding of the ClipID safe for filesystems and URLs.
func (hash ClipID) FSsafeString() string {
	return base32.StdEncoding.EncodeToString(hash[:])
}

func (hash ClipID) String() string {
	return urbitname.Encode(hash[0:4])
}

// VocabName is the name of a vocabulary
type VocabName string

// State is one exclusive thing the NN can output
type State int

// Just for testing
const (
	Nil State = iota
	Unknown
	Foo
	Bar
	Baz
	Yes
	No
)

// Vocabulary is one exclusive list of words
type Vocabulary struct {
	Name       VocabName
	Size       int
	Names      map[State]string
	Hue        map[State]float64
	KeyMapping map[string]State
}

// Color turns a cmd into something that implements the color.Color interface
func (voc Vocabulary) Color(state State) (c color.Color) {
	c = colorful.Hsv(voc.Hue[state], 1, 1)
	return
}

var testVocab Vocabulary

// Hue returns the hue as a float64
func (state State) Hue() (hue float64) {
	hash := sha256.Sum256([]byte{uint8(state)})
	hue = float64(hash[0]) * float64(hash[1]) / (255 * 255) * 360
	return
}

// RGBA implements color.Color
func (state State) RGBA() (uint32, uint32, uint32, uint32) {
	return colorful.Hsv(state.Hue(), 1, 1).RGBA()
}

// Label is one period of time.
type Label struct {
	State State
	Start float64 // the duration since the start of the clip.
	End   float64
}

// LabelSet is the set of labels for one Clip
type LabelSet struct {
	VocabName VocabName
	ID        ClipID
	Labels    []Label
}

// ToStateIDArray converts the labelSet to a slice of State IDs
func (labels *LabelSet) ToStateIDArray() (stateArray [StridesPerClip]int32) {
	for i := range stateArray {
		loc := float64(i) / float64(StridesPerClip) * float64(Duration)
		for _, label := range labels.Labels {
			if loc > label.Start && loc < label.End {
				stateArray[i] = int32(label.State)
				continue
			}
		}
	}
	return
}

// ToStateArray converts the labelSet to a slice of States
func (labels *LabelSet) ToStateArray() (stateArray [StridesPerClip]State) {
	for i := range stateArray {
		loc := float64(i) / float64(StridesPerClip) * float64(Duration)
		for _, label := range labels.Labels {
			if loc > label.Start && loc < label.End {
				stateArray[i] = label.State
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

// GenFakeLabelSet creates a fake LabelSet for testing.
func GenFakeLabelSet() (output LabelSet) {
	output.Labels = []Label{
		Label{
			State: Nil,
			Start: 0,
			End:   1,
		},
		Label{
			State: Yes,
			Start: 1,
			End:   2,
		},
		Label{
			State: No,
			Start: 2,
			End:   3,
		},
		Label{
			State: Nil,
			Start: 3,
			End:   4,
		},
		Label{
			State: Yes,
			Start: 4,
			End:   5,
		},
		Label{
			State: No,
			Start: 5,
			End:   6,
		},
		Label{
			State: Nil,
			Start: 6,
			End:   7,
		},
		Label{
			State: Yes,
			Start: 7,
			End:   8,
		},
		Label{
			State: No,
			Start: 8,
			End:   9,
		},
		Label{
			State: Nil,
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
