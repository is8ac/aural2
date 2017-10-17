package libaural2

import (
	"bytes"
	"crypto/sha256"
	"testing"
)

func TestSerialize(t *testing.T) {
	hash := sha256.Sum256([]byte("some fake raw data"))
	labelSet := LabelSet{
		ID: hash,
		Labels: []Label{
			Label{
				Cmd:   Yes,
				Start: 1.23,
				End:   2.23,
			},
		},
	}
	serialized, err := labelSet.Serialize()
	if err != nil {
		t.Fail()
	}
	labelSet2, err := DeserializeLabelSet(serialized)
	if err != nil {
		t.Fail()
	}
	serialized2, err := labelSet2.Serialize()
	if err != nil {
		t.Fail()
	}
	if !bytes.Equal(serialized, serialized2) {
		t.Fail()
	}
}

func TestToCmdArray(t *testing.T) {
	hash := sha256.Sum256([]byte("some fake raw data"))
	labelSet := LabelSet{
		ID: hash,
		Labels: []Label{
			Label{
				Cmd:   Yes,
				Start: 0.23,
				End:   1.23,
			},
			Label{
				Cmd:   Nil,
				Start: 2.23,
				End:   3.23,
			},
			Label{
				Cmd:   No,
				Start: 4.23,
				End:   5.23,
			},
			Label{
				Cmd:   Unknown,
				Start: 6.23,
				End:   7.23,
			},
			Label{
				Cmd:   No,
				Start: 8.23,
				End:   9.23,
			},
		},
	}
	cmdArray := labelSet.ToCmdArray()
	if cmdArray[0] != Nil {
		t.Fatal("!silence")
	}
}

func TestIsGood(t *testing.T) {
	hash := sha256.Sum256([]byte("some fake raw data"))
	goodLabelSet := LabelSet{
		ID: hash,
		Labels: []Label{
			Label{
				Cmd:   No,
				Start: 0.23,
				End:   1.23,
			},
			Label{
				Cmd:   Yes,
				Start: 2.23,
				End:   3.23,
			},
			Label{
				Cmd:   No,
				Start: 4.23,
				End:   5.23,
			},
			Label{
				Cmd:   Yes,
				Start: 6.23,
				End:   7.23,
			},
			Label{
				Cmd:   Unknown,
				Start: 8.23,
				End:   9.23,
			},
		},
	}
	if !goodLabelSet.IsGood() {
		t.Fatal("is not good")
	}
	overlappingLabelSet := LabelSet{
		ID: hash,
		Labels: []Label{
			Label{
				Cmd:   No,
				Start: 0.23,
				End:   3.4,
			},
			Label{
				Cmd:   Yes,
				Start: 3.21,
				End:   5.0,
			},
		},
	}
	if overlappingLabelSet.IsGood() {
		t.Fatal("overlapping is good")
	}
	outOfBoundLabelSet := LabelSet{
		ID: hash,
		Labels: []Label{
			Label{
				Cmd:   No,
				Start: -2.0,
				End:   3.4,
			},
			Label{
				Cmd:   Yes,
				Start: 9.21,
				End:   10.3,
			},
		},
	}
	if outOfBoundLabelSet.IsGood() {
		t.Fatal("out of bound is good")
	}
}
