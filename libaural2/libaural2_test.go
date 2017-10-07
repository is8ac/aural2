package libaural2

import (
	"bytes"
	"crypto/sha256"
	"testing"
)

func TestNumericLookup(t *testing.T) {
	if NumericToInt[Zero] != 0 {
		t.Fail()
	}
	if NumericToInt[One] != 1 {
		t.Fail()
	}
}

func TestSerialize(t *testing.T) {
	hash := sha256.Sum256([]byte("some fake raw data"))
	labelSet := LabelSet{
		ID: hash,
		Labels: []Label{
			Label{
				Cmd:  Yes,
				Time: 1.23,
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

func TestToOutputSet(t *testing.T) {
	hash := sha256.Sum256([]byte("some fake raw data"))
	labelSet := LabelSet{
		ID: hash,
		Labels: []Label{
			Label{
				Cmd:  Yes,
				Time: 1.23,
			},
			Label{
				Cmd:  Yes,
				Time: 8.23,
			},
			Label{
				Cmd:  Yes,
				Time: 2.00,
			},
			Label{
				Cmd:  Yes,
				Time: 1,
			},
			Label{
				Cmd:  Yes,
				Time: 9.4,
			},
		},
	}
	outputSet := labelSet.ToOutputSet()
	_ = outputSet
}
