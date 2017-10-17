package boltstore

import (
	"crypto/sha256"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
	"os"
	"testing"
)

func TestInit(t *testing.T) {
	db, err := Init("test.db")
	if err != nil {
		t.Fatal(err)
	}
	if err = db.Close(); err != nil {
		t.Fatal(err)
	}
	os.Remove("test.db")
}

func TestPutLabelSet(t *testing.T) {
	db, err := Init("test.db")
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256([]byte("some fake raw data"))
	labelSet := libaural2.LabelSet{
		ID: hash,
		Labels: []libaural2.Label{
			libaural2.Label{
				Cmd:   libaural2.Who,
				Start: 1.23,
				End:   2.8,
			},
			libaural2.Label{
				Cmd:   libaural2.What,
				Start: 3.23,
				End:   4.8,
			},
			libaural2.Label{
				Cmd:   libaural2.When,
				Start: 5.23,
				End:   6.8,
			},
		},
	}

	if err := db.PutLabelSet(labelSet); err != nil {
		t.Fatal(err)
	}
	if err = db.Close(); err != nil {
		t.Fatal(err)
	}
	os.Remove("test.db")
}

func TestGetLabelSet(t *testing.T) {
	db, err := Init("test.db")
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256([]byte("some fake raw data"))
	labelSet := libaural2.LabelSet{
		ID: hash,
		Labels: []libaural2.Label{
			libaural2.Label{
				Cmd:   libaural2.Who,
				Start: 1.23,
				End:   2.8,
			},
			libaural2.Label{
				Cmd:   libaural2.What,
				Start: 3.23,
				End:   4.8,
			},
			libaural2.Label{
				Cmd:   libaural2.When,
				Start: 5.23,
				End:   6.8,
			},
		},
	}

	if err := db.PutLabelSet(labelSet); err != nil {
		t.Fatal(err)
	}
	outLabelSet, err := db.GetLabelSet(hash)
	if err != nil {
		t.Fatal(err)
	}
	if outLabelSet.ID != hash {
		t.Fatal("wrong id")
	}
	if len(outLabelSet.Labels) != len(labelSet.Labels) {
		t.Fatal("lens do not match")
	}
	if err = db.Close(); err != nil {
		t.Fatal(err)
	}
	os.Remove("test.db")
}
