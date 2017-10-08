package main

import (
	"fmt"

	"github.com/boltdb/bolt"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
)

func initDB() (put func(libaural2.LabelSet) error, get func(libaural2.ClipID) (libaural2.LabelSet, error), close func(), err error) {
	db, err := bolt.Open("label_serve.db", 0600, nil)
	if err != nil {
		return
	}
	db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists([]byte("labelsets"))
		if err != nil {
			return fmt.Errorf("create bucket: %s", err)
		}
		return nil
	})
	close = func() {
		db.Close()
		return
	}
	put = func(labelSet libaural2.LabelSet) (err error) {
		serialized, err := labelSet.Serialize()
		if err != nil {
			return
		}
		err = db.Update(func(tx *bolt.Tx) error {
			b := tx.Bucket([]byte("labelsets"))
			b.Put(labelSet.ID[:], serialized)
			return nil
		})
		return
	}
	get = func(sampleID libaural2.ClipID) (labelSet libaural2.LabelSet, err error) {
		var serialized []byte
		err = db.View(func(tx *bolt.Tx) error {
			b := tx.Bucket([]byte("labelsets"))
			serialized = b.Get(sampleID[:])
			return nil
		})
		if len(serialized) == 0 {
			labelSet = libaural2.LabelSet{
				ID:     sampleID,
				Labels: []libaural2.Label{},
			}
			return
		}
		labelSet, err = libaural2.DeserializeLabelSet(serialized)
		return
	}
	return
}