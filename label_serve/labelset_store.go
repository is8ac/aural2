package main

import (
	"encoding/base32"
	"errors"
	"fmt"

	"github.com/boltdb/bolt"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
)

func initDB() (
	put func(libaural2.LabelSet) error,
	get func(libaural2.ClipID) (libaural2.LabelSet, error),
	getAll func() (map[libaural2.ClipID]libaural2.LabelSet, error),
	list func() []libaural2.ClipID,
	close func(),
	err error,
	) {
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
	getAll = func() (labelSets map[libaural2.ClipID]libaural2.LabelSet, err error) {
		labelSets = map[libaural2.ClipID]libaural2.LabelSet{}
		db.View(func(tx *bolt.Tx) error {
			// Assume bucket exists and has keys
			b := tx.Bucket([]byte("labelsets"))
			c := b.Cursor()
			for k, v := c.First(); k != nil; k, v = c.Next() {
				labelSet, err := libaural2.DeserializeLabelSet(v)
				if err != nil {
					return err
				}
				var clipID libaural2.ClipID
				copy(clipID[:], k)
				labelSets[clipID] = labelSet
			}
			return nil
		})
		return
	}
	list = func() (ids []libaural2.ClipID) {
		db.View(func(tx *bolt.Tx) error {
			// Assume bucket exists and has keys
			b := tx.Bucket([]byte("labelsets"))
			c := b.Cursor()
			for k, v := c.First(); k != nil; k, v = c.Next() {
				_ = v
				logger.Println("iter")
				clipIDBytes, err := base32.StdEncoding.DecodeString(string(k))
				if err != nil {
					return err
				}
				if len(clipIDBytes) != 32 {
					err = errors.New("hash length must be 32 bytes")
					return err
				}
				var clipID libaural2.ClipID
				copy(clipID[:], clipIDBytes)
				logger.Println(clipID)
				ids = append(ids, clipID)
			}
			return nil
		})
		return
	}
	return
}
