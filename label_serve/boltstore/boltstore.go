package boltstore

import (
	"errors"
	"fmt"

	"github.com/boltdb/bolt"
	"github.ibm.com/Blue-Horizon/aural2/libaural2"
)

var labelSetsBucketName = []byte("labelsets")

var clipBucketName = []byte("clips")

// DB holds
type DB struct {
	boltConn *bolt.DB
}

// Init creates a new boltDB file, or opens the existing file if it already exists.
func Init(path string) (db DB, err error) {
	db.boltConn, err = bolt.Open(path, 0600, nil)
	if err != nil {
		return
	}
	db.boltConn.Update(func(tx *bolt.Tx) (err error) {
		_, err = tx.CreateBucketIfNotExists(labelSetsBucketName)
		if err != nil {
			return fmt.Errorf("create bucket: %s", err)
		}
		return nil
	})
	db.boltConn.Update(func(tx *bolt.Tx) (err error) {
		_, err = tx.CreateBucketIfNotExists(clipBucketName)
		if err != nil {
			return fmt.Errorf("create bucket: %s", err)
		}
		return nil
	})
	return
}

// Close the DB
func (db DB) Close() (err error) {
	err = db.boltConn.Close()
	return
}

// PutLabelSet inserts one labelSet into the DB.
func (db DB) PutLabelSet(labelSet libaural2.LabelSet) (err error) {
	serialized, err := labelSet.Serialize()
	if err != nil {
		return
	}
	err = db.boltConn.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket(labelSetsBucketName)
		b.Put(labelSet.ID[:], serialized)
		return nil
	})
	return
}

// GetLabelSet gets one LabelSet
func (db DB) GetLabelSet(sampleID libaural2.ClipID) (labelSet libaural2.LabelSet, err error) {
	var serialized []byte
	err = db.boltConn.View(func(tx *bolt.Tx) error {
		b := tx.Bucket(labelSetsBucketName)
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

// GetAllLabelSets returns all the labelSets
func (db DB) GetAllLabelSets() (labelSets map[libaural2.ClipID]libaural2.LabelSet, err error) {
	labelSets = map[libaural2.ClipID]libaural2.LabelSet{}
	db.boltConn.View(func(tx *bolt.Tx) error {
		// Assume bucket exists and has keys
		b := tx.Bucket(labelSetsBucketName)
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

// ListLabelSets lists all labelSets
func (db DB) ListLabelSets() (ids []libaural2.ClipID) {
	db.boltConn.View(func(tx *bolt.Tx) (err error) {
		// Assume bucket exists and has keys
		b := tx.Bucket(labelSetsBucketName)
		c := b.Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			_ = v
			if len(k) != 32 {
				err = errors.New("hash length must be 32 bytes")
				return err
			}
			var clipID libaural2.ClipID
			copy(clipID[:], k)
			ids = append(ids, clipID)
		}
		return nil
	})
	return
}

// PutClipID inserts one clipID into the DB
func (db DB) PutClipID(id libaural2.ClipID) (err error) {
	err = db.boltConn.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket(clipBucketName)
		b.Put(id[:], []byte{})
		return nil
	})
	return
}

// ListAudioClips lists all AudioClips
func (db DB) ListAudioClips() (ids []libaural2.ClipID) {
	db.boltConn.View(func(tx *bolt.Tx) (err error) {
		b := tx.Bucket(clipBucketName)
		c := b.Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			_ = v
			if len(k) != 32 {
				err = errors.New("hash length must be 32 bytes")
				return err
			}
			var clipID libaural2.ClipID
			copy(clipID[:], k)
			ids = append(ids, clipID)
		}
		return nil
	})
	fmt.Println(len(ids))
	return
}
