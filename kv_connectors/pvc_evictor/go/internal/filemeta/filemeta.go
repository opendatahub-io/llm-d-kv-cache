// Package filemeta reads file and directory metadata for the evictor.
package filemeta

import (
	"os"
	"time"
)

// Info is the metadata of a cache file that the evictor needs.
type Info struct {
	AccessTime time.Time
	Size       int64
}

// DirModTime returns the modification time of a directory. It returns the zero
// time when the path cannot be read or is not a directory.
func DirModTime(path string) time.Time {
	info, err := os.Stat(path)
	if err != nil || !info.IsDir() {
		return time.Time{}
	}
	return info.ModTime()
}

// DirCleanupDue reports whether an empty directory may be removed. The crawler
// and the deleter both use it, so that they apply the same rule.
//
// A directory that was modified within ttl is left alone, because a writer may
// be about to use it. A ttl of zero or less means there is no wait. A zero
// modifiedAt means the directory could not be read, so it is not due.
func DirCleanupDue(modifiedAt, now time.Time, ttl time.Duration) bool {
	if ttl <= 0 {
		return true
	}
	if modifiedAt.IsZero() {
		return false
	}
	return now.Sub(modifiedAt) >= ttl
}
