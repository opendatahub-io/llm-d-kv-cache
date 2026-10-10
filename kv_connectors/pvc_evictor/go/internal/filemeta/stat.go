package filemeta

import (
	"syscall"
	"time"
)

type Info struct {
	AccessTime time.Time
	Size       int64
}

func Stat(path string) (Info, error) {
	var stat syscall.Stat_t
	if err := syscall.Stat(path, &stat); err != nil {
		return Info{}, err
	}
	return Info{
		AccessTime: time.Unix(int64(stat.Atim.Sec), int64(stat.Atim.Nsec)), //nolint:unconvert // int32 on 32-bit Linux.
		Size:       stat.Size,
	}, nil
}
