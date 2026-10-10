package disk

import (
	"fmt"
	"syscall"
)

type Usage struct {
	TotalBytes     uint64
	UsedBytes      uint64
	AvailableBytes uint64
	UsagePercent   float64
}

func UsageFromStatfs(path string) (Usage, error) {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return Usage{}, err
	}

	// The block counts are in fragment-size units (Frsize), not in the
	// preferred I/O block size (Bsize). The two differ on some filesystems.
	// Some kernels report no fragment size. Use Bsize in that case.
	blockSize := int64(stat.Frsize) //nolint:unconvert // int32 on 32-bit Linux.
	if blockSize == 0 {
		blockSize = int64(stat.Bsize) //nolint:unconvert // int32 on 32-bit Linux.
	}
	return usageFromBlocks(blockSize, stat.Blocks, stat.Bfree)
}

func usageFromBlocks(blockSize int64, totalBlocks, freeBlocks uint64) (Usage, error) {
	if blockSize <= 0 {
		return Usage{}, fmt.Errorf("filesystem block size is %d", blockSize)
	}

	unit := uint64(blockSize)
	totalBytes := totalBlocks * unit
	freeBytes := freeBlocks * unit
	usedBytes := totalBytes - freeBytes
	usagePercent := 0.0
	if totalBytes > 0 {
		usagePercent = float64(usedBytes) / float64(totalBytes) * 100
	}

	return Usage{
		TotalBytes:     totalBytes,
		UsedBytes:      usedBytes,
		AvailableBytes: freeBytes,
		UsagePercent:   usagePercent,
	}, nil
}
