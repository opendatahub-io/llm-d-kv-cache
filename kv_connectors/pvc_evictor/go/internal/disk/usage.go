package disk

import (
	"fmt"
	"math"
)

type Usage struct {
	TotalBytes     uint64
	UsedBytes      uint64
	AvailableBytes uint64
	UsagePercent   float64
}

func usageFromBlocks(blockSize int64, totalBlocks, freeBlocks uint64) (Usage, error) {
	if blockSize <= 0 {
		return Usage{}, fmt.Errorf("filesystem block size is %d", blockSize)
	}
	if freeBlocks > totalBlocks {
		return Usage{}, fmt.Errorf("free blocks %d exceed total blocks %d", freeBlocks, totalBlocks)
	}

	unit := uint64(blockSize)
	if totalBlocks > math.MaxUint64/unit {
		return Usage{}, fmt.Errorf("filesystem size overflows uint64: %d blocks of %d bytes", totalBlocks, unit)
	}
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
