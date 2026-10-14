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

func usageFromBlocks(blockSize int64, totalBlocks, freeBlocks, availableBlocks uint64) (Usage, error) {
	if blockSize <= 0 {
		return Usage{}, fmt.Errorf("filesystem block size is %d", blockSize)
	}
	if freeBlocks > totalBlocks {
		return Usage{}, fmt.Errorf("free blocks %d exceed total blocks %d", freeBlocks, totalBlocks)
	}
	if availableBlocks > freeBlocks {
		return Usage{}, fmt.Errorf("available blocks %d exceed free blocks %d", availableBlocks, freeBlocks)
	}

	unit := uint64(blockSize)
	if totalBlocks > math.MaxUint64/unit {
		return Usage{}, fmt.Errorf("filesystem size overflows uint64: %d blocks of %d bytes", totalBlocks, unit)
	}
	totalBytes := totalBlocks * unit
	freeBytes := freeBlocks * unit
	availableBytes := availableBlocks * unit
	usedBytes := totalBytes - freeBytes
	usableBytes := usedBytes + availableBytes
	usagePercent := 0.0
	if usableBytes > 0 {
		usagePercent = float64(usedBytes) / float64(usableBytes) * 100
	}

	return Usage{
		TotalBytes:     totalBytes,
		UsedBytes:      usedBytes,
		AvailableBytes: availableBytes,
		UsagePercent:   usagePercent,
	}, nil
}
