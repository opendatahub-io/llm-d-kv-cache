//go:build linux

package disk

import "syscall"

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
