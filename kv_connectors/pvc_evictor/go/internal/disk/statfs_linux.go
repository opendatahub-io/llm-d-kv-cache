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
	// Bfree includes blocks reserved for privileged users. Bavail is the free
	// space available to the cache writer, so both values are needed to report
	// used capacity and calculate pressure against usable capacity.
	return usageFromBlocks(blockSize, stat.Blocks, stat.Bfree, stat.Bavail)
}
