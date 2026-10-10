//nolint:testpackage // tests exercise the unexported block arithmetic.
package disk

import "testing"

func TestUsageFromBlocksUsesGivenBlockSize(t *testing.T) {
	// 1000 blocks of 512 bytes, 250 free. A caller that used a different
	// block size would get different byte totals.
	usage, err := usageFromBlocks(512, 1000, 250)
	if err != nil {
		t.Fatal(err)
	}
	if usage.TotalBytes != 512_000 || usage.AvailableBytes != 128_000 || usage.UsedBytes != 384_000 {
		t.Fatalf("unexpected byte totals: %+v", usage)
	}
	if usage.UsagePercent != 75 {
		t.Fatalf("usage percent = %v, want 75", usage.UsagePercent)
	}
}

func TestUsageFromBlocksRejectsBadBlockSize(t *testing.T) {
	for _, blockSize := range []int64{0, -1} {
		if _, err := usageFromBlocks(blockSize, 1000, 250); err == nil {
			t.Fatalf("expected an error for block size %d", blockSize)
		}
	}
}

func TestUsageFromBlocksEmptyFilesystem(t *testing.T) {
	usage, err := usageFromBlocks(4096, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if usage.UsagePercent != 0 {
		t.Fatalf("usage percent = %v, want 0", usage.UsagePercent)
	}
}
