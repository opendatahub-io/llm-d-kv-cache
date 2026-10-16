//nolint:testpackage // tests exercise the unexported block arithmetic.
package disk

import "testing"

func TestUsageFromBlocksUsesGivenBlockSize(t *testing.T) {
	// 1000 blocks of 512 bytes, 250 free. A caller that used a different
	// block size would get different byte totals.
	usage, err := usageFromBlocks(512, 1000, 250, 250)
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

func TestUsageFromBlocksExcludesReservedBlocks(t *testing.T) {
	usage, err := usageFromBlocks(1, 100, 20, 10)
	if err != nil {
		t.Fatal(err)
	}
	if usage.TotalBytes != 100 || usage.AvailableBytes != 10 || usage.UsedBytes != 80 {
		t.Fatalf("unexpected byte totals: %+v", usage)
	}
	want := float64(80) / 90 * 100
	if usage.UsagePercent != want {
		t.Fatalf("usage percent = %v, want %v", usage.UsagePercent, want)
	}
}

func TestUsageFromBlocksRejectsBadBlockSize(t *testing.T) {
	for _, blockSize := range []int64{0, -1} {
		if _, err := usageFromBlocks(blockSize, 1000, 250, 250); err == nil {
			t.Fatalf("expected an error for block size %d", blockSize)
		}
	}
}

func TestUsageFromBlocksRejectsAvailableBlocksAboveFree(t *testing.T) {
	if _, err := usageFromBlocks(4096, 100, 50, 51); err == nil {
		t.Fatal("expected an error when available blocks exceed free blocks")
	}
}

func TestUsageFromBlocksRejectsFreeBlocksAboveTotal(t *testing.T) {
	if _, err := usageFromBlocks(4096, 100, 101, 101); err == nil {
		t.Fatal("expected an error when free blocks exceed total blocks")
	}
}

func TestUsageFromBlocksRejectsByteOverflow(t *testing.T) {
	if _, err := usageFromBlocks(4096, ^uint64(0), 0, 0); err == nil {
		t.Fatal("expected an error when byte totals overflow")
	}
}

func TestUsageFromBlocksEmptyFilesystem(t *testing.T) {
	usage, err := usageFromBlocks(4096, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if usage.UsagePercent != 0 {
		t.Fatalf("usage percent = %v, want 0", usage.UsagePercent)
	}
}
