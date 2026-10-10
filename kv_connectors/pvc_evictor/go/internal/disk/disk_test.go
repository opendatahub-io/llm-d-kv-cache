package disk_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/disk"
)

func TestUsageFromStatfs(t *testing.T) {
	path := t.TempDir()

	usage, err := disk.UsageFromStatfs(path)
	if err != nil {
		t.Fatal(err)
	}
	if usage.TotalBytes == 0 {
		t.Fatal("total bytes is zero")
	}
	if usage.AvailableBytes > usage.TotalBytes {
		t.Fatalf("available bytes %d exceeds total bytes %d", usage.AvailableBytes, usage.TotalBytes)
	}
	if usage.UsedBytes+usage.AvailableBytes != usage.TotalBytes {
		t.Fatalf("used %d + available %d does not equal total %d", usage.UsedBytes, usage.AvailableBytes, usage.TotalBytes)
	}

	want := float64(usage.UsedBytes) / float64(usage.TotalBytes) * 100
	if usage.UsagePercent < 0 || usage.UsagePercent > 100 {
		t.Fatalf("usage percent %v is outside 0-100", usage.UsagePercent)
	}
	if usage.UsagePercent != want {
		t.Fatalf("usage percent = %v, want %v", usage.UsagePercent, want)
	}
}

func TestUsageFromStatfsMissingPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing")

	if _, err := disk.UsageFromStatfs(path); !os.IsNotExist(err) {
		t.Fatalf("error = %v, want not-exist", err)
	}
}
