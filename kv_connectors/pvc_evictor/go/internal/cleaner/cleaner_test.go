//nolint:testpackage // tests exercise the unexported parent cleanup helper.
package cleaner

import (
	"log/slog"
	"os"
	"path/filepath"
	"testing"
)

func TestRemoveEmptyParents(t *testing.T) {
	root := t.TempDir()
	cachePath := filepath.Join(root, "cache")
	emptyPath := filepath.Join(cachePath, "model_r0", "abc", "de_g0")
	if err := os.MkdirAll(emptyPath, 0o750); err != nil {
		t.Fatal(err)
	}

	removed := removeEmptyParents(emptyPath, cachePath, slog.New(slog.DiscardHandler))
	if removed != 3 {
		t.Fatalf("removed %d directories, want 3", removed)
	}
	if _, err := os.Stat(filepath.Join(cachePath, "model_r0")); !os.IsNotExist(err) {
		t.Fatalf("empty rank directory still exists: %v", err)
	}
	if _, err := os.Stat(cachePath); err != nil {
		t.Fatalf("cache root was removed: %v", err)
	}
}

func TestRemoveEmptyParentsIgnoresPathOutsideCache(t *testing.T) {
	root := t.TempDir()
	cachePath := filepath.Join(root, "cache")
	outsidePath := filepath.Join(root, "other", "empty")
	if err := os.MkdirAll(outsidePath, 0o750); err != nil {
		t.Fatal(err)
	}

	if removed := removeEmptyParents(outsidePath, cachePath, slog.New(slog.DiscardHandler)); removed != 0 {
		t.Fatalf("removed %d directories outside the cache root", removed)
	}
	if _, err := os.Stat(outsidePath); err != nil {
		t.Fatalf("directory outside the cache root was removed: %v", err)
	}
}
