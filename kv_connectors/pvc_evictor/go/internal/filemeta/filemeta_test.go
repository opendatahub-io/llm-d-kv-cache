package filemeta_test

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/filemeta"
)

func TestDirCleanupDue(t *testing.T) {
	now := time.Now()
	ttl := time.Minute

	tests := []struct {
		name       string
		modifiedAt time.Time
		ttl        time.Duration
		want       bool
	}{
		{name: "modified before the ttl", modifiedAt: now.Add(-2 * ttl), ttl: ttl, want: true},
		{name: "modified exactly ttl ago", modifiedAt: now.Add(-ttl), ttl: ttl, want: true},
		{name: "modified within the ttl", modifiedAt: now.Add(-ttl / 2), ttl: ttl, want: false},
		{name: "unreadable directory", modifiedAt: time.Time{}, ttl: ttl, want: false},
		{name: "no ttl", modifiedAt: now, ttl: 0, want: true},
		{name: "no ttl and unreadable directory", modifiedAt: time.Time{}, ttl: 0, want: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := filemeta.DirCleanupDue(test.modifiedAt, now, test.ttl); got != test.want {
				t.Fatalf("DirCleanupDue() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestDirModTime(t *testing.T) {
	dir := t.TempDir()
	want := time.Now().Add(-time.Hour).Truncate(time.Second)
	if err := os.Chtimes(dir, want, want); err != nil {
		t.Fatal(err)
	}
	if got := filemeta.DirModTime(dir); !got.Equal(want) {
		t.Fatalf("DirModTime() = %v, want %v", got, want)
	}

	file := filepath.Join(dir, "file")
	if err := os.WriteFile(file, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := filemeta.DirModTime(file); !got.IsZero() {
		t.Fatalf("DirModTime() of a file = %v, want the zero time", got)
	}
	if got := filemeta.DirModTime(filepath.Join(dir, "missing")); !got.IsZero() {
		t.Fatalf("DirModTime() of a missing path = %v, want the zero time", got)
	}
}
