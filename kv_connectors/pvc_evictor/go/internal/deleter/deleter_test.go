//nolint:testpackage // tests exercise the unexported batch flush helper.
package deleter

import (
	"context"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/config"
)

func TestFlushDeletesFileAndQueuesParent(t *testing.T) {
	root := t.TempDir()
	cachePath := filepath.Join(root, "cache")
	filePath := filepath.Join(cachePath, "model_r0", "abc", "de_g0", "0000000000000000.bin")
	if err := os.MkdirAll(filepath.Dir(filePath), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filePath, []byte("cache"), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg := config.Config{PVCMountPath: root, CacheDirectory: "cache"}
	deleter := New(&cfg, nil)
	folderQueue := make(chan string, 1)
	deleter.flush(context.Background(), []string{filePath}, folderQueue, slog.New(slog.DiscardHandler))

	if _, err := os.Stat(filePath); !os.IsNotExist(err) {
		t.Fatalf("file still exists after deletion: %v", err)
	}
	if got := <-folderQueue; got != filepath.Dir(filePath) {
		t.Fatalf("queued parent %q, want %q", got, filepath.Dir(filePath))
	}
	if deleter.filesDeleted.Load() != 1 || deleter.bytesFreed.Load() == 0 {
		t.Fatalf("unexpected deletion totals: files=%d bytes=%d", deleter.filesDeleted.Load(), deleter.bytesFreed.Load())
	}
}

func TestFlushDryRunKeepsFile(t *testing.T) {
	root := t.TempDir()
	filePath := filepath.Join(root, "cache", "model_r0", "abc", "de_g0", "0000000000000000.bin")
	if err := os.MkdirAll(filepath.Dir(filePath), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filePath, []byte("cache"), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg := config.Config{PVCMountPath: root, CacheDirectory: "cache", DryRun: true}
	deleter := New(&cfg, nil)
	deleter.flush(context.Background(), []string{filePath}, make(chan string, 1), slog.New(slog.DiscardHandler))

	if _, err := os.Stat(filePath); err != nil {
		t.Fatalf("dry run removed the file: %v", err)
	}
	if deleter.filesDeleted.Load() != 1 {
		t.Fatalf("dry-run count = %d, want 1", deleter.filesDeleted.Load())
	}
}

func TestFlushSkipsFileAccessedSinceQueued(t *testing.T) {
	root := t.TempDir()
	filePath := filepath.Join(root, "cache", "model_r0", "abc", "de_g0", "0000000000000000.bin")
	if err := os.MkdirAll(filepath.Dir(filePath), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filePath, []byte("cache"), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg := &config.Config{
		PVCMountPath:            root,
		CacheDirectory:          "cache",
		FileAccessTimeThreshold: time.Minute,
	}
	deleter := New(cfg, nil)
	deleter.flush(context.Background(), []string{filePath}, make(chan string, 1), slog.New(slog.DiscardHandler))

	if _, err := os.Stat(filePath); err != nil {
		t.Fatalf("deleter removed a file accessed since it was queued: %v", err)
	}
	if deleter.filesDeleted.Load() != 0 {
		t.Fatalf("deleted count = %d, want 0", deleter.filesDeleted.Load())
	}
}

type recordingPublisher struct {
	hashes map[string][]uint64
}

func (p *recordingPublisher) PublishBlocksRemoved(_ context.Context, blockHashes []uint64, modelName string) error {
	p.hashes[modelName] = append(p.hashes[modelName], blockHashes...)
	return nil
}

func TestFlushReportsDeletedFilesWhenStoppedMidBatch(t *testing.T) {
	root := t.TempDir()
	cachePath := filepath.Join(root, "cache")
	if err := os.MkdirAll(filepath.Join(cachePath, "model_digest"), 0o750); err != nil {
		t.Fatal(err)
	}
	metadataPath := filepath.Join(cachePath, "model_digest", "config.json")
	if err := os.WriteFile(metadataPath, []byte(`{"model_name":"test-model"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	bucket := filepath.Join(cachePath, "model_digest_r0", "abc", "de_g0")
	if err := os.MkdirAll(bucket, 0o750); err != nil {
		t.Fatal(err)
	}
	first := filepath.Join(bucket, "0000000000000001.bin")
	second := filepath.Join(bucket, "0000000000000002.bin")
	for _, path := range []string{first, second} {
		if err := os.WriteFile(path, []byte("cache"), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	// One file per second: the first file is deleted at once and the
	// second waits for the pacer, which is stopped by the cancel below.
	cfg := &config.Config{PVCMountPath: root, CacheDirectory: "cache", DeletionMaxFilesPerSecond: 1}
	publisher := &recordingPublisher{hashes: map[string][]uint64{}}
	deleter := New(cfg, publisher)
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(200*time.Millisecond, cancel)
	defer cancel()

	deleter.flush(ctx, []string{first, second}, make(chan string, 1), slog.New(slog.DiscardHandler))

	if _, err := os.Stat(second); err != nil {
		t.Fatalf("second file was deleted after the stop: %v", err)
	}
	if deleter.filesDeleted.Load() != 1 {
		t.Fatalf("deleted count = %d, want 1", deleter.filesDeleted.Load())
	}
	if got := publisher.hashes["test-model"]; len(got) != 1 || got[0] != 1 {
		t.Fatalf("published hashes = %v, want [1]", got)
	}
}

func TestModelName(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "cache", "model_digest_r0"), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "cache", "model_digest"), 0o750); err != nil {
		t.Fatal(err)
	}
	metadataPath := filepath.Join(root, "cache", "model_digest", "config.json")
	if err := os.WriteFile(metadataPath, []byte(`{"model_name":"test-model"}`), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg := config.Config{PVCMountPath: root, CacheDirectory: "cache"}
	deleter := New(&cfg, nil)
	modelName, ok := deleter.modelName(filepath.Join(root, "cache", "model_digest_r0", "abc", "de_g0", "hash.bin"))
	if !ok || modelName != "test-model" {
		t.Fatalf("model name = %q, %v; want test-model, true", modelName, ok)
	}
}

func TestIntervalForRate(t *testing.T) {
	tests := []struct {
		name string
		rate float64
		want time.Duration
	}{
		{name: "no limit", rate: 0, want: 0},
		{name: "negative means no limit", rate: -5, want: 0},
		{name: "NaN means no limit", rate: math.NaN(), want: 0},
		{name: "one per second", rate: 1, want: time.Second},
		{name: "ten per second", rate: 10, want: 100 * time.Millisecond},
		{name: "one billion per second", rate: 1e9, want: time.Nanosecond},
		{name: "above one billion rounds to no limit", rate: 2e9, want: 0},
		{name: "infinity means no limit", rate: math.Inf(1), want: 0},
		{name: "delay just fits", rate: 1.1e-10, want: time.Duration(float64(time.Second) / 1.1e-10)},
		{name: "delay too long uses the longest delay", rate: 1e-10, want: time.Duration(math.MaxInt64)},
		{name: "tiny rate uses the longest delay", rate: 1e-300, want: time.Duration(math.MaxInt64)},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := intervalForRate(test.rate)
			if got != test.want {
				t.Fatalf("intervalForRate(%v) = %d, want %d", test.rate, got, test.want)
			}
			if got < 0 {
				t.Fatalf("intervalForRate(%v) = %d, must never be negative", test.rate, got)
			}
		})
	}
}

// A huge DELETION_BATCH_SIZE must not make the deleter reserve memory up
// front. Reserving 2^40 strings crashes the process with an out-of-memory error.
func TestRunDoesNotReserveMemoryForHugeBatchSize(t *testing.T) {
	cfg := config.Config{PVCMountPath: t.TempDir(), CacheDirectory: "cache", DeletionBatchSize: 1 << 40}
	deleter := New(&cfg, nil)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	done := make(chan error, 1)
	go func() {
		done <- deleter.Run(ctx, &atomic.Bool{}, make(chan string), nil, slog.New(slog.DiscardHandler))
	}()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run returned %v, want nil", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not stop after the context was cancelled")
	}
}
