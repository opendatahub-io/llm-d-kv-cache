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
	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/pending"
)

func newTestDeleter(cfg *config.Config, publisher EventPublisher) (*Deleter, *pending.Paths) {
	pendingPaths := pending.NewPaths()
	return New(cfg, publisher, pendingPaths), pendingPaths
}

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
	deleter, _ := newTestDeleter(&cfg, nil)
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

func TestFlushDoesNotQueueRecentlyModifiedParent(t *testing.T) {
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
		DirCleanupTTL:           time.Hour,
		FileAccessTimeThreshold: 0,
	}
	deleter, _ := newTestDeleter(cfg, nil)
	folderQueue := make(chan string, 1)
	deleter.flush(context.Background(), []string{filePath}, folderQueue, slog.New(slog.DiscardHandler))

	select {
	case parent := <-folderQueue:
		t.Fatalf("queued recently modified parent %q", parent)
	default:
	}
}

func TestFlushQueuesOldParentBeforeDeletionUpdatesMtime(t *testing.T) {
	root := t.TempDir()
	parent := filepath.Join(root, "cache", "model_r0", "abc", "de_g0")
	filePath := filepath.Join(parent, "0000000000000000.bin")
	if err := os.MkdirAll(parent, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filePath, []byte("cache"), 0o600); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(parent, old, old); err != nil {
		t.Fatal(err)
	}

	cfg := &config.Config{
		PVCMountPath:            root,
		CacheDirectory:          "cache",
		DirCleanupTTL:           time.Minute,
		FileAccessTimeThreshold: 0,
	}
	deleter, _ := newTestDeleter(cfg, nil)
	folderQueue := make(chan string, 1)
	deleter.flush(context.Background(), []string{filePath}, folderQueue, slog.New(slog.DiscardHandler))

	select {
	case got := <-folderQueue:
		if got != parent {
			t.Fatalf("queued parent %q, want %q", got, parent)
		}
	default:
		t.Fatal("old parent was not queued")
	}
}

// Deleting the first file updates the mtime of the directory. The second file
// of the same batch must still use the mtime from before the batch.
func TestFlushQueuesOldParentOnceForSeveralFiles(t *testing.T) {
	root := t.TempDir()
	parent := filepath.Join(root, "cache", "model_r0", "abc", "de_g0")
	if err := os.MkdirAll(parent, 0o750); err != nil {
		t.Fatal(err)
	}
	batch := make([]string, 0, 2)
	for _, name := range []string{"0000000000000000.bin", "0000000000000001.bin"} {
		filePath := filepath.Join(parent, name)
		if err := os.WriteFile(filePath, []byte("cache"), 0o600); err != nil {
			t.Fatal(err)
		}
		batch = append(batch, filePath)
	}
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(parent, old, old); err != nil {
		t.Fatal(err)
	}

	cfg := &config.Config{
		PVCMountPath:            root,
		CacheDirectory:          "cache",
		DirCleanupTTL:           time.Minute,
		FileAccessTimeThreshold: 0,
	}
	deleter, _ := newTestDeleter(cfg, nil)
	folderQueue := make(chan string, 2)
	deleter.flush(context.Background(), batch, folderQueue, slog.New(slog.DiscardHandler))

	if got := deleter.filesDeleted.Load(); got != 2 {
		t.Fatalf("deleted %d files, want 2", got)
	}
	if got := len(folderQueue); got != 1 {
		t.Fatalf("queued %d parents, want 1", got)
	}
	if got := <-folderQueue; got != parent {
		t.Fatalf("queued parent %q, want %q", got, parent)
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
	deleter, _ := newTestDeleter(&cfg, nil)
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
	deleter, _ := newTestDeleter(cfg, nil)
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
	deleter, _ := newTestDeleter(cfg, publisher)
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

func TestFlushStopsBeforeDeletingWhenContextIsAlreadyCancelled(t *testing.T) {
	root := t.TempDir()
	bucket := filepath.Join(root, "cache", "model_r0", "abc", "de_g0")
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

	// The default rate is unlimited, so acquisition must still honor the
	// canceled context before either file is deleted.
	cfg := &config.Config{PVCMountPath: root, CacheDirectory: "cache"}
	deleter, _ := newTestDeleter(cfg, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	deleter.flush(ctx, []string{first, second}, make(chan string, 1), slog.New(slog.DiscardHandler))

	for _, path := range []string{first, second} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("file was removed after cancellation: %v", err)
		}
	}
	if deleter.filesDeleted.Load() != 0 {
		t.Fatalf("deleted count = %d, want 0", deleter.filesDeleted.Load())
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
	deleter, _ := newTestDeleter(&cfg, nil)
	modelName, ok := deleter.modelName(filepath.Join(root, "cache", "model_digest_r0", "abc", "de_g0", "hash.bin"))
	if !ok || modelName != "test-model" {
		t.Fatalf("model name = %q, %v; want test-model, true", modelName, ok)
	}
}

func TestFlushDisablesEventsAfterMetadataRetries(t *testing.T) {
	originalRetryDelay := metadataRetryDelay
	metadataRetryDelay = 0
	t.Cleanup(func() { metadataRetryDelay = originalRetryDelay })

	root := t.TempDir()
	bucket := filepath.Join(root, "cache", "model_r0", "abc", "de_g0")
	if err := os.MkdirAll(bucket, 0o750); err != nil {
		t.Fatal(err)
	}
	filePath := filepath.Join(bucket, "0000000000000001.bin")
	if err := os.WriteFile(filePath, []byte("cache"), 0o600); err != nil {
		t.Fatal(err)
	}

	publisher := &recordingPublisher{hashes: map[string][]uint64{}}
	cfg := &config.Config{PVCMountPath: root, CacheDirectory: "cache"}
	deleter, _ := newTestDeleter(cfg, publisher)
	deleter.flush(context.Background(), []string{filePath}, make(chan string, 1), slog.New(slog.DiscardHandler))

	if _, err := os.Stat(filePath); !os.IsNotExist(err) {
		t.Fatalf("file still exists after deletion: %v", err)
	}
	if deleter.publisher != nil {
		t.Fatal("publisher remains enabled after metadata retries failed")
	}
	if len(publisher.hashes) != 0 {
		t.Fatalf("published hashes = %v, want none", publisher.hashes)
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
	deleter, _ := newTestDeleter(&cfg, nil)

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

func TestFlushReleasesPendingPath(t *testing.T) {
	root := t.TempDir()
	filePath := filepath.Join(root, "cache", "model_r0", "abc", "de_g0", "0000000000000000.bin")
	if err := os.MkdirAll(filepath.Dir(filePath), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filePath, []byte("cache"), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg := &config.Config{PVCMountPath: root, CacheDirectory: "cache"}
	deleter, pendingPaths := newTestDeleter(cfg, nil)
	if !pendingPaths.Add(filePath) {
		t.Fatal("pending path was already claimed")
	}

	deleter.flush(context.Background(), []string{filePath}, make(chan string, 1), slog.New(slog.DiscardHandler))

	if got := pendingPaths.Len(); got != 0 {
		t.Fatalf("pending paths after flush = %d, want 0", got)
	}
}

func TestReleasePathsAllowsCrawlerToQueueAgain(t *testing.T) {
	cfg := &config.Config{PVCMountPath: t.TempDir(), CacheDirectory: "cache"}
	deleter, pendingPaths := newTestDeleter(cfg, nil)
	path := "/cache/model_r0/abc/de_g0/0000000000000000.bin"
	if !pendingPaths.Add(path) {
		t.Fatal("pending path was already claimed")
	}

	deleter.releasePaths([]string{path})

	if !pendingPaths.Add(path) {
		t.Fatal("path remained pending after release")
	}
}
