//nolint:testpackage // tests exercise the unexported streaming scanner.
package crawler

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/config"
)

func TestScanQueuesColdFileAndSkipsHotFile(t *testing.T) {
	root := t.TempDir()
	rankDir := filepath.Join(root, "cache", "model_digest_r0")
	firstBucket := filepath.Join(rankDir, "abc")
	secondBucket := filepath.Join(firstBucket, "de_g0")
	if err := os.MkdirAll(secondBucket, 0o750); err != nil {
		t.Fatal(err)
	}

	coldFile := filepath.Join(secondBucket, "0000000000000000.bin")
	hotFile := filepath.Join(secondBucket, "0000000000000001.bin")
	writeFile(t, coldFile, "cold")
	writeFile(t, hotFile, "hot")

	oldTime := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(coldFile, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}

	cfg := config.Config{
		PVCMountPath:            root,
		CacheDirectory:          "cache",
		FileQueueMaxSize:        10,
		FileQueueMinSize:        10,
		FileAccessTimeThreshold: time.Hour,
		HexBucketLen:            3,
		DirCleanupTTL:           0,
	}
	fileQueue := make(chan string, 10)
	folderQueue := make(chan string, 10)
	stats := Stats{}

	err := scan(
		context.Background(),
		&cfg,
		filepath.Join(root, "cache"),
		[2]int{0, 15},
		&atomic.Bool{},
		fileQueue,
		folderQueue,
		&stats,
		slog.New(slog.DiscardHandler),
	)
	if err != nil {
		t.Fatal(err)
	}

	if stats.Discovered != 2 || stats.Queued != 1 || stats.SkippedHot != 1 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
	if got := <-fileQueue; got != coldFile {
		t.Fatalf("queued %q, want %q", got, coldFile)
	}
}

func TestHexModuloRanges(t *testing.T) {
	ranges := HexModuloRanges(4)
	want := [][2]int{{0, 3}, {4, 7}, {8, 11}, {12, 15}}
	if len(ranges) != len(want) {
		t.Fatalf("got %d ranges, want %d", len(ranges), len(want))
	}
	for i := range want {
		if ranges[i] != want[i] {
			t.Fatalf("range %d = %v, want %v", i, ranges[i], want[i])
		}
	}
}

func TestScanOffersEmptyFirstBucketOnlyToOwner(t *testing.T) {
	root := t.TempDir()
	rankDir := filepath.Join(root, "cache", "model_digest_r0")
	if err := os.MkdirAll(filepath.Join(rankDir, "abc", "de_g0"), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(rankDir, "abc", "de_g0", "keep.bin"), []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	emptyBucket := filepath.Join(rankDir, "00f")
	if err := os.Mkdir(emptyBucket, 0o750); err != nil {
		t.Fatal(err)
	}

	cfg := &config.Config{
		PVCMountPath:            root,
		CacheDirectory:          "cache",
		FileQueueMaxSize:        10,
		FileQueueMinSize:        10,
		FileAccessTimeThreshold: time.Hour,
		HexBucketLen:            3,
		DirCleanupTTL:           0,
	}
	ownerFolders := make(chan string, 1)
	otherFolders := make(chan string, 1)
	stats := Stats{}

	if err := scan(
		context.Background(),
		cfg,
		filepath.Join(root, "cache"),
		[2]int{15, 15},
		&atomic.Bool{},
		make(chan string, 10),
		ownerFolders,
		&stats,
		slog.New(slog.DiscardHandler),
	); err != nil {
		t.Fatal(err)
	}
	if err := scan(
		context.Background(),
		cfg,
		filepath.Join(root, "cache"),
		[2]int{12, 12},
		&atomic.Bool{},
		make(chan string, 10),
		otherFolders,
		&stats,
		slog.New(slog.DiscardHandler),
	); err != nil {
		t.Fatal(err)
	}

	if got := <-ownerFolders; got != emptyBucket {
		t.Fatalf("owner queued %q, want %q", got, emptyBucket)
	}
	select {
	case path := <-otherFolders:
		t.Fatalf("non-owner queued %q", path)
	default:
	}
}

func TestNextRescanDelay(t *testing.T) {
	tests := []struct {
		previous time.Duration
		queued   int
		want     time.Duration
	}{
		{previous: time.Minute, queued: 1, want: time.Second},
		{previous: time.Second, queued: 0, want: 2 * time.Second},
		{previous: time.Minute, queued: 0, want: time.Minute},
	}
	for _, test := range tests {
		if got := nextRescanDelay(test.previous, test.queued); got != test.want {
			t.Fatalf("nextRescanDelay(%v, %d) = %v, want %v", test.previous, test.queued, got, test.want)
		}
	}
}

func TestWaitForQueueSlotBlocksWhileQueueIsFull(t *testing.T) {
	fileQueue := make(chan string, 1)
	fileQueue <- "full"
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	start := time.Now()

	if waitForQueueSlot(ctx, &atomic.Bool{}, fileQueue, 1, 10) {
		t.Fatal("expected the crawler to pause while the queue is at its minimum")
	}
	if elapsed := time.Since(start); elapsed < 100*time.Millisecond {
		t.Fatalf("waitForQueueSlot returned too early after %v", elapsed)
	}
}

func TestWaitForQueueSlotStopsWhenContextIsCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if waitForQueueSlot(ctx, &atomic.Bool{}, make(chan string, 10), 10, 10) {
		t.Fatal("expected no slot after the context is cancelled, even with free space")
	}
}

func TestScanStopsInsideRankDirWhenContextIsCancelled(t *testing.T) {
	root := t.TempDir()
	secondBucket := filepath.Join(root, "cache", "model_digest_r0", "abc", "de_g0")
	if err := os.MkdirAll(secondBucket, 0o750); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(secondBucket, "0000000000000000.bin"), "data")
	cfg := &config.Config{
		PVCMountPath:     root,
		CacheDirectory:   "cache",
		FileQueueMaxSize: 10,
		FileQueueMinSize: 10,
		HexBucketLen:     3,
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	stats := Stats{}

	processRankDir(ctx, cfg, filepath.Join(root, "cache", "model_digest_r0"), [2]int{0, 15},
		&atomic.Bool{}, make(chan string, 10), make(chan string, 10), &stats)
	if stats.Discovered != 0 {
		t.Fatalf("discovered %d files after the context was cancelled", stats.Discovered)
	}
}

func TestScanSkipsFoldersWhenDirectoryCleanupIsDisabled(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "cache", "model_digest_r0", "00f"), 0o750); err != nil {
		t.Fatal(err)
	}
	cfg := &config.Config{
		PVCMountPath:     root,
		CacheDirectory:   "cache",
		FileQueueMaxSize: 10,
		HexBucketLen:     3,
	}
	stats := Stats{}

	if err := scan(context.Background(), cfg, filepath.Join(root, "cache"), [2]int{0, 15},
		&atomic.Bool{}, make(chan string, 10), nil, &stats, slog.New(slog.DiscardHandler)); err != nil {
		t.Fatal(err)
	}
	if stats.EmptyDirs != 0 {
		t.Fatalf("queued %d empty dirs with cleanup disabled", stats.EmptyDirs)
	}
}

func writeFile(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatal(err)
	}
}
