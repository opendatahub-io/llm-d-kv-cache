package pending_test

import (
	"sync"
	"sync/atomic"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/pending"
)

func TestPathsAddClaimsOnce(t *testing.T) {
	paths := pending.NewPaths()

	if !paths.Add("/cache/a.bin") {
		t.Fatal("first Add returned false")
	}
	if paths.Add("/cache/a.bin") {
		t.Fatal("second Add returned true")
	}
	if got := paths.Len(); got != 1 {
		t.Fatalf("Len() = %d, want 1", got)
	}
}

func TestPathsRemoveAllowsAddAgain(t *testing.T) {
	paths := pending.NewPaths()
	paths.Add("/cache/a.bin")
	paths.Remove("/cache/a.bin")

	if !paths.Add("/cache/a.bin") {
		t.Fatal("Add after Remove returned false")
	}
}

func TestPathsConcurrentAdds(t *testing.T) {
	paths := pending.NewPaths()
	var successes atomic.Int32
	var workers sync.WaitGroup

	for i := 0; i < 16; i++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			if paths.Add("/cache/a.bin") {
				successes.Add(1)
			}
		}()
	}
	workers.Wait()

	if got := successes.Load(); got != 1 {
		t.Fatalf("successful concurrent Adds = %d, want 1", got)
	}
}
