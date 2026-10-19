package pending

import "sync"

// Paths records file paths that are waiting in the deletion queue or in the
// deleter's current batch. Crawlers use it to avoid queueing the same path
// again while an earlier copy is still pending.
type Paths struct {
	mu    sync.Mutex
	paths map[string]struct{}
}

func NewPaths() *Paths {
	return &Paths{paths: make(map[string]struct{})}
}

// Add claims a path and returns false when that path is already pending.
func (p *Paths) Add(path string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()

	if _, exists := p.paths[path]; exists {
		return false
	}
	p.paths[path] = struct{}{}
	return true
}

// Remove releases a path after the deleter has processed or discarded it.
func (p *Paths) Remove(path string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	delete(p.paths, path)
}

func (p *Paths) Contains(path string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()

	_, exists := p.paths[path]
	return exists
}

func (p *Paths) Len() int {
	p.mu.Lock()
	defer p.mu.Unlock()

	return len(p.paths)
}
