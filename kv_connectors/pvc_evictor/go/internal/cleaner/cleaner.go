package cleaner

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func Run(ctx context.Context, cachePath string, folderQueue <-chan string, logger *slog.Logger) error {
	var foldersPurged uint64
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			logger.Info("Folder cleaner stopped", slog.Uint64("foldersPurged", foldersPurged))
			return nil
		case path := <-folderQueue:
			foldersPurged += removeEmptyParents(path, cachePath, logger)
		case <-ticker.C:
			logger.Info("Folder cleaner status", slog.Uint64("foldersPurged", foldersPurged))
		}
	}
}

func removeEmptyParents(path, cachePath string, logger *slog.Logger) uint64 {
	current := path
	if info, err := os.Stat(path); err == nil && !info.IsDir() {
		current = filepath.Dir(path)
	}

	var removed uint64
	for {
		// Stop at the cache root, and never touch a path outside it.
		relative, err := filepath.Rel(cachePath, current)
		if err != nil || relative == "." || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
			break
		}
		info, err := os.Stat(current)
		if err != nil || !info.IsDir() {
			current = filepath.Dir(current)
			continue
		}
		if err := os.Remove(current); err != nil {
			break
		}
		removed++
		current = filepath.Dir(current)
	}
	if removed > 0 {
		logger.Debug("Removed empty directories", slog.String("from", path), slog.Uint64("count", removed))
	}
	return removed
}
