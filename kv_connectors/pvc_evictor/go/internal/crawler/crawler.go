package crawler

import (
	"context"
	"errors"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/config"
	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/filemeta"
)

const (
	hexModuloBase     = 16
	rescanDelayMin    = time.Second
	rescanDelayMax    = time.Minute
	queuePollInterval = 100 * time.Millisecond
	statusInterval    = 30 * time.Second
)

type Stats struct {
	Discovered int
	Queued     int
	SkippedHot int
	StatErrors int
	EmptyDirs  int
}

func HexModuloRanges(count int) [][2]int {
	if !validCount(count) {
		return nil
	}
	valuesPerCrawler := hexModuloBase / count
	ranges := make([][2]int, count)
	for i := range ranges {
		ranges[i] = [2]int{i * valuesPerCrawler, (i+1)*valuesPerCrawler - 1}
	}
	return ranges
}

func Run(
	ctx context.Context,
	cfg *config.Config,
	moduloRange [2]int,
	deletionActive *atomic.Bool,
	fileQueue chan<- string,
	folderQueue chan<- string,
	logger *slog.Logger,
) error {
	cachePath := filepath.Join(cfg.PVCMountPath, cfg.CacheDirectory)
	stats := Stats{}
	rescanDelay := rescanDelayMin
	lastStatus := time.Now()

	for {
		queuedBeforeSweep := stats.Queued
		if err := scan(ctx, cfg, cachePath, moduloRange, deletionActive, fileQueue, folderQueue, &stats, logger); err != nil {
			if errors.Is(err, context.Canceled) {
				return nil
			}
			logger.Error("Crawler scan failed", slog.Any("error", err))
		}
		rescanDelay = nextRescanDelay(rescanDelay, stats.Queued-queuedBeforeSweep)

		// Log status between sweeps, at most once per statusInterval.
		// This is kept out of the select below so it cannot cut the rescan delay short.
		if time.Since(lastStatus) >= statusInterval {
			lastStatus = time.Now()
			logger.Info("Crawler status",
				slog.Int("discovered", stats.Discovered),
				slog.Int("queued", stats.Queued),
				slog.Int("skippedHot", stats.SkippedHot),
				slog.Int("statErrors", stats.StatErrors),
				slog.Int("emptyDirsQueued", stats.EmptyDirs),
				slog.Int("queueSize", len(fileQueue)),
			)
		}

		select {
		case <-ctx.Done():
			logger.Info("Crawler stopped", slog.Int("discovered", stats.Discovered), slog.Int("queued", stats.Queued))
			return nil
		case <-time.After(rescanDelay):
		}
	}
}

func scan(
	ctx context.Context,
	cfg *config.Config,
	cachePath string,
	moduloRange [2]int,
	deletionActive *atomic.Bool,
	fileQueue chan<- string,
	folderQueue chan<- string,
	stats *Stats,
	logger *slog.Logger,
) error {
	return filepath.WalkDir(cachePath, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			if path == cachePath {
				return walkErr
			}
			logger.Debug("Cannot read directory", slog.String("path", path), slog.Any("error", walkErr))
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if !entry.IsDir() || path == cachePath {
			return nil
		}

		if containsHexBucket(path) {
			processRankDir(ctx, cfg, path, moduloRange, deletionActive, fileQueue, folderQueue, stats)
			return fs.SkipDir
		}
		return nil
	})
}

func processRankDir(
	ctx context.Context,
	cfg *config.Config,
	rankPath string,
	moduloRange [2]int,
	deletionActive *atomic.Bool,
	fileQueue chan<- string,
	folderQueue chan<- string,
	stats *Stats,
) {
	firstEntries, err := os.ReadDir(rankPath)
	if err != nil {
		return
	}
	hasFirstBucket := false
	for _, firstEntry := range firstEntries {
		if ctx.Err() != nil {
			return
		}
		if !firstEntry.IsDir() || len(firstEntry.Name()) != cfg.HexBucketLen {
			continue
		}
		hexValue, err := strconv.ParseUint(firstEntry.Name(), 16, 64)
		if err != nil {
			continue
		}
		hasFirstBucket = true
		firstPath := filepath.Join(rankPath, firstEntry.Name())

		modulo := int(hexValue % hexModuloBase)
		if modulo < moduloRange[0] || modulo > moduloRange[1] {
			continue
		}
		if isEmptyDir(firstPath) {
			offerFolder(ctx, folderQueue, firstPath, cfg.DirCleanupTTL, stats)
			continue
		}

		secondEntries, err := os.ReadDir(firstPath)
		if err != nil {
			continue
		}
		hasSecondBucket := false
		for _, secondEntry := range secondEntries {
			if !secondEntry.IsDir() {
				continue
			}
			hasSecondBucket = true
			secondPath := filepath.Join(firstPath, secondEntry.Name())
			hasBinFile := false
			binEntries, err := os.ReadDir(secondPath)
			if err != nil {
				continue
			}
			for _, binEntry := range binEntries {
				if ctx.Err() != nil {
					return
				}
				if binEntry.IsDir() || !strings.HasSuffix(binEntry.Name(), ".bin") {
					continue
				}
				hasBinFile = true
				offerFile(ctx, cfg, filepath.Join(secondPath, binEntry.Name()), deletionActive, fileQueue, stats)
			}
			if !hasBinFile {
				offerFolder(ctx, folderQueue, secondPath, cfg.DirCleanupTTL, stats)
			}
		}
		if !hasSecondBucket {
			offerFolder(ctx, folderQueue, firstPath, cfg.DirCleanupTTL, stats)
		}
	}
	if !hasFirstBucket {
		offerFolder(ctx, folderQueue, rankPath, cfg.DirCleanupTTL, stats)
	}
}

func offerFile(
	ctx context.Context,
	cfg *config.Config,
	path string,
	deletionActive *atomic.Bool,
	fileQueue chan<- string,
	stats *Stats,
) {
	if !waitForQueueSlot(ctx, deletionActive, fileQueue, cfg.FileQueueMinSize, cfg.FileQueueMaxSize) {
		return
	}

	stats.Discovered++
	fileInfo, err := filemeta.Stat(path)
	if err != nil {
		stats.StatErrors++
		return
	}
	if time.Since(fileInfo.AccessTime) < cfg.FileAccessTimeThreshold {
		stats.SkippedHot++
		return
	}

	for {
		select {
		case fileQueue <- path:
			stats.Queued++
			return
		case <-ctx.Done():
			return
		case <-time.After(queuePollInterval):
		}
	}
}

func waitForQueueSlot(
	ctx context.Context,
	deletionActive *atomic.Bool,
	fileQueue chan<- string,
	minQueueSize int,
	maxQueueSize int,
) bool {
	for {
		if ctx.Err() != nil {
			return false
		}
		targetSize := minQueueSize
		pollInterval := time.Second
		if deletionActive.Load() {
			targetSize = maxQueueSize
			pollInterval = queuePollInterval
		}
		if len(fileQueue) < targetSize {
			return true
		}

		select {
		case <-ctx.Done():
			return false
		case <-time.After(pollInterval):
		}
	}
}

func nextRescanDelay(previousDelay time.Duration, queuedThisSweep int) time.Duration {
	if queuedThisSweep > 0 {
		return rescanDelayMin
	}
	return min(previousDelay*2, rescanDelayMax)
}

// offerFolder does nothing when folderQueue is nil, which is the case when
// directory cleanup is disabled.
func offerFolder(ctx context.Context, folderQueue chan<- string, path string, ttl time.Duration, stats *Stats) {
	if folderQueue == nil {
		return
	}
	if ttl > 0 {
		info, err := os.Stat(path)
		if err != nil || time.Since(info.ModTime()) < ttl {
			return
		}
	}
	select {
	case folderQueue <- path:
		stats.EmptyDirs++
	default:
	case <-ctx.Done():
	}
}

func containsHexBucket(path string) bool {
	entries, err := os.ReadDir(path)
	if err != nil {
		return false
	}
	for _, entry := range entries {
		if !entry.IsDir() || len(entry.Name()) < 2 || len(entry.Name()) > 4 {
			continue
		}
		if _, err := strconv.ParseUint(entry.Name(), 16, 64); err == nil {
			return true
		}
	}
	return false
}

func isEmptyDir(path string) bool {
	entries, err := os.ReadDir(path)
	return err == nil && len(entries) == 0
}

func validCount(count int) bool {
	return count == 1 || count == 2 || count == 4 || count == 8 || count == 16
}
