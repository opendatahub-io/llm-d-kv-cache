package service

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/cleaner"
	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/config"
	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/crawler"
	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/deleter"
	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/disk"
	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/events"
)

func Run(ctx context.Context, cfg *config.Config, logger *slog.Logger) error {
	if err := waitForMount(ctx, cfg.PVCMountPath, logger); err != nil {
		return err
	}

	var publisher deleter.EventPublisher
	if cfg.StorageEventsEndpoint != "" {
		// Not cancelled with ctx: the socket must stay open after SIGTERM so
		// the deleter's final flush can still send its BlockRemoved events.
		// The deferred Close below waits briefly for those events to leave,
		// then shuts the socket down.
		created, err := events.NewPublisher(context.WithoutCancel(ctx), cfg.StorageEventsEndpoint)
		if err != nil {
			logger.Warn("Storage event publisher unavailable; continuing without events",
				slog.String("endpoint", cfg.StorageEventsEndpoint), slog.Any("error", err))
		} else {
			publisher = created
			defer func() {
				if err := created.Close(); err != nil {
					logger.Warn("Cannot close storage event publisher", slog.Any("error", err))
				}
			}()
			logger.Info("Storage event publisher ready", slog.String("endpoint", cfg.StorageEventsEndpoint))
		}
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	fileQueue := make(chan string, cfg.FileQueueMaxSize)
	// folderQueue stays nil when directory cleanup is disabled. Crawlers and
	// the deleter then skip empty directories instead of filling a queue
	// that nobody reads.
	var folderQueue chan string
	if cfg.EnableDirCleanup {
		folderQueue = make(chan string, cfg.FileQueueMaxSize)
	}
	deletionActive := &atomic.Bool{}

	var workers sync.WaitGroup
	cachePath := filepath.Join(cfg.PVCMountPath, cfg.CacheDirectory)

	for workerID, moduloRange := range crawler.HexModuloRanges(cfg.NumCrawlerProcesses) {
		workerID := workerID
		moduloRange := moduloRange
		workerLogger := logger.With(slog.Int("worker", workerID+1))
		startSupervisedWorker(runCtx, &workers, workerLogger, "crawler", func(ctx context.Context) error {
			return crawler.Run(
				ctx,
				cfg,
				moduloRange,
				deletionActive,
				fileQueue,
				folderQueue,
				workerLogger,
			)
		})
	}

	startSupervisedWorker(runCtx, &workers, logger, "activator", func(ctx context.Context) error {
		return runActivator(ctx, cfg, deletionActive, logger)
	})

	startSupervisedWorker(runCtx, &workers, logger, "deleter", func(ctx context.Context) error {
		return deleter.New(cfg, publisher).Run(ctx, deletionActive, fileQueue, folderQueue, logger)
	})

	if cfg.EnableDirCleanup {
		startSupervisedWorker(runCtx, &workers, logger, "folder cleaner", func(ctx context.Context) error {
			return cleaner.Run(ctx, cachePath, folderQueue, logger)
		})
	}

	workers.Wait()
	return nil
}

func startSupervisedWorker(
	ctx context.Context,
	workers *sync.WaitGroup,
	logger *slog.Logger,
	name string,
	worker func(context.Context) error,
) {
	workers.Add(1)
	go func() {
		defer workers.Done()
		for {
			err := runWorker(ctx, worker)
			if ctx.Err() != nil {
				return
			}
			logger.Error("Worker exited unexpectedly; restarting", slog.String("worker", name), slog.Any("error", err))
			select {
			case <-ctx.Done():
				return
			case <-time.After(time.Second):
			}
		}
	}()
}

func runWorker(ctx context.Context, worker func(context.Context) error) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("panic: %v", recovered)
		}
	}()
	return worker(ctx)
}

func runActivator(ctx context.Context, cfg *config.Config, deletionActive *atomic.Bool, logger *slog.Logger) error {
	ticker := time.NewTicker(cfg.LoggerInterval)
	defer ticker.Stop()

	for {
		usage, err := disk.UsageFromStatfs(cfg.PVCMountPath)
		if err != nil {
			logger.Warn("Cannot read PVC usage", slog.Any("error", err))
		} else {
			if usage.UsagePercent >= cfg.CleanupThreshold && !deletionActive.Load() {
				deletionActive.Store(true)
				logger.Warn(
					"Deletion activated",
					slog.Float64("usagePercent", usage.UsagePercent),
					slog.Float64("threshold", cfg.CleanupThreshold),
				)
			} else if usage.UsagePercent <= cfg.TargetThreshold && deletionActive.Load() {
				deletionActive.Store(false)
				logger.Info(
					"Deletion deactivated",
					slog.Float64("usagePercent", usage.UsagePercent),
					slog.Float64("threshold", cfg.TargetThreshold),
				)
			}
		}

		select {
		case <-ctx.Done():
			deletionActive.Store(false)
			return nil
		case <-ticker.C:
		}
	}
}

func waitForMount(ctx context.Context, path string, logger *slog.Logger) error {
	deadline := time.Now().Add(60 * time.Second)
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		if _, err := os.Stat(path); err == nil {
			logger.Info("PVC mount path is ready", slog.String("path", path))
			return nil
		} else if !os.IsNotExist(err) {
			logger.Warn("Cannot check PVC mount path", slog.String("path", path), slog.Any("error", err))
		}

		if time.Now().After(deadline) {
			return fmt.Errorf("PVC mount path %q was not available after 60s: %w", path, os.ErrNotExist)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			logger.Info("Waiting for PVC mount", slog.String("path", path))
		}
	}
}
