package deleter

import (
	"context"
	"encoding/json"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/config"
	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/filemeta"
)

const (
	partialBatchTimeout     = 5 * time.Second
	modelMetadataRetryDelay = 5 * time.Second
	modelMetadataRetryCount = 3
)

var metadataRetryDelay = modelMetadataRetryDelay

type EventPublisher interface {
	PublishBlocksRemoved(ctx context.Context, blockHashes []uint64, modelName string) error
}

type Deleter struct {
	cfg          *config.Config
	cachePath    string
	publisher    EventPublisher
	modelNames   sync.Map
	pacer        *deletionPacer
	filesDeleted atomic.Uint64
	bytesFreed   atomic.Uint64
}

func New(cfg *config.Config, publisher EventPublisher) *Deleter {
	return &Deleter{
		cfg:       cfg,
		cachePath: filepath.Join(cfg.PVCMountPath, cfg.CacheDirectory),
		publisher: publisher,
		pacer:     newDeletionPacer(cfg.DeletionMaxFilesPerSecond),
	}
}

func (d *Deleter) Run(
	ctx context.Context,
	deletionActive *atomic.Bool,
	fileQueue <-chan string,
	folderQueue chan<- string,
	logger *slog.Logger,
) error {
	// The batch grows as files arrive. It is not sized up front, because
	// DELETION_BATCH_SIZE has no upper limit.
	var batch []string
	// batchStarted is when the first file of the current batch arrived.
	// A partial batch is flushed once it is partialBatchTimeout old, so a
	// slow trickle of files cannot hold it back until the batch is full.
	batchStarted := time.Now()
	ticker := time.NewTicker(100 * time.Millisecond)
	statusTicker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	defer statusTicker.Stop()

	for {
		if !deletionActive.Load() {
			batch = batch[:0]
			select {
			case <-ctx.Done():
				d.logStopped(logger)
				return nil
			case <-statusTicker.C:
				logger.Debug("Deletion is off")
			case <-ticker.C:
			}
			continue
		}

		select {
		case <-ctx.Done():
			d.flush(ctx, batch, folderQueue, logger)
			d.logStopped(logger)
			return nil
		case path := <-fileQueue:
			if len(batch) == 0 {
				batchStarted = time.Now()
			}
			batch = append(batch, path)
			if len(batch) >= d.cfg.DeletionBatchSize {
				d.flush(ctx, batch, folderQueue, logger)
				batch = batch[:0]
			}
		case <-ticker.C:
			if len(batch) > 0 && time.Since(batchStarted) >= partialBatchTimeout {
				d.flush(ctx, batch, folderQueue, logger)
				batch = batch[:0]
			}
		case <-statusTicker.C:
			logger.Info("Deleter status",
				slog.Uint64("filesDeleted", d.filesDeleted.Load()),
				slog.Uint64("bytesFreed", d.bytesFreed.Load()),
			)
		}
	}
}

func (d *Deleter) flush(ctx context.Context, batch []string, folderQueue chan<- string, logger *slog.Logger) {
	if len(batch) == 0 {
		return
	}
	if d.cfg.DryRun {
		d.filesDeleted.Add(uint64(len(batch)))
		logger.Debug("Dry-run deletion batch", slog.Int("files", len(batch)))
		return
	}

	events := make(map[string][]uint64)
	parents := make(map[string]struct{})
	// parentModTimes holds the mtime each parent directory had before this
	// batch deleted anything from it. Deleting a file updates the mtime of its
	// directory, so the value must be read before the first deletion. Reading
	// it once per directory also saves one stat call per file.
	//
	// If the files of one directory are split over two batches, the later batch
	// sees the mtime the earlier batch left behind and does not queue the
	// directory. The crawler removes it on a later sweep.
	parentModTimes := make(map[string]time.Time)
	var filesDeleted uint64
	var bytesFreed uint64

	for _, path := range batch {
		// On shutdown, stop deleting but still report the files already
		// deleted in this batch below. Otherwise their BlockRemoved events are lost.
		if err := d.pacer.acquire(ctx); err != nil {
			break
		}
		info, err := filemeta.Stat(path)
		if err != nil {
			if !os.IsNotExist(err) {
				logger.Warn("Cannot stat cache file", slog.String("path", path), slog.Any("error", err))
			}
			continue
		}
		if time.Since(info.AccessTime) < d.cfg.FileAccessTimeThreshold {
			continue
		}
		parent := ""
		if folderQueue != nil {
			parent = filepath.Dir(path)
			if _, seen := parentModTimes[parent]; !seen && d.cfg.DirCleanupTTL > 0 {
				parentModTimes[parent] = filemeta.DirModTime(parent)
			}
		}
		var (
			blockHash   uint64
			blockHashOK bool
			modelName   string
			modelNameOK bool
		)
		if d.publisher != nil {
			blockHash, blockHashOK = blockHashFromPath(path)
			if blockHashOK {
				modelName, modelNameOK = d.modelNameWithRetry(ctx, path, logger)
			}
		}

		if err := os.Remove(path); err != nil {
			if !os.IsNotExist(err) {
				logger.Warn("Cannot delete cache file", slog.String("path", path), slog.Any("error", err))
			}
			continue
		}
		filesDeleted++
		if info.Size < 0 {
			logger.Warn("Cache file has a negative size", slog.String("path", path), slog.Int64("size", info.Size))
			continue
		}
		bytesFreed += uint64(info.Size) //nolint:gosec // size is checked as non-negative.
		if folderQueue != nil && filemeta.DirCleanupDue(parentModTimes[parent], time.Now(), d.cfg.DirCleanupTTL) {
			parents[parent] = struct{}{}
		}

		if d.publisher != nil && blockHashOK && modelNameOK {
			events[modelName] = append(events[modelName], blockHash)
		}
	}

	for parent := range parents {
		select {
		case folderQueue <- parent:
		default:
		case <-ctx.Done():
		}
	}
	if d.publisher != nil {
		for modelName, hashes := range events {
			if err := d.publisher.PublishBlocksRemoved(ctx, hashes, modelName); err != nil {
				logger.Warn("Cannot publish BlockRemoved event", slog.String("model", modelName), slog.Any("error", err))
			}
		}
	}

	d.filesDeleted.Add(filesDeleted)
	d.bytesFreed.Add(bytesFreed)
}

type deletionPacer struct {
	interval time.Duration
	nextSlot time.Time
}

func newDeletionPacer(maxFilesPerSecond float64) *deletionPacer {
	return &deletionPacer{interval: intervalForRate(maxFilesPerSecond)}
}

// intervalForRate returns the delay between two deletions. A rate of 0 (or
// less) means no limit and gives 0. A rate above one billion per second gives
// a delay below one nanosecond, which rounds to 0, so it is also no limit.
// A rate so low that the delay does not fit in a time.Duration gets the
// longest delay a time.Duration can hold, about 292 years.
func intervalForRate(maxFilesPerSecond float64) time.Duration {
	if maxFilesPerSecond <= 0 || math.IsNaN(maxFilesPerSecond) {
		return 0
	}
	interval := float64(time.Second) / maxFilesPerSecond
	if interval >= math.MaxInt64 {
		return time.Duration(math.MaxInt64)
	}
	return time.Duration(interval)
}

func (p *deletionPacer) acquire(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if p.interval == 0 {
		return nil
	}

	now := time.Now()
	slot := p.nextSlot
	if slot.Before(now) {
		slot = now
	}
	p.nextSlot = slot.Add(p.interval)

	timer := time.NewTimer(time.Until(slot))
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (d *Deleter) logStopped(logger *slog.Logger) {
	logger.Info("Deleter stopped",
		slog.Uint64("filesDeleted", d.filesDeleted.Load()),
		slog.Uint64("bytesFreed", d.bytesFreed.Load()),
	)
}

func blockHashFromPath(path string) (uint64, bool) {
	base := filepath.Base(path)
	if !strings.HasSuffix(base, ".bin") {
		return 0, false
	}
	value := strings.TrimSuffix(base, ".bin")
	if len(value) != 16 {
		return 0, false
	}
	parsed, err := strconv.ParseUint(value, 16, 64)
	return parsed, err == nil
}

var rankDirPattern = regexp.MustCompile(`^(.+)_r\d+$`)

func (d *Deleter) modelName(path string) (string, bool) {
	configPath, ok := d.modelConfigPath(path)
	if !ok {
		return "", false
	}
	baseDir := filepath.Dir(configPath)
	if value, ok := d.modelNames.Load(baseDir); ok {
		modelName, ok := value.(string)
		if !ok {
			return "", false
		}
		return modelName, modelName != ""
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return "", false
	}
	var metadata struct {
		ModelName string `json:"model_name"`
	}
	if err := json.Unmarshal(data, &metadata); err != nil {
		return "", false
	}
	if metadata.ModelName == "" {
		return "", false
	}
	d.modelNames.Store(baseDir, metadata.ModelName)
	return metadata.ModelName, true
}

func (d *Deleter) modelConfigPath(path string) (string, bool) {
	relative, err := filepath.Rel(d.cachePath, path)
	if err != nil {
		return "", false
	}
	parts := strings.Split(relative, string(os.PathSeparator))
	if len(parts) < 2 {
		return "", false
	}
	match := rankDirPattern.FindStringSubmatch(parts[0])
	if match == nil {
		return "", false
	}
	return filepath.Join(d.cachePath, match[1], "config.json"), true
}

func (d *Deleter) modelNameWithRetry(ctx context.Context, path string, logger *slog.Logger) (string, bool) {
	configPath, _ := d.modelConfigPath(path)
	for attempt := 0; attempt <= modelMetadataRetryCount; attempt++ {
		if modelName, ok := d.modelName(path); ok {
			return modelName, true
		}
		if attempt == modelMetadataRetryCount {
			logger.Error(
				"Disabling storage event publishing because config.json is unavailable",
				slog.String("configFile", configPath),
				slog.String("cacheFile", path),
				slog.Int("retries", modelMetadataRetryCount),
			)
			d.publisher = nil
			return "", false
		}

		timer := time.NewTimer(metadataRetryDelay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return "", false
		case <-timer.C:
		}
	}
	return "", false
}
