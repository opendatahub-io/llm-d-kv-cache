package config

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	DefaultPVCMountPath              = "/kv-cache"
	DefaultCleanupThreshold          = 85.0
	DefaultTargetThreshold           = 70.0
	DefaultCacheDirectory            = "kv/model-cache/models"
	DefaultDryRun                    = false
	DefaultLogLevel                  = "INFO"
	DefaultNumCrawlerProcesses       = 8
	DefaultLoggerInterval            = 500 * time.Millisecond
	DefaultFileQueueMaxSize          = 10000
	DefaultFileQueueMinSize          = 1000
	DefaultDeletionBatchSize         = 100
	DefaultDeletionMaxFilesPerSecond = 0.0
	DefaultFileAccessTimeThreshold   = time.Hour
	DefaultHexBucketLen              = 3
	DefaultEnableDirCleanup          = true
	DefaultDirCleanupTTL             = 120 * time.Second
	DefaultStorageEventsEndpoint     = ""

	// MaxFileQueueSize caps FILE_QUEUE_MAXSIZE. Both queues are Go channels,
	// and a channel reserves memory for its full size at startup.
	MaxFileQueueSize = 1_000_000
)

type Config struct {
	PVCMountPath              string
	CacheDirectory            string
	CleanupThreshold          float64
	TargetThreshold           float64
	DryRun                    bool
	LogLevel                  string
	NumCrawlerProcesses       int
	LoggerInterval            time.Duration
	FileQueueMaxSize          int
	FileQueueMinSize          int
	DeletionBatchSize         int
	DeletionMaxFilesPerSecond float64
	FileAccessTimeThreshold   time.Duration
	HexBucketLen              int
	EnableDirCleanup          bool
	DirCleanupTTL             time.Duration
	LogFilePath               string
	StorageEventsEndpoint     string
}

func FromEnv() (Config, error) {
	cleanupThreshold, err := envFloat("CLEANUP_THRESHOLD", DefaultCleanupThreshold)
	if err != nil {
		return Config{}, fmt.Errorf("invalid CLEANUP_THRESHOLD: %w", err)
	}
	targetThreshold, err := envFloat("TARGET_THRESHOLD", DefaultTargetThreshold)
	if err != nil {
		return Config{}, fmt.Errorf("invalid TARGET_THRESHOLD: %w", err)
	}
	dryRun, err := envBool("DRY_RUN", DefaultDryRun)
	if err != nil {
		return Config{}, fmt.Errorf("invalid DRY_RUN: %w", err)
	}
	numCrawlerProcesses, err := envInt("NUM_CRAWLER_PROCESSES", DefaultNumCrawlerProcesses)
	if err != nil {
		return Config{}, fmt.Errorf("invalid NUM_CRAWLER_PROCESSES: %w", err)
	}
	loggerIntervalSeconds, err := envFloat("LOGGER_INTERVAL_SECONDS", durationToSeconds(DefaultLoggerInterval))
	if err != nil {
		return Config{}, fmt.Errorf("invalid LOGGER_INTERVAL_SECONDS: %w", err)
	}
	fileQueueMaxSize, err := envInt("FILE_QUEUE_MAXSIZE", DefaultFileQueueMaxSize)
	if err != nil {
		return Config{}, fmt.Errorf("invalid FILE_QUEUE_MAXSIZE: %w", err)
	}
	fileQueueMinSize, err := envInt("FILE_QUEUE_MIN_SIZE", DefaultFileQueueMinSize)
	if err != nil {
		return Config{}, fmt.Errorf("invalid FILE_QUEUE_MIN_SIZE: %w", err)
	}
	deletionBatchSize, err := envInt("DELETION_BATCH_SIZE", DefaultDeletionBatchSize)
	if err != nil {
		return Config{}, fmt.Errorf("invalid DELETION_BATCH_SIZE: %w", err)
	}
	deletionMaxFilesPerSecond, err := envFloat("DELETION_MAX_FILES_PER_SECOND", DefaultDeletionMaxFilesPerSecond)
	if err != nil {
		return Config{}, fmt.Errorf("invalid DELETION_MAX_FILES_PER_SECOND: %w", err)
	}
	accessThresholdMinutes, err := envFloat("FILE_ACCESS_TIME_THRESHOLD_MINUTES", durationToMinutes(DefaultFileAccessTimeThreshold))
	if err != nil {
		return Config{}, fmt.Errorf("invalid FILE_ACCESS_TIME_THRESHOLD_MINUTES: %w", err)
	}
	hexBucketLen, err := envInt("HEX_BUCKET_LEN", DefaultHexBucketLen)
	if err != nil {
		return Config{}, fmt.Errorf("invalid HEX_BUCKET_LEN: %w", err)
	}
	dirCleanupTTLSeconds, err := envFloat("DIR_CLEANUP_TTL_SECONDS", durationToSeconds(DefaultDirCleanupTTL))
	if err != nil {
		return Config{}, fmt.Errorf("invalid DIR_CLEANUP_TTL_SECONDS: %w", err)
	}
	enableDirCleanup, err := envBool("ENABLE_DIR_CLEANUP", DefaultEnableDirCleanup)
	if err != nil {
		return Config{}, fmt.Errorf("invalid ENABLE_DIR_CLEANUP: %w", err)
	}

	cfg := Config{
		PVCMountPath:              envString("PVC_MOUNT_PATH", DefaultPVCMountPath),
		CacheDirectory:            envString("CACHE_DIRECTORY", DefaultCacheDirectory),
		CleanupThreshold:          cleanupThreshold,
		TargetThreshold:           targetThreshold,
		DryRun:                    dryRun,
		LogLevel:                  envString("LOG_LEVEL", DefaultLogLevel),
		NumCrawlerProcesses:       numCrawlerProcesses,
		LoggerInterval:            secondsToDuration(loggerIntervalSeconds),
		FileQueueMaxSize:          fileQueueMaxSize,
		FileQueueMinSize:          fileQueueMinSize,
		DeletionBatchSize:         deletionBatchSize,
		DeletionMaxFilesPerSecond: deletionMaxFilesPerSecond,
		FileAccessTimeThreshold:   minutesToDuration(accessThresholdMinutes),
		HexBucketLen:              hexBucketLen,
		EnableDirCleanup:          enableDirCleanup,
		DirCleanupTTL:             secondsToDuration(dirCleanupTTLSeconds),
		LogFilePath:               os.Getenv("LOG_FILE_PATH"),
		StorageEventsEndpoint:     envString("STORAGE_EVENTS_ENDPOINT", DefaultStorageEventsEndpoint),
	}

	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c *Config) Validate() error {
	switch {
	case c.PVCMountPath == "":
		return fmt.Errorf("PVC_MOUNT_PATH must not be empty")
	case c.CacheDirectory == "":
		return fmt.Errorf("CACHE_DIRECTORY must not be empty")
	}
	cleanCacheDirectory := filepath.Clean(c.CacheDirectory)
	if filepath.IsAbs(c.CacheDirectory) || cleanCacheDirectory == "." || cleanCacheDirectory == ".." || strings.HasPrefix(cleanCacheDirectory, ".."+string(filepath.Separator)) {
		return fmt.Errorf("CACHE_DIRECTORY must be a relative subdirectory of PVC_MOUNT_PATH, got %q", c.CacheDirectory)
	}
	if math.IsNaN(c.CleanupThreshold) || math.IsInf(c.CleanupThreshold, 0) {
		return fmt.Errorf("CLEANUP_THRESHOLD must be a finite number, got %v", c.CleanupThreshold)
	}
	if c.CleanupThreshold < 0 || c.CleanupThreshold > 100 {
		return fmt.Errorf("CLEANUP_THRESHOLD must be between 0 and 100, got %v", c.CleanupThreshold)
	}
	if math.IsNaN(c.TargetThreshold) || math.IsInf(c.TargetThreshold, 0) {
		return fmt.Errorf("TARGET_THRESHOLD must be a finite number, got %v", c.TargetThreshold)
	}
	if c.TargetThreshold < 0 || c.TargetThreshold > 100 {
		return fmt.Errorf("TARGET_THRESHOLD must be between 0 and 100, got %v", c.TargetThreshold)
	}
	if c.TargetThreshold >= c.CleanupThreshold {
		return fmt.Errorf("TARGET_THRESHOLD (%v) must be below CLEANUP_THRESHOLD (%v)", c.TargetThreshold, c.CleanupThreshold)
	}
	if !validCrawlerCount(c.NumCrawlerProcesses) {
		return fmt.Errorf("NUM_CRAWLER_PROCESSES must be a power of 2 from 1 to 16, got %d", c.NumCrawlerProcesses)
	}
	if c.LoggerInterval <= 0 {
		return fmt.Errorf("LOGGER_INTERVAL_SECONDS must be greater than 0, got %v", c.LoggerInterval)
	}
	if c.FileQueueMaxSize <= 0 || c.FileQueueMaxSize > MaxFileQueueSize {
		return fmt.Errorf("FILE_QUEUE_MAXSIZE must be between 1 and %d, got %d", MaxFileQueueSize, c.FileQueueMaxSize)
	}
	if c.FileQueueMinSize < 0 || c.FileQueueMinSize > c.FileQueueMaxSize {
		return fmt.Errorf("FILE_QUEUE_MIN_SIZE must be between 0 and FILE_QUEUE_MAXSIZE, got %d", c.FileQueueMinSize)
	}
	if c.DeletionBatchSize <= 0 {
		return fmt.Errorf("DELETION_BATCH_SIZE must be greater than 0, got %d", c.DeletionBatchSize)
	}
	if c.DeletionMaxFilesPerSecond < 0 || math.IsNaN(c.DeletionMaxFilesPerSecond) || math.IsInf(c.DeletionMaxFilesPerSecond, 0) {
		return fmt.Errorf("DELETION_MAX_FILES_PER_SECOND must be a finite non-negative number, got %v", c.DeletionMaxFilesPerSecond)
	}
	if c.FileAccessTimeThreshold < 0 {
		return fmt.Errorf("FILE_ACCESS_TIME_THRESHOLD_MINUTES must not be negative")
	}
	if c.HexBucketLen < 2 || c.HexBucketLen > 4 {
		return fmt.Errorf("HEX_BUCKET_LEN must be between 2 and 4, got %d", c.HexBucketLen)
	}
	if c.DirCleanupTTL < 0 {
		return fmt.Errorf("DIR_CLEANUP_TTL_SECONDS must not be negative")
	}
	return nil
}

func validCrawlerCount(count int) bool {
	return count == 1 || count == 2 || count == 4 || count == 8 || count == 16
}

func envString(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func envFloat(name string, fallback float64) (float64, error) {
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0, err
	}
	return parsed, nil
}

func envInt(name string, fallback int) (int, error) {
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0, err
	}
	return int(parsed), nil
}

func envBool(name string, fallback bool) (bool, error) {
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return false, err
	}
	return parsed, nil
}

func secondsToDuration(seconds float64) time.Duration {
	return time.Duration(seconds * float64(time.Second))
}

func durationToSeconds(value time.Duration) float64 {
	return float64(value) / float64(time.Second)
}

func minutesToDuration(minutes float64) time.Duration {
	return time.Duration(minutes * float64(time.Minute))
}

func durationToMinutes(value time.Duration) float64 {
	return float64(value) / float64(time.Minute)
}
