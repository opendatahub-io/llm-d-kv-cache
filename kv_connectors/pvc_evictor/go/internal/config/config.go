package config

import (
	"flag"
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
	return FromEnvAndFlags(nil)
}

func FromEnvAndFlags(args []string) (Config, error) {
	flags := flag.NewFlagSet("pvc-evictor", flag.ContinueOnError)
	pvcMountPath := flags.String(
		"pvc-mount-path", DefaultPVCMountPath, "filesystem mount path",
	)
	cacheDirectory := flags.String(
		"cache-directory", DefaultCacheDirectory, "relative cache directory below the mount path",
	)
	cleanupThreshold := flags.Float64(
		"cleanup-threshold", DefaultCleanupThreshold, "disk usage percentage that starts deletion",
	)
	targetThreshold := flags.Float64(
		"target-threshold", DefaultTargetThreshold, "disk usage percentage that stops deletion",
	)
	dryRun := flags.Bool("dry-run", DefaultDryRun, "log deletion decisions without deleting files")
	logLevel := flags.String("log-level", DefaultLogLevel, "log level: DEBUG, INFO, WARNING, or ERROR")
	numCrawlerProcesses := flags.Int("num-crawler-processes", DefaultNumCrawlerProcesses, "number of crawler workers")
	loggerIntervalSeconds := flags.Float64(
		"logger-interval-seconds", durationToSeconds(DefaultLoggerInterval), "disk usage check interval in seconds",
	)
	fileQueueMaxSize := flags.Int("file-queue-max-size", DefaultFileQueueMaxSize, "maximum file queue size")
	fileQueueMinSize := flags.Int("file-queue-min-size", DefaultFileQueueMinSize, "minimum file queue size")
	deletionBatchSize := flags.Int("deletion-batch-size", DefaultDeletionBatchSize, "files per deletion batch")
	deletionMaxFilesPerSecond := flags.Float64(
		"deletion-max-files-per-second", DefaultDeletionMaxFilesPerSecond, "maximum deletion rate; zero means unlimited",
	)
	accessThresholdMinutes := flags.Float64(
		"file-access-time-threshold-minutes",
		durationToMinutes(DefaultFileAccessTimeThreshold),
		"protect files accessed within this many minutes",
	)
	hexBucketLen := flags.Int("hex-bucket-len", DefaultHexBucketLen, "hex bucket name length")
	enableDirCleanup := flags.Bool("enable-dir-cleanup", DefaultEnableDirCleanup, "remove eligible empty directories")
	dirCleanupTTLSeconds := flags.Float64(
		"dir-cleanup-ttl-seconds", durationToSeconds(DefaultDirCleanupTTL), "empty-directory quiet period in seconds",
	)
	logFilePath := flags.String("log-file-path", "", "additional log file path")
	storageEventsEndpoint := flags.String(
		"storage-events-endpoint", DefaultStorageEventsEndpoint, "ZeroMQ endpoint for BlockRemoved events",
	)

	if err := flags.Parse(args); err != nil {
		return Config{}, err
	}

	cleanupThresholdValue, err := floatSetting(
		flags, "cleanup-threshold", *cleanupThreshold, "CLEANUP_THRESHOLD", DefaultCleanupThreshold,
	)
	if err != nil {
		return Config{}, fmt.Errorf("invalid CLEANUP_THRESHOLD: %w", err)
	}
	targetThresholdValue, err := floatSetting(
		flags, "target-threshold", *targetThreshold, "TARGET_THRESHOLD", DefaultTargetThreshold,
	)
	if err != nil {
		return Config{}, fmt.Errorf("invalid TARGET_THRESHOLD: %w", err)
	}
	dryRunValue, err := boolSetting(flags, "dry-run", *dryRun, "DRY_RUN", DefaultDryRun)
	if err != nil {
		return Config{}, fmt.Errorf("invalid DRY_RUN: %w", err)
	}
	numCrawlerProcessesValue, err := intSetting(
		flags, "num-crawler-processes", *numCrawlerProcesses, "NUM_CRAWLER_PROCESSES", DefaultNumCrawlerProcesses,
	)
	if err != nil {
		return Config{}, fmt.Errorf("invalid NUM_CRAWLER_PROCESSES: %w", err)
	}
	loggerIntervalSecondsValue, err := floatSetting(
		flags, "logger-interval-seconds", *loggerIntervalSeconds, "LOGGER_INTERVAL_SECONDS", durationToSeconds(DefaultLoggerInterval),
	)
	if err != nil {
		return Config{}, fmt.Errorf("invalid LOGGER_INTERVAL_SECONDS: %w", err)
	}
	fileQueueMaxSizeValue, err := intSetting(
		flags, "file-queue-max-size", *fileQueueMaxSize, "FILE_QUEUE_MAXSIZE", DefaultFileQueueMaxSize,
	)
	if err != nil {
		return Config{}, fmt.Errorf("invalid FILE_QUEUE_MAXSIZE: %w", err)
	}
	fileQueueMinSizeValue, err := intSetting(
		flags, "file-queue-min-size", *fileQueueMinSize, "FILE_QUEUE_MIN_SIZE", DefaultFileQueueMinSize,
	)
	if err != nil {
		return Config{}, fmt.Errorf("invalid FILE_QUEUE_MIN_SIZE: %w", err)
	}
	deletionBatchSizeValue, err := intSetting(
		flags, "deletion-batch-size", *deletionBatchSize, "DELETION_BATCH_SIZE", DefaultDeletionBatchSize,
	)
	if err != nil {
		return Config{}, fmt.Errorf("invalid DELETION_BATCH_SIZE: %w", err)
	}
	deletionMaxFilesPerSecondValue, err := floatSetting(
		flags,
		"deletion-max-files-per-second",
		*deletionMaxFilesPerSecond,
		"DELETION_MAX_FILES_PER_SECOND",
		DefaultDeletionMaxFilesPerSecond,
	)
	if err != nil {
		return Config{}, fmt.Errorf("invalid DELETION_MAX_FILES_PER_SECOND: %w", err)
	}
	accessThresholdMinutesValue, err := floatSetting(
		flags,
		"file-access-time-threshold-minutes",
		*accessThresholdMinutes,
		"FILE_ACCESS_TIME_THRESHOLD_MINUTES",
		durationToMinutes(DefaultFileAccessTimeThreshold),
	)
	if err != nil {
		return Config{}, fmt.Errorf("invalid FILE_ACCESS_TIME_THRESHOLD_MINUTES: %w", err)
	}
	hexBucketLenValue, err := intSetting(flags, "hex-bucket-len", *hexBucketLen, "HEX_BUCKET_LEN", DefaultHexBucketLen)
	if err != nil {
		return Config{}, fmt.Errorf("invalid HEX_BUCKET_LEN: %w", err)
	}
	dirCleanupTTLSecondsValue, err := floatSetting(
		flags, "dir-cleanup-ttl-seconds", *dirCleanupTTLSeconds, "DIR_CLEANUP_TTL_SECONDS", durationToSeconds(DefaultDirCleanupTTL),
	)
	if err != nil {
		return Config{}, fmt.Errorf("invalid DIR_CLEANUP_TTL_SECONDS: %w", err)
	}
	enableDirCleanupValue, err := boolSetting(
		flags, "enable-dir-cleanup", *enableDirCleanup, "ENABLE_DIR_CLEANUP", DefaultEnableDirCleanup,
	)
	if err != nil {
		return Config{}, fmt.Errorf("invalid ENABLE_DIR_CLEANUP: %w", err)
	}

	cfg := Config{
		PVCMountPath:              stringSetting(flags, "pvc-mount-path", *pvcMountPath, "PVC_MOUNT_PATH", DefaultPVCMountPath),
		CacheDirectory:            stringSetting(flags, "cache-directory", *cacheDirectory, "CACHE_DIRECTORY", DefaultCacheDirectory),
		CleanupThreshold:          cleanupThresholdValue,
		TargetThreshold:           targetThresholdValue,
		DryRun:                    dryRunValue,
		LogLevel:                  stringSetting(flags, "log-level", *logLevel, "LOG_LEVEL", DefaultLogLevel),
		NumCrawlerProcesses:       numCrawlerProcessesValue,
		LoggerInterval:            secondsToDuration(loggerIntervalSecondsValue),
		FileQueueMaxSize:          fileQueueMaxSizeValue,
		FileQueueMinSize:          fileQueueMinSizeValue,
		DeletionBatchSize:         deletionBatchSizeValue,
		DeletionMaxFilesPerSecond: deletionMaxFilesPerSecondValue,
		FileAccessTimeThreshold:   minutesToDuration(accessThresholdMinutesValue),
		HexBucketLen:              hexBucketLenValue,
		EnableDirCleanup:          enableDirCleanupValue,
		DirCleanupTTL:             secondsToDuration(dirCleanupTTLSecondsValue),
		LogFilePath:               stringSetting(flags, "log-file-path", *logFilePath, "LOG_FILE_PATH", ""),
		StorageEventsEndpoint: stringSetting(
			flags,
			"storage-events-endpoint",
			*storageEventsEndpoint,
			"STORAGE_EVENTS_ENDPOINT",
			DefaultStorageEventsEndpoint,
		),
	}

	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func stringSetting(_ *flag.FlagSet, _, flagValue, envName, fallback string) string {
	if value := os.Getenv(envName); value != "" {
		return value
	}
	if flagValue != fallback {
		return flagValue
	}
	return fallback
}

func floatSetting(_ *flag.FlagSet, _ string, flagValue float64, envName string, fallback float64) (float64, error) {
	if os.Getenv(envName) != "" {
		return envFloat(envName, fallback)
	}
	return flagValue, nil
}

func intSetting(_ *flag.FlagSet, _ string, flagValue int, envName string, fallback int) (int, error) {
	if os.Getenv(envName) != "" {
		return envInt(envName, fallback)
	}
	return flagValue, nil
}

func boolSetting(_ *flag.FlagSet, _ string, flagValue bool, envName string, fallback bool) (bool, error) {
	if os.Getenv(envName) != "" {
		return envBool(envName, fallback)
	}
	return flagValue, nil
}

func (c *Config) Validate() error {
	switch {
	case c.PVCMountPath == "":
		return fmt.Errorf("PVC_MOUNT_PATH must not be empty")
	case c.CacheDirectory == "":
		return fmt.Errorf("CACHE_DIRECTORY must not be empty")
	}
	cleanCacheDirectory := filepath.Clean(c.CacheDirectory)
	if filepath.IsAbs(c.CacheDirectory) ||
		cleanCacheDirectory == "." ||
		cleanCacheDirectory == ".." ||
		strings.HasPrefix(cleanCacheDirectory, ".."+string(filepath.Separator)) {
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
	if parsed, err := strconv.Atoi(value); err == nil {
		return parsed, nil
	}

	// Accept whole numbers written as 8.0 or 1e+06. The Python evictor accepts
	// them, and Helm renders large numbers in that form.
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0, err
	}
	if math.IsNaN(parsed) || math.IsInf(parsed, 0) || parsed != math.Trunc(parsed) {
		return 0, fmt.Errorf("%q is not a whole number", value)
	}
	// Above 2^53 a float64 cannot hold every whole number exactly.
	const maxExactFloat = 1 << 53
	if math.Abs(parsed) > maxExactFloat {
		return 0, fmt.Errorf("%q is out of range", value)
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
