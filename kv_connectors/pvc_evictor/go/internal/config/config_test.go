//nolint:testpackage // tests exercise the environment parser and validator together.
package config

import (
	"math"
	"testing"
	"time"
)

func TestFromEnvInvalidNumber(t *testing.T) {
	t.Setenv("PVC_MOUNT_PATH", "/cache")
	t.Setenv("CACHE_DIRECTORY", "models")
	t.Setenv("CLEANUP_THRESHOLD", "not-a-number")

	if _, err := FromEnv(); err == nil {
		t.Fatal("expected an error for invalid CLEANUP_THRESHOLD")
	}
}

func TestFromEnvInvalidDeletionRate(t *testing.T) {
	t.Setenv("PVC_MOUNT_PATH", "/cache")
	t.Setenv("CACHE_DIRECTORY", "models")
	t.Setenv("DELETION_MAX_FILES_PER_SECOND", "-1")

	if _, err := FromEnv(); err == nil {
		t.Fatal("expected an error for a negative deletion rate")
	}
}

func TestFromEnvInvalidBoolean(t *testing.T) {
	tests := []struct {
		name     string
		variable string
	}{
		{name: "dry run", variable: "DRY_RUN"},
		{name: "directory cleanup", variable: "ENABLE_DIR_CLEANUP"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Setenv("PVC_MOUNT_PATH", "/cache")
			t.Setenv("CACHE_DIRECTORY", "models")
			t.Setenv(test.variable, "tru")

			if _, err := FromEnv(); err == nil {
				t.Fatalf("expected an error for invalid %s", test.variable)
			}
		})
	}
}

func TestValidate(t *testing.T) {
	valid := Config{
		PVCMountPath:              "/cache",
		CacheDirectory:            "models",
		CleanupThreshold:          85,
		TargetThreshold:           70,
		NumCrawlerProcesses:       8,
		LoggerInterval:            time.Second,
		FileQueueMaxSize:          100,
		FileQueueMinSize:          10,
		DeletionBatchSize:         10,
		DeletionMaxFilesPerSecond: 0,
		FileAccessTimeThreshold:   time.Minute,
		HexBucketLen:              3,
		DirCleanupTTL:             time.Minute,
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid configuration was rejected: %v", err)
	}

	valid.NumCrawlerProcesses = 3
	if err := valid.Validate(); err == nil {
		t.Fatal("expected an error for an invalid crawler count")
	}

	valid.NumCrawlerProcesses = 8
	valid.FileQueueMaxSize = MaxFileQueueSize + 1
	if err := valid.Validate(); err == nil {
		t.Fatal("expected an error for a FILE_QUEUE_MAXSIZE above the limit")
	}
}

func validBaseConfig() Config {
	return Config{
		PVCMountPath:            "/cache",
		CacheDirectory:          "models",
		CleanupThreshold:        85,
		TargetThreshold:         70,
		NumCrawlerProcesses:     8,
		LoggerInterval:          time.Second,
		FileQueueMaxSize:        100,
		FileQueueMinSize:        10,
		DeletionBatchSize:       10,
		FileAccessTimeThreshold: time.Minute,
		HexBucketLen:            3,
		DirCleanupTTL:           time.Minute,
	}
}

func TestValidateDeletionBatchSize(t *testing.T) {
	tests := []struct {
		size    int
		wantErr bool
	}{
		{size: 1},
		{size: 100},
		{size: 1_000_000_000},
		{size: 0, wantErr: true},
		{size: -1, wantErr: true},
	}

	for _, test := range tests {
		cfg := validBaseConfig()
		cfg.DeletionBatchSize = test.size
		if err := cfg.Validate(); (err != nil) != test.wantErr {
			t.Errorf("DeletionBatchSize=%d: error = %v, wantErr %v", test.size, err, test.wantErr)
		}
	}
}

func TestValidateDeletionRate(t *testing.T) {
	tests := []struct {
		name    string
		rate    float64
		wantErr bool
	}{
		{name: "no limit", rate: 0},
		{name: "typical", rate: 500},
		{name: "very low", rate: 1e-10},
		{name: "very high", rate: 1e12},
		{name: "negative", rate: -1, wantErr: true},
		{name: "NaN", rate: math.NaN(), wantErr: true},
		{name: "positive infinity", rate: math.Inf(1), wantErr: true},
		{name: "negative infinity", rate: math.Inf(-1), wantErr: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := validBaseConfig()
			cfg.DeletionMaxFilesPerSecond = test.rate
			if err := cfg.Validate(); (err != nil) != test.wantErr {
				t.Fatalf("DeletionMaxFilesPerSecond=%v: error = %v, wantErr %v", test.rate, err, test.wantErr)
			}
		})
	}
}

func TestValidateRejectsNonFiniteThresholds(t *testing.T) {
	base := Config{
		PVCMountPath:            "/cache",
		CacheDirectory:          "models",
		CleanupThreshold:        85,
		TargetThreshold:         70,
		NumCrawlerProcesses:     8,
		LoggerInterval:          time.Second,
		FileQueueMaxSize:        100,
		FileQueueMinSize:        10,
		DeletionBatchSize:       10,
		FileAccessTimeThreshold: time.Minute,
		HexBucketLen:            3,
		DirCleanupTTL:           time.Minute,
	}

	base.CleanupThreshold = math.NaN()
	if err := base.Validate(); err == nil {
		t.Fatal("expected an error for NaN CLEANUP_THRESHOLD")
	}

	base.CleanupThreshold = 85
	base.TargetThreshold = math.NaN()
	if err := base.Validate(); err == nil {
		t.Fatal("expected an error for NaN TARGET_THRESHOLD")
	}
}

func TestValidateRejectsCacheDirectoryOutsidePVC(t *testing.T) {
	base := Config{
		PVCMountPath:        "/cache",
		CacheDirectory:      "models",
		CleanupThreshold:    85,
		TargetThreshold:     70,
		NumCrawlerProcesses: 8,
		LoggerInterval:      time.Second,
		FileQueueMaxSize:    100,
		FileQueueMinSize:    10,
		DeletionBatchSize:   10,
		HexBucketLen:        3,
		DirCleanupTTL:       time.Minute,
	}

	for _, directory := range []string{"../outside", "models/../../outside", ".", "/outside"} {
		t.Run(directory, func(t *testing.T) {
			cfg := base
			cfg.CacheDirectory = directory
			if err := cfg.Validate(); err == nil {
				t.Fatalf("Validate() accepted CACHE_DIRECTORY=%q", directory)
			}
		})
	}
}
