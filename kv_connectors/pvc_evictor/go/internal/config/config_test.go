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

func TestEnvironmentOverridesFlags(t *testing.T) {
	t.Setenv("PVC_MOUNT_PATH", "/env-cache")
	t.Setenv("CACHE_DIRECTORY", "env-models")
	t.Setenv("CLEANUP_THRESHOLD", "80")
	t.Setenv("TARGET_THRESHOLD", "60")
	t.Setenv("DRY_RUN", "false")
	t.Setenv("NUM_CRAWLER_PROCESSES", "2")

	cfg, err := FromEnvAndFlags([]string{
		"--pvc-mount-path", "/flag-cache",
		"--cache-directory", "flag-models",
		"--cleanup-threshold", "90",
		"--target-threshold", "50",
		"--dry-run",
		"--num-crawler-processes", "4",
	})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.PVCMountPath != "/env-cache" || cfg.CacheDirectory != "env-models" {
		t.Fatalf("paths = %q/%q, want environment values", cfg.PVCMountPath, cfg.CacheDirectory)
	}
	if cfg.CleanupThreshold != 80 || cfg.TargetThreshold != 60 || cfg.DryRun {
		t.Fatalf("thresholds/dry-run = %v/%v/%v, want 80/60/false", cfg.CleanupThreshold, cfg.TargetThreshold, cfg.DryRun)
	}
	if cfg.NumCrawlerProcesses != 2 {
		t.Fatalf("crawler count = %d, want 2", cfg.NumCrawlerProcesses)
	}
}

func TestInvalidEnvironmentOverridesFlag(t *testing.T) {
	t.Setenv("CLEANUP_THRESHOLD", "not-a-number")
	t.Setenv("TARGET_THRESHOLD", "60")

	if _, err := FromEnvAndFlags([]string{"--cleanup-threshold", "90"}); err == nil {
		t.Fatal("expected invalid environment value to override the flag")
	}
}

func TestFlagsUsedWhenEnvironmentIsUnset(t *testing.T) {
	t.Setenv("PVC_MOUNT_PATH", "")
	t.Setenv("CACHE_DIRECTORY", "")
	t.Setenv("CLEANUP_THRESHOLD", "")
	t.Setenv("TARGET_THRESHOLD", "")

	cfg, err := FromEnvAndFlags([]string{
		"--pvc-mount-path", "/flag-cache",
		"--cache-directory", "flag-models",
		"--cleanup-threshold", "90",
		"--target-threshold", "50",
	})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.PVCMountPath != "/flag-cache" || cfg.CacheDirectory != "flag-models" ||
		cfg.CleanupThreshold != 90 || cfg.TargetThreshold != 50 {
		t.Fatalf("configuration did not use flags: %+v", cfg)
	}
}

func TestEnvIntRejectsNonIntegerValues(t *testing.T) {
	for _, value := range []string{"1.5", "NaN", "+Inf", "-Inf"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv("TEST_INTEGER", value)
			if _, err := envInt("TEST_INTEGER", 1); err == nil {
				t.Fatalf("expected an error for %q", value)
			}
		})
	}
}

func TestEnvIntAcceptsWholeNumbers(t *testing.T) {
	tests := map[string]int{
		"8":     8,
		"8.0":   8,
		"1e+06": 1_000_000,
		"1e6":   1_000_000,
		"-3.0":  -3,
	}
	for value, want := range tests {
		t.Run(value, func(t *testing.T) {
			t.Setenv("TEST_INTEGER", value)
			got, err := envInt("TEST_INTEGER", 1)
			if err != nil {
				t.Fatal(err)
			}
			if got != want {
				t.Fatalf("envInt(%q) = %d, want %d", value, got, want)
			}
		})
	}
}

func TestEnvIntRejectsOutOfRangeValues(t *testing.T) {
	for _, value := range []string{"1e30", "-1e30", "99999999999999999999"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv("TEST_INTEGER", value)
			if _, err := envInt("TEST_INTEGER", 1); err == nil {
				t.Fatalf("expected an error for %q", value)
			}
		})
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
			t.Setenv(test.variable, "invalid")

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
