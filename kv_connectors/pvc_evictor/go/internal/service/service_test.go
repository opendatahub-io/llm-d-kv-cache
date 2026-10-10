//nolint:testpackage // tests exercise the unexported activator and worker helpers.
package service

import (
	"context"
	"errors"
	"log/slog"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/config"
	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/disk"
)

func TestWaitForMountReady(t *testing.T) {
	path := t.TempDir()

	if err := waitForMount(context.Background(), path, testLogger()); err != nil {
		t.Fatalf("waitForMount() = %v, want nil", err)
	}
}

func TestWaitForMountCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	path := filepath.Join(t.TempDir(), "missing")

	if err := waitForMount(ctx, path, testLogger()); !errors.Is(err, context.Canceled) {
		t.Fatalf("waitForMount() = %v, want context.Canceled", err)
	}
}

func TestRunActivatorActivatesDeletion(t *testing.T) {
	path := t.TempDir()
	usage, err := disk.UsageFromStatfs(path)
	if err != nil {
		t.Fatal(err)
	}
	if usage.UsagePercent <= 0 || usage.UsagePercent >= 100 {
		t.Skipf("filesystem usage %v does not support an activation threshold test", usage.UsagePercent)
	}

	cfg := &config.Config{
		PVCMountPath:     path,
		CleanupThreshold: activationCleanupThreshold(usage.UsagePercent),
		TargetThreshold:  activationTargetThreshold(usage.UsagePercent),
		LoggerInterval:   time.Millisecond,
	}
	assertValidHysteresis(t, cfg)

	deletionActive := &atomic.Bool{}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		if err := runActivator(ctx, cfg, deletionActive, testLogger()); err != nil {
			t.Errorf("runActivator() = %v", err)
		}
		close(done)
	}()
	defer func() {
		cancel()
		<-done
	}()

	waitForCondition(t, func() bool { return deletionActive.Load() })
}

func TestRunActivatorDeactivatesDeletion(t *testing.T) {
	path := t.TempDir()
	usage, err := disk.UsageFromStatfs(path)
	if err != nil {
		t.Fatal(err)
	}
	if usage.UsagePercent <= 0 || usage.UsagePercent >= 100 {
		t.Skipf("filesystem usage %v does not support a deactivation threshold test", usage.UsagePercent)
	}

	cfg := &config.Config{
		PVCMountPath:     path,
		CleanupThreshold: deactivationCleanupThreshold(usage.UsagePercent),
		TargetThreshold:  deactivationTargetThreshold(usage.UsagePercent),
		LoggerInterval:   time.Millisecond,
	}
	assertValidHysteresis(t, cfg)

	deletionActive := &atomic.Bool{}
	deletionActive.Store(true)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		if err := runActivator(ctx, cfg, deletionActive, testLogger()); err != nil {
			t.Errorf("runActivator() = %v", err)
		}
		close(done)
	}()
	defer func() {
		cancel()
		<-done
	}()

	waitForCondition(t, func() bool { return !deletionActive.Load() })
}

func TestRunWorkerConvertsPanicToError(t *testing.T) {
	err := runWorker(context.Background(), func(context.Context) error {
		panic("worker failed")
	})
	if err == nil || !strings.Contains(err.Error(), "worker failed") {
		t.Fatalf("runWorker() = %v, want panic error", err)
	}
}

func TestStartSupervisedWorkerRestartsAfterPanic(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var calls atomic.Int32
	calledTwice := make(chan struct{})
	var workers sync.WaitGroup
	startSupervisedWorker(ctx, &workers, testLogger(), "test", func(context.Context) error {
		if calls.Add(1) == 1 {
			panic("first run failed")
		}
		close(calledTwice)
		<-ctx.Done()
		return nil
	})

	select {
	case <-calledTwice:
	case <-time.After(2 * time.Second):
		t.Fatal("worker was not restarted after panic")
	}
	cancel()
	workers.Wait()
}

func waitForCondition(t *testing.T, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("condition was not met before timeout")
}

func activationCleanupThreshold(usagePercent float64) float64 {
	return usagePercent / 2
}

func activationTargetThreshold(usagePercent float64) float64 {
	return activationCleanupThreshold(usagePercent) / 2
}

func deactivationTargetThreshold(usagePercent float64) float64 {
	return (usagePercent + 100) / 2
}

func deactivationCleanupThreshold(usagePercent float64) float64 {
	return (deactivationTargetThreshold(usagePercent) + 100) / 2
}

func assertValidHysteresis(t *testing.T, cfg *config.Config) {
	t.Helper()
	if cfg.TargetThreshold >= cfg.CleanupThreshold {
		t.Fatalf("invalid hysteresis: target %v is not below cleanup %v", cfg.TargetThreshold, cfg.CleanupThreshold)
	}
}

func testLogger() *slog.Logger {
	return slog.New(slog.DiscardHandler)
}
