package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/config"
	"github.com/llm-d/llm-d-kv-cache/kv_connectors/pvc_evictor/go/internal/service"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "FATAL ERROR: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.FromEnv()
	if err != nil {
		return err
	}

	logger, closeLogger, err := newLogger(&cfg)
	if err != nil {
		return err
	}
	defer closeLogger()

	ctx, stopSignals := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stopSignals()

	logger.Info("PVC Evictor starting",
		slog.String("pvcMountPath", cfg.PVCMountPath),
		slog.String("cacheDirectory", cfg.CacheDirectory),
		slog.Int("crawlers", cfg.NumCrawlerProcesses),
		slog.Bool("dryRun", cfg.DryRun),
		slog.Bool("directoryCleanup", cfg.EnableDirCleanup),
	)

	return service.Run(ctx, &cfg, logger)
}

func newLogger(cfg *config.Config) (*slog.Logger, func(), error) {
	level, err := parseLogLevel(cfg.LogLevel)
	if err != nil {
		return nil, nil, err
	}

	writers := []io.Writer{os.Stdout}
	closeFn := func() {}

	if cfg.LogFilePath != "" {
		file, err := os.OpenFile(cfg.LogFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
		if err != nil {
			return nil, nil, fmt.Errorf("open LOG_FILE_PATH: %w", err)
		}
		writers = append(writers, file)
		closeFn = func() { _ = file.Close() }
	}

	handler := slog.NewTextHandler(io.MultiWriter(writers...), &slog.HandlerOptions{Level: level})
	return slog.New(handler), closeFn, nil
}

func parseLogLevel(value string) (slog.Level, error) {
	switch strings.ToUpper(value) {
	case "DEBUG":
		return slog.LevelDebug, nil
	case "INFO":
		return slog.LevelInfo, nil
	case "WARNING", "WARN":
		return slog.LevelWarn, nil
	case "ERROR":
		return slog.LevelError, nil
	default:
		return slog.LevelInfo, fmt.Errorf("invalid LOG_LEVEL %q", value)
	}
}
