package events

import (
	"context"
	"encoding/binary"
	"fmt"
	"sync"
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/vmihailenco/msgpack/v5"
)

const (
	sendHighWaterMark = 100000

	// closeDrainDelay is how long Close lets a recent event leave the queue.
	closeDrainDelay = 500 * time.Millisecond
)

type Publisher struct {
	socket      zmq4.Socket
	sequence    uint64
	lastPublish time.Time
	mutex       sync.Mutex
}

func NewPublisher(ctx context.Context, endpoint string) (*Publisher, error) {
	socket := zmq4.NewPub(ctx)
	if err := socket.SetOption(zmq4.OptionHWM, sendHighWaterMark); err != nil {
		_ = socket.Close()
		return nil, fmt.Errorf("set ZeroMQ send high-water mark: %w", err)
	}
	if err := socket.Listen(endpoint); err != nil {
		_ = socket.Close()
		return nil, fmt.Errorf("listen on ZeroMQ endpoint %q: %w", endpoint, err)
	}
	return &Publisher{socket: socket}, nil
}

func (p *Publisher) PublishBlocksRemoved(ctx context.Context, blockHashes []uint64, modelName string) error {
	if len(blockHashes) == 0 || modelName == "" {
		return nil
	}

	event, err := msgpack.Marshal([]any{"BlockRemoved", blockHashes, "SHARED_STORAGE"})
	if err != nil {
		return fmt.Errorf("encode BlockRemoved event: %w", err)
	}
	payload, err := msgpack.Marshal([]any{float64(time.Now().UnixNano()) / 1e9, [][]byte{event}})
	if err != nil {
		return fmt.Errorf("encode BlockRemoved batch: %w", err)
	}

	p.mutex.Lock()
	defer p.mutex.Unlock()
	if p.socket == nil {
		return nil
	}
	p.sequence++
	sequence := make([]byte, binary.Size(uint64(0)))
	binary.BigEndian.PutUint64(sequence, p.sequence)
	topic := "kv@SHARED_STORAGE@" + modelName

	err = p.socket.Send(zmq4.NewMsgFrom([]byte(topic), sequence, payload))
	p.lastPublish = time.Now()
	return err
}

// Close shuts the socket down. If an event was published less than
// closeDrainDelay ago, it first waits out the rest of that time.
//
// Send only queues the event. The socket sends it later from a background
// goroutine, and closing the socket drops anything still queued. The ZeroMQ
// library has no flush call, so this wait is a best-effort way to let the
// final BlockRemoved events leave before shutdown.
func (p *Publisher) Close() error {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	if p.socket == nil {
		return nil
	}
	if wait := closeDrainDelay - time.Since(p.lastPublish); wait > 0 {
		time.Sleep(wait)
	}
	socket := p.socket
	p.socket = nil
	return socket.Close()
}
