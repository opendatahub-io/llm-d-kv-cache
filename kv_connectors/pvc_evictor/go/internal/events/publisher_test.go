//nolint:testpackage // tests exercise the internal publisher wire helper.
package events

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/vmihailenco/msgpack/v5"
)

func TestPublisherSendsBlockRemovedEvent(t *testing.T) {
	endpoint := "inproc://pvc-evictor-block-removed-test"

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	publisher, err := NewPublisher(ctx, endpoint)
	if err != nil {
		t.Fatal(err)
	}
	defer publisher.Close()

	subscriber := zmq4.NewSub(ctx)
	defer subscriber.Close()
	if err := subscriber.SetOption(zmq4.OptionSubscribe, "kv@SHARED_STORAGE@test-model"); err != nil {
		t.Fatal(err)
	}
	if err := subscriber.Dial(endpoint); err != nil {
		t.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)

	if err := publisher.PublishBlocksRemoved(ctx, []uint64{42}, "test-model"); err != nil {
		t.Fatal(err)
	}

	messages := make(chan zmq4.Msg, 1)
	errors := make(chan error, 1)
	go func() {
		message, err := subscriber.Recv()
		if err != nil {
			errors <- err
			return
		}
		messages <- message
	}()

	select {
	case message := <-messages:
		assertBlockRemovedMessage(t, message)
	case err := <-errors:
		t.Fatal(err)
	case <-ctx.Done():
		t.Fatal("timeout waiting for storage event")
	}
}

// The service closes the publisher right after the deleter's final flush.
// Events published just before Close must still reach the subscriber.
func TestCloseDeliversRecentlyQueuedEvents(t *testing.T) {
	endpoint := "inproc://pvc-evictor-close-drain-test"
	const wantEvents = 3

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	publisher, err := NewPublisher(ctx, endpoint)
	if err != nil {
		t.Fatal(err)
	}

	subscriber := zmq4.NewSub(ctx)
	defer subscriber.Close()
	if err := subscriber.SetOption(zmq4.OptionSubscribe, "kv@SHARED_STORAGE@test-model"); err != nil {
		t.Fatal(err)
	}
	if err := subscriber.Dial(endpoint); err != nil {
		t.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)

	var received atomic.Int64
	go func() {
		for {
			if _, err := subscriber.Recv(); err != nil {
				return
			}
			received.Add(1)
		}
	}()

	for i := range wantEvents {
		if err := publisher.PublishBlocksRemoved(ctx, []uint64{uint64(i)}, "test-model"); err != nil {
			t.Fatal(err)
		}
	}
	if err := publisher.Close(); err != nil {
		t.Fatal(err)
	}

	deadline := time.Now().Add(2 * time.Second)
	for received.Load() < wantEvents && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if got := received.Load(); got != wantEvents {
		t.Fatalf("subscriber received %d events, want %d", got, wantEvents)
	}
}

func TestCloseIsIdempotentAndStopsPublishing(t *testing.T) {
	endpoint := "inproc://pvc-evictor-close-twice-test"

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	publisher, err := NewPublisher(ctx, endpoint)
	if err != nil {
		t.Fatal(err)
	}
	if err := publisher.Close(); err != nil {
		t.Fatal(err)
	}
	if err := publisher.Close(); err != nil {
		t.Fatalf("second Close returned %v, want nil", err)
	}
	if err := publisher.PublishBlocksRemoved(ctx, []uint64{1}, "test-model"); err != nil {
		t.Fatalf("publish after Close returned %v, want nil", err)
	}
}

func assertBlockRemovedMessage(t *testing.T, message zmq4.Msg) {
	t.Helper()
	if len(message.Frames) != 3 {
		t.Fatalf("got %d frames, want 3", len(message.Frames))
	}
	if topic := string(message.Frames[0]); topic != "kv@SHARED_STORAGE@test-model" {
		t.Fatalf("topic = %q", topic)
	}
	if sequence := message.Frames[1]; len(sequence) != 8 || sequence[7] != 1 {
		t.Fatalf("sequence = %v, want big-endian 1", sequence)
	}

	var batch []any
	if err := msgpack.Unmarshal(message.Frames[2], &batch); err != nil {
		t.Fatal(err)
	}
	if len(batch) != 2 {
		t.Fatalf("batch has %d fields, want 2", len(batch))
	}
	rawEvents, ok := batch[1].([]any)
	if !ok || len(rawEvents) != 1 {
		t.Fatalf("events = %#v, want one event", batch[1])
	}
	rawEvent, ok := rawEvents[0].([]byte)
	if !ok {
		t.Fatalf("event = %T, want []byte", rawEvents[0])
	}
	var event []any
	if err := msgpack.Unmarshal(rawEvent, &event); err != nil {
		t.Fatal(err)
	}
	if len(event) != 3 || event[0] != "BlockRemoved" || event[2] != "SHARED_STORAGE" {
		t.Fatalf("event = %#v", event)
	}
	hashes, ok := event[1].([]any)
	if !ok || len(hashes) != 1 || hashes[0] != uint64(42) {
		t.Fatalf("hashes = %#v, want [42]", event[1])
	}
}
