package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type fakeWorker struct {
	server  *httptest.Server
	healthy atomic.Bool
	hits    atomic.Int64
}

func newFakeWorker(t *testing.T, completion http.HandlerFunc) *fakeWorker {
	f := &fakeWorker{}
	f.healthy.Store(true)
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if !f.healthy.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	})
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		f.hits.Add(1)
		completion(w, r)
	})
	f.server = httptest.NewServer(mux)
	t.Cleanup(f.server.Close)
	return f
}

func echo(name string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"worker":%q,"body":%s}`, name, body)
	}
}

func setup(t *testing.T, workers ...*fakeWorker) (*gateway, *httptest.Server) {
	var urls []*url.URL
	for _, f := range workers {
		u, _ := url.Parse(f.server.URL)
		urls = append(urls, u)
	}
	g := newGateway(urls)
	g.checkHealth(context.Background())
	server := httptest.NewServer(g.handler())
	t.Cleanup(server.Close)
	return g, server
}

func post(t *testing.T, base string) (int, string) {
	resp, err := http.Post(base+"/v1/completions", "application/json", strings.NewReader(`{"prompt":"hi"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(body)
}

func TestRoutesToLeastLoadedHealthyWorker(t *testing.T) {
	a, b := newFakeWorker(t, echo("a")), newFakeWorker(t, echo("b"))
	g, server := setup(t, a, b)
	g.backends[0].active.Store(5)
	if code, body := post(t, server.URL); code != 200 || !strings.Contains(body, `"worker":"b"`) || !strings.Contains(body, `"prompt":"hi"`) {
		t.Fatalf("got %d %s", code, body)
	}
	g.backends[0].active.Store(0)
	g.backends[1].active.Store(5)
	if _, body := post(t, server.URL); !strings.Contains(body, `"worker":"a"`) {
		t.Fatalf("expected worker a, got %s", body)
	}
}

func TestSkipsUnhealthyWorkers(t *testing.T) {
	a, b := newFakeWorker(t, echo("a")), newFakeWorker(t, echo("b"))
	a.healthy.Store(false)
	g, server := setup(t, a, b)
	g.backends[1].active.Store(100)
	for i := 0; i < 3; i++ {
		if _, body := post(t, server.URL); !strings.Contains(body, `"worker":"b"`) {
			t.Fatalf("routed to unhealthy worker: %s", body)
		}
	}
	if a.hits.Load() != 0 {
		t.Fatal("unhealthy worker received traffic")
	}
	b.healthy.Store(false)
	g.checkHealth(context.Background())
	if code, _ := post(t, server.URL); code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 with no healthy workers, got %d", code)
	}
}

func TestFailsOverWhenWorkerIsDown(t *testing.T) {
	dead, live := newFakeWorker(t, echo("dead")), newFakeWorker(t, echo("live"))
	g, server := setup(t, dead, live)
	dead.server.Close() // healthy in the cache, but connections now fail
	g.backends[1].active.Store(1)
	if code, body := post(t, server.URL); code != 200 || !strings.Contains(body, `"worker":"live"`) || !strings.Contains(body, `"prompt":"hi"`) {
		t.Fatalf("failover did not resend the body: %d %s", code, body)
	}
	if g.backends[0].healthy.Load() {
		t.Fatal("failed worker should be marked unhealthy")
	}
}

func TestRetriesWorkerAtCapacity(t *testing.T) {
	full := newFakeWorker(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"request queue is full"}`, http.StatusServiceUnavailable)
	})
	free := newFakeWorker(t, echo("free"))
	g, server := setup(t, full, free)
	g.backends[1].active.Store(1)
	if code, body := post(t, server.URL); code != 200 || !strings.Contains(body, `"worker":"free"`) {
		t.Fatalf("got %d %s", code, body)
	}
	if !g.backends[0].healthy.Load() {
		t.Fatal("a busy worker is still healthy")
	}
}

func TestStreamsEventsIncrementally(t *testing.T) {
	release := make(chan struct{})
	streamer := newFakeWorker(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Connection", "close")
		fmt.Fprint(w, "data: {\"token\":\"a\",\"done\":false}\n\n")
		w.(http.Flusher).Flush()
		<-release
		fmt.Fprint(w, "data: {\"token\":\"b\",\"done\":true}\n\n")
	})
	_, server := setup(t, streamer)
	resp, err := http.Post(server.URL+"/v1/completions", "application/json", strings.NewReader(`{"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("content type %q", resp.Header.Get("Content-Type"))
	}
	reader := bufio.NewReader(resp.Body)
	first := make(chan string, 1)
	go func() { line, _ := reader.ReadString('\n'); first <- line }()
	select {
	case line := <-first:
		if !strings.Contains(line, `"token":"a"`) {
			t.Fatalf("first event %q", line)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("first event was buffered instead of forwarded")
	}
	close(release)
	rest, _ := io.ReadAll(reader)
	if !strings.Contains(string(rest), `"token":"b","done":true`) {
		t.Fatalf("missing final event: %q", rest)
	}
}

func TestHealthLoopRestoresRecoveredWorker(t *testing.T) {
	a := newFakeWorker(t, echo("a"))
	a.healthy.Store(false)
	g, _ := setup(t, a)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go g.healthLoop(ctx, 10*time.Millisecond)
	a.healthy.Store(true)
	deadline := time.Now().Add(2 * time.Second)
	for !g.backends[0].healthy.Load() {
		if time.Now().After(deadline) {
			t.Fatal("worker never marked healthy")
		}
		time.Sleep(5 * time.Millisecond)
	}
}
