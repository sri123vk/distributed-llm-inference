package main

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

//go:embed index.html
var dashboard []byte

const maxBody = 1 << 20

type backend struct {
	url     *url.URL
	active  atomic.Int64
	healthy atomic.Bool
}

type gateway struct {
	backends []*backend
	probe    *http.Client // short timeout for health and stats
	upstream *http.Client // no overall timeout: generations and streams can be long
}

func newGateway(urls []*url.URL) *gateway {
	g := &gateway{
		probe: &http.Client{Timeout: 2 * time.Second},
		upstream: &http.Client{Transport: &http.Transport{
			DialContext:         (&net.Dialer{Timeout: 2 * time.Second}).DialContext,
			MaxIdleConnsPerHost: 64,
			IdleConnTimeout:     90 * time.Second,
		}},
	}
	for _, u := range urls {
		g.backends = append(g.backends, &backend{url: u})
	}
	return g
}

// checkHealth probes every worker concurrently and updates the cached healthy set.
func (g *gateway) checkHealth(ctx context.Context) {
	var wg sync.WaitGroup
	for _, b := range g.backends {
		wg.Add(1)
		go func(b *backend) {
			defer wg.Done()
			req, _ := http.NewRequestWithContext(ctx, http.MethodGet, b.url.String()+"/healthz", nil)
			resp, err := g.probe.Do(req)
			if err != nil {
				b.healthy.Store(false)
				return
			}
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			b.healthy.Store(resp.StatusCode == http.StatusOK)
		}(b)
	}
	wg.Wait()
}

func (g *gateway) healthLoop(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			g.checkHealth(ctx)
		}
	}
}

// pick returns the healthy worker with the fewest in-flight requests, skipping tried ones.
func (g *gateway) pick(tried map[*backend]bool) *backend {
	var best *backend
	for _, b := range g.backends {
		if tried[b] || !b.healthy.Load() {
			continue
		}
		if best == nil || b.active.Load() < best.active.Load() {
			best = b
		}
	}
	return best
}

var hopHeaders = map[string]bool{"Connection": true, "Keep-Alive": true, "Proxy-Connection": true,
	"Transfer-Encoding": true, "Te": true, "Trailer": true, "Upgrade": true}

func (g *gateway) completions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBody))
	if err != nil {
		http.Error(w, "request body too large or unreadable", http.StatusBadRequest)
		return
	}
	tried := map[*backend]bool{}
	for {
		b := g.pick(tried)
		if b == nil {
			http.Error(w, "no healthy workers", http.StatusServiceUnavailable)
			return
		}
		tried[b] = true
		if g.forward(w, r, b, body) {
			return
		}
	}
}

// forward sends one attempt to b. It returns false only when nothing has been written
// to the client, so the request can be retried on another worker.
func (g *gateway) forward(w http.ResponseWriter, r *http.Request, b *backend, body []byte) bool {
	b.active.Add(1)
	defer b.active.Add(-1)
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, b.url.String()+r.URL.Path, bytes.NewReader(body))
	if err != nil {
		http.Error(w, "bad worker URL", http.StatusInternalServerError)
		return true
	}
	req.Header.Set("Content-Type", r.Header.Get("Content-Type"))
	resp, err := g.upstream.Do(req)
	if err != nil {
		if r.Context().Err() != nil {
			return true // client went away; nothing to retry for
		}
		b.healthy.Store(false) // the health loop restores it once the worker answers again
		log.Printf("worker %s failed: %v; retrying elsewhere", b.url, err)
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusServiceUnavailable {
		io.Copy(io.Discard, resp.Body)
		log.Printf("worker %s at capacity; retrying elsewhere", b.url)
		return false
	}
	for key, values := range resp.Header {
		if !hopHeaders[key] {
			w.Header()[key] = values
		}
	}
	w.WriteHeader(resp.StatusCode)
	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 32<<10)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return true
			}
			if flusher != nil {
				flusher.Flush() // forward each SSE chunk immediately
			}
		}
		if err != nil {
			if !errors.Is(err, io.EOF) {
				log.Printf("worker %s stream interrupted: %v", b.url, err)
			}
			return true
		}
	}
}

func (g *gateway) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(dashboard)
	})
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok\n")) })
	mux.HandleFunc("/info", func(w http.ResponseWriter, r *http.Request) {
		for _, b := range g.backends {
			if !b.healthy.Load() {
				continue
			}
			resp, err := g.probe.Get(b.url.String() + "/info")
			if err != nil {
				continue
			}
			var info map[string]interface{}
			err = json.NewDecoder(resp.Body).Decode(&info)
			resp.Body.Close()
			if resp.StatusCode == 200 && err == nil {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(info)
				return
			}
		}
		http.Error(w, "no healthy workers", http.StatusServiceUnavailable)
	})
	mux.HandleFunc("/v1/completions", g.completions)
	mux.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		type workerStat struct {
			URL      string         `json:"url"`
			InFlight int64          `json:"in_flight"`
			Healthy  bool           `json:"healthy"`
			Cache    map[string]int `json:"cache,omitempty"`
		}
		stats := make([]workerStat, len(g.backends))
		var wg sync.WaitGroup
		for i, b := range g.backends {
			stats[i] = workerStat{URL: b.url.String(), InFlight: b.active.Load(), Healthy: b.healthy.Load()}
			if !stats[i].Healthy {
				continue
			}
			wg.Add(1)
			go func(stat *workerStat) {
				defer wg.Done()
				resp, err := g.probe.Get(stat.URL + "/stats")
				if err != nil {
					return
				}
				defer resp.Body.Close()
				if resp.StatusCode != 200 || json.NewDecoder(resp.Body).Decode(&stat.Cache) != nil {
					stat.Cache = nil
				}
			}(&stats[i])
		}
		wg.Wait()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(stats)
	})
	return mux
}

func main() {
	listen := flag.String("listen", ":8000", "listen address")
	workers := flag.String("workers", os.Getenv("WORKERS"), "comma-separated worker URLs")
	interval := flag.Duration("health-interval", time.Second, "worker health check interval")
	flag.Parse()
	var urls []*url.URL
	for _, raw := range strings.Split(*workers, ",") {
		if strings.TrimSpace(raw) == "" {
			continue
		}
		parsed, err := url.Parse(strings.TrimSpace(raw))
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
			log.Fatalf("invalid worker URL %q", raw)
		}
		urls = append(urls, parsed)
	}
	if len(urls) == 0 {
		log.Fatal("provide at least one worker with --workers or WORKERS")
	}
	g := newGateway(urls)
	g.checkHealth(context.Background())
	go g.healthLoop(context.Background(), *interval)
	log.Printf("gateway listening on %s with %d workers", *listen, len(urls))
	log.Fatal(http.ListenAndServe(*listen, g.handler()))
}
