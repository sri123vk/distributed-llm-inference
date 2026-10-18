package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

const payload = `{"pod":{"metadata":{"annotations":{"inference.example.com/vram-mib":"12000"}},"spec":{"containers":[{"resources":{"limits":{"nvidia.com/gpu":"1"}}}]}},"nodes":{"items":[{"metadata":{"name":"gpu-a","annotations":{"inference.example.com/free-gpus":"1","inference.example.com/free-vram-mib":"16000","inference.example.com/active-requests":"1"}},"status":{"allocatable":{"nvidia.com/gpu":"1"}}},{"metadata":{"name":"gpu-b","annotations":{"inference.example.com/free-gpus":"1","inference.example.com/free-vram-mib":"8000"}},"status":{"allocatable":{"nvidia.com/gpu":"1"}}}]}}`

func TestFilterAndPrioritize(t *testing.T) {
	fresh := strings.ReplaceAll(payload, `"inference.example.com/free-gpus":"1"`, `"inference.example.com/telemetry-at-unix":"`+strconv.FormatInt(time.Now().Unix(), 10)+`"`)
	req := httptest.NewRequest(http.MethodPost, "/filter", bytes.NewBufferString(fresh))
	out := httptest.NewRecorder()
	serveFilter(out, req)
	if out.Code != 200 {
		t.Fatalf("status %d", out.Code)
	}
	var result filterResult
	if err := json.Unmarshal(out.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Nodes.Items) != 1 || result.Nodes.Items[0].Metadata.Name != "gpu-a" {
		t.Fatalf("unexpected nodes: %+v", result.Nodes.Items)
	}
	if result.FailedNodes["gpu-b"] != "insufficient free GPU memory" {
		t.Fatalf("failure reason: %+v", result.FailedNodes)
	}
	req = httptest.NewRequest(http.MethodPost, "/prioritize", bytes.NewBufferString(fresh))
	out = httptest.NewRecorder()
	servePrioritize(out, req)
	var scores []hostPriority
	if err := json.Unmarshal(out.Body.Bytes(), &scores); err != nil {
		t.Fatal(err)
	}
	if len(scores) != 2 || scores[0].Score <= scores[1].Score {
		t.Fatalf("unexpected scores: %+v", scores)
	}
}

func TestStaleTelemetryRejectedAndNodeFieldsPreserved(t *testing.T) {
	var a args
	if err := json.Unmarshal([]byte(payload), &a); err != nil {
		t.Fatal(err)
	}
	if ok, reason := eligible(a, a.Nodes.Items[0]); ok || reason != "GPU telemetry missing or stale" {
		t.Fatalf("expected stale telemetry rejection, got %v %s", ok, reason)
	}
	raw := `{"metadata":{"name":"node-a","annotations":{}},"spec":{"taints":[{"key":"test","effect":"NoSchedule"}]},"status":{"allocatable":{"cpu":"4"}}}`
	var n node
	if err := json.Unmarshal([]byte(raw), &n); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(n)
	if err != nil || !bytes.Contains(encoded, []byte(`"taints"`)) {
		t.Fatalf("node fields lost: %s, %v", encoded, err)
	}
}

func TestRequestsAndInitContainersDemand(t *testing.T) {
	var a args
	data := `{"pod":{"spec":{"containers":[{"resources":{"requests":{"nvidia.com/gpu":"1"}}}],"initContainers":[{"resources":{"requests":{"nvidia.com/gpu":"2"}}}]}}}`
	if err := json.Unmarshal([]byte(data), &a); err != nil {
		t.Fatal(err)
	}
	if got := gpuDemand(a); got != 2 {
		t.Fatalf("GPU demand %d, expected 2", got)
	}
}
