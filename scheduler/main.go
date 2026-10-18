// GPU-aware kube-scheduler extender using the standard Filter/Prioritize HTTP API.
package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"strconv"
	"time"
)

var gpuResource = "nvidia.com/gpu"

const telemetryTTL = 30 * time.Second

type container struct {
	Resources struct {
		Limits   map[string]string `json:"limits"`
		Requests map[string]string `json:"requests"`
	} `json:"resources"`
	RestartPolicy string `json:"restartPolicy"`
}

type node struct {
	raw      json.RawMessage
	Metadata struct {
		Name        string            `json:"name"`
		Labels      map[string]string `json:"labels"`
		Annotations map[string]string `json:"annotations"`
	} `json:"metadata"`
	Status struct {
		Allocatable map[string]string `json:"allocatable"`
	} `json:"status"`
}

func (n *node) UnmarshalJSON(data []byte) error {
	type plain node
	if err := json.Unmarshal(data, (*plain)(n)); err != nil {
		return err
	}
	n.raw = append(n.raw[:0], data...)
	return nil
}
func (n node) MarshalJSON() ([]byte, error) {
	if n.raw != nil {
		return n.raw, nil
	}
	type plain node
	return json.Marshal(plain(n))
}

type args struct {
	Pod struct {
		Metadata struct {
			Annotations map[string]string `json:"annotations"`
		} `json:"metadata"`
		Spec struct {
			Containers     []container `json:"containers"`
			InitContainers []container `json:"initContainers"`
		} `json:"spec"`
	} `json:"pod"`
	Nodes *struct {
		Items []node `json:"items"`
	} `json:"nodes"`
}
type filterResult struct {
	Nodes struct {
		Items []node `json:"items"`
	} `json:"nodes"`
	FailedNodes map[string]string `json:"failedNodes"`
	Error       string            `json:"error,omitempty"`
}
type hostPriority struct {
	Host  string `json:"host"`
	Score int    `json:"score"`
}

func number(m map[string]string, key string) int {
	n, _ := strconv.Atoi(m[key])
	return n
}
func gpuDemand(a args) int {
	demandFor := func(c container) int {
		if _, exists := c.Resources.Requests[gpuResource]; exists {
			return number(c.Resources.Requests, gpuResource)
		}
		return number(c.Resources.Limits, gpuResource)
	}
	demand := 0
	for _, c := range a.Pod.Spec.Containers {
		demand += demandFor(c)
	}
	sidecars, initPeak := 0, 0
	for _, c := range a.Pod.Spec.InitContainers {
		count := demandFor(c)
		if c.RestartPolicy == "Always" {
			sidecars += count
			count = 0
		}
		if count+sidecars > initPeak {
			initPeak = count + sidecars
		}
	}
	demand += sidecars
	if initPeak > demand {
		demand = initPeak
	}
	return demand
}
func eligible(a args, n node) (bool, string) {
	demand := gpuDemand(a)
	if demand < 1 {
		return true, ""
	}
	if number(n.Status.Allocatable, gpuResource) < demand {
		return false, "insufficient allocatable GPUs"
	}
	stamp, err := strconv.ParseInt(n.Metadata.Annotations["inference.example.com/telemetry-at-unix"], 10, 64)
	age := time.Since(time.Unix(stamp, 0))
	if err != nil || age > telemetryTTL || age < -5*time.Second {
		return false, "GPU telemetry missing or stale"
	}
	wantedVRAM := number(a.Pod.Metadata.Annotations, "inference.example.com/vram-mib")
	if raw := a.Pod.Metadata.Annotations["inference.example.com/vram-mib"]; raw != "" && wantedVRAM <= 0 {
		return false, "invalid requested GPU memory"
	}
	if wantedVRAM > 0 && number(n.Metadata.Annotations, "inference.example.com/free-vram-mib") < wantedVRAM {
		return false, "insufficient free GPU memory"
	}
	return true, ""
}
func score(n node) int {
	// Prioritize spare VRAM and fewer active requests; cap within scheduler's 0..10 range.
	freeVRAM := number(n.Metadata.Annotations, "inference.example.com/free-vram-mib")
	active := number(n.Metadata.Annotations, "inference.example.com/active-requests")
	s := freeVRAM/4096 + 5 - active
	if s < 0 {
		return 0
	}
	if s > 10 {
		return 10
	}
	return s
}
func decode(w http.ResponseWriter, r *http.Request) (args, bool) {
	var a args
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", 405)
		return a, false
	}
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	if err := json.NewDecoder(r.Body).Decode(&a); err != nil || a.Nodes == nil {
		http.Error(w, "invalid ExtenderArgs: nodes list required", 400)
		return a, false
	}
	return a, true
}
func serveFilter(w http.ResponseWriter, r *http.Request) {
	a, ok := decode(w, r)
	if !ok {
		return
	}
	out := filterResult{FailedNodes: map[string]string{}}
	out.Nodes.Items = []node{}
	for _, n := range a.Nodes.Items {
		if pass, reason := eligible(a, n); pass {
			out.Nodes.Items = append(out.Nodes.Items, n)
		} else {
			out.FailedNodes[n.Metadata.Name] = reason
		}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}
func servePrioritize(w http.ResponseWriter, r *http.Request) {
	a, ok := decode(w, r)
	if !ok {
		return
	}
	out := []hostPriority{}
	for _, n := range a.Nodes.Items {
		out = append(out, hostPriority{n.Metadata.Name, score(n)})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}
func main() {
	listen := flag.String("listen", ":8002", "listen address")
	flag.StringVar(&gpuResource, "gpu-resource", gpuResource, "GPU extended resource name")
	flag.Parse()
	http.HandleFunc("/filter", serveFilter)
	http.HandleFunc("/prioritize", servePrioritize)
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok\n")) })
	log.Printf("scheduler extender listening on %s", *listen)
	log.Fatal(http.ListenAndServe(*listen, nil))
}
