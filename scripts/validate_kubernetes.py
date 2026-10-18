"""Validate real Kubernetes scheduling with explicitly simulated GPU inventory.

Uses only the dedicated kind-inference-lab context and local image builds.
"""
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
import urllib.request
import yaml

ROOT = Path(__file__).resolve().parents[1]
KUBE = ['kubectl', '--kubeconfig', str(ROOT / '.run/kubeconfig')]
DOCKER = ['docker', '--context', 'colima-inference']
ENV = dict(os.environ, DOCKER_CONTEXT='colima-inference')
RESOURCE = 'inference.example.com/gpu'
NAMESPACE = 'llm-inference'


def run(command, **kwargs):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True, **kwargs)
    return result.stdout


def apply(documents):
    return run(KUBE + ['apply', '-f', '-'], input=yaml.safe_dump_all(documents))


def main():
    assert run(KUBE + ['config', 'current-context']).strip() == 'kind-inference-lab'
    version = json.loads(run(KUBE + ['version', '-o', 'json']))['serverVersion']['gitVersion']
    print('Cluster version: ' + version, flush=True)
    build = ROOT / '.run/container-build'
    build.mkdir(exist_ok=True)
    (build / 'Dockerfile').write_text('FROM scratch\nCOPY app /app\nENTRYPOINT ["/app"]\n')
    arch = 'arm64' if platform.machine() in ('arm64', 'aarch64') else 'amd64'
    for component in ('gateway', 'scheduler'):
        run(['go', 'build', '-o', str(build / 'app'), './' + component],
            env=dict(os.environ, GOOS='linux', GOARCH=arch, CGO_ENABLED='0'))
        run(DOCKER + ['build', '-t', 'inference-' + component + ':local', str(build)])
    run(DOCKER + ['build', '-f', 'worker/Dockerfile.reference', '-t', 'inference-worker:local', '.'])
    for component in ('gateway', 'scheduler', 'worker'):
        run(['kind', 'load', 'docker-image', 'inference-' + component + ':local', '--name', 'inference-lab'], env=ENV)
    print('Local images loaded', flush=True)
    apply([{'apiVersion': 'v1', 'kind': 'Namespace', 'metadata': {'name': NAMESPACE}}])
    documents = list(yaml.safe_load_all((ROOT / 'k8s/platform.yaml').read_text()))
    for document in documents:
        document['metadata']['namespace'] = NAMESPACE
        if document['kind'] in ('StatefulSet', 'Deployment'):
            containers = document['spec']['template']['spec']['containers']
            for container in containers:
                name = container['name']
                component = {'worker': 'worker', 'gateway': 'gateway', 'extender': 'scheduler'}[name]
                container['image'] = 'inference-' + component + ':local'
                container['imagePullPolicy'] = 'IfNotPresent'
                if name == 'extender':
                    container['args'] = ['--gpu-resource', RESOURCE]
    apply(documents)
    config = (ROOT / 'k8s/custom-scheduler.yaml').read_text().replace('REPLACE_KUBERNETES_VERSION', version).replace('nvidia.com/gpu', RESOURCE)
    apply(list(yaml.safe_load_all(config)))
    for resource in ('deployment/inference-gateway', 'deployment/gpu-scheduler-extender',
                     'deployment/inference-scheduler', 'statefulset/inference-worker'):
        print('Waiting for ' + resource, flush=True)
        run(KUBE + ['-n', NAMESPACE, 'rollout', 'status', resource, '--timeout=180s'])
    nodes = json.loads(run(KUBE + ['get', 'nodes', '-o', 'json']))['items']
    workers = sorted(n['metadata']['name'] for n in nodes
                     if 'node-role.kubernetes.io/control-plane' not in n['metadata'].get('labels', {}))
    assert len(workers) == 2
    for node in workers:
        patch = {'status': {'capacity': {RESOURCE: '2'}, 'allocatable': {RESOURCE: '2'}}}
        run(KUBE + ['patch', 'node', node, '--subresource=status', '--type=merge', '-p', json.dumps(patch)])
        run(KUBE + ['label', 'node', node, 'inference.example.com/test-node=true', '--overwrite'])

    def publish():
        for node, memory in zip(workers, (4000, 16000)):
            run(KUBE + ['annotate', 'node', node,
                'inference.example.com/free-vram-mib=' + str(memory),
                'inference.example.com/active-requests=0',
                'inference.example.com/telemetry-at-unix=' + str(int(time.time())), '--overwrite'])

    publish()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(5):
            publish()

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    forward = None
    suffix = str(int(time.time()))
    try:
        def pod(name, memory):
            return {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'name': name, 'namespace': NAMESPACE,
                'annotations': {'inference.example.com/vram-mib': str(memory)}},
                'spec': {'schedulerName': 'inference-scheduler',
                         'nodeSelector': {'inference.example.com/test-node': 'true'},
                         'containers': [{'name': 'pause', 'image': 'registry.k8s.io/pause:3.10',
                                         'resources': {'limits': {RESOURCE: '1'}}}]}}
        fit_name, reject_name = 'gpu-fit-' + suffix, 'gpu-too-large-' + suffix
        apply([pod(fit_name, 12000), pod(reject_name, 999999)])
        assigned = None
        rejection = None
        for _ in range(60):
            fit = json.loads(run(KUBE + ['-n', NAMESPACE, 'get', 'pod', fit_name, '-o', 'json']))
            too_large = json.loads(run(KUBE + ['-n', NAMESPACE, 'get', 'pod', reject_name, '-o', 'json']))
            assigned = fit['spec'].get('nodeName')
            rejection = next((c.get('message') for c in too_large.get('status', {}).get('conditions', [])
                              if c['type'] == 'PodScheduled' and c['status'] == 'False'), None)
            if assigned and rejection:
                break
            time.sleep(1)
        assert assigned == workers[1], (assigned, workers)
        assert rejection and 'insufficient free GPU memory' in rejection, rejection
        forward = subprocess.Popen(KUBE + ['-n', NAMESPACE, 'port-forward', 'svc/inference-gateway', '18080:8000'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                with urllib.request.urlopen('http://127.0.0.1:18080/healthz', timeout=1):
                    break
            except Exception:
                time.sleep(.2)
        request = urllib.request.Request('http://127.0.0.1:18080/v1/completions',
            json.dumps({'prompt': 'Cluster test', 'max_tokens': 4}).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            generation = json.load(response)
        assert generation['tokens'] == 4
        report = {'kubernetes_version': version, 'cluster': 'kind-inference-lab',
                  'gpu_inventory': 'simulated; no physical GPU on this Mac',
                  'expected_node': workers[1], 'scheduled_node': assigned,
                  'insufficient_memory_rejection': rejection,
                  'gateway_generation': generation, 'worker_backend': 'reference'}
        (ROOT / '.run/kubernetes-validation.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        stop.set()
        heartbeat_thread.join(timeout=10)
        if forward:
            forward.terminate()
            forward.wait(timeout=5)


if __name__ == '__main__':
    main()
