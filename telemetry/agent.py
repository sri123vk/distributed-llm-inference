"""Publish real node GPU telemetry using the pod's Kubernetes service account.

Free VRAM is the minimum across GPUs, so the extender cannot assume the device
plugin will assign the particular GPU with the most free memory.
"""
import json
import os
from pathlib import Path
import ssl
import subprocess
import time
import urllib.parse
import urllib.request

SA = Path('/var/run/secrets/kubernetes.io/serviceaccount')


def api(path, body=None):
    token = (SA / 'token').read_text().strip()
    host = os.environ['KUBERNETES_SERVICE_HOST']
    port = os.environ.get('KUBERNETES_SERVICE_PORT_HTTPS', '443')
    request = urllib.request.Request('https://' + host + ':' + port + path,
        data=json.dumps(body).encode() if body is not None else None,
        method='PATCH' if body is not None else 'GET',
        headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/merge-patch+json'})
    with urllib.request.urlopen(request, context=ssl.create_default_context(cafile=str(SA / 'ca.crt')), timeout=5) as response:
        return json.load(response)


def sample(node):
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=5)
    free = [int(float(line.strip())) for line in result.stdout.splitlines() if line.strip()]
    if not free:
        raise RuntimeError('no NVIDIA GPUs visible')
    query = urllib.parse.urlencode({'fieldSelector': 'spec.nodeName=' + node,
                                   'labelSelector': 'app=inference-worker'})
    pods = api('/api/v1/pods?' + query)['items']
    active = 0
    for pod in pods:
        address = pod.get('status', {}).get('podIP')
        if not address or pod.get('status', {}).get('phase') != 'Running':
            continue
        with urllib.request.urlopen('http://' + address + ':8001/stats', timeout=2) as response:
            stats = json.load(response)
            active += stats['active'] + stats['queued']
    return {'inference.example.com/free-vram-mib': str(min(free)),
            'inference.example.com/active-requests': str(active),
            'inference.example.com/telemetry-at-unix': str(int(time.time()))}


def main():
    node = os.environ['NODE_NAME']
    while True:
        try:
            annotations = sample(node)
            api('/api/v1/nodes/' + urllib.parse.quote(node, safe=''), {'metadata': {'annotations': annotations}})
            print(json.dumps({'node': node, 'published': annotations}), flush=True)
        except Exception as exc:
            # Do not refresh timestamps on failure; the extender rejects stale data.
            print(json.dumps({'node': node, 'error': str(exc)}), flush=True)
        time.sleep(5)


if __name__ == '__main__':
    main()
