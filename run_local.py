"""Run the dashboard and two local workers: python3 run_local.py."""
import json
import argparse
import pathlib
import signal
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['llama', 'reference'], default='llama')
    args = parser.parse_args()
    runtime = ROOT / '.run'
    runtime.mkdir(exist_ok=True)
    for name in ('gateway', 'scheduler'):
        subprocess.run(['go', 'build', '-o', str(runtime / name), './' + name], cwd=ROOT, check=True)
    python = str(ROOT / '.venv' / 'bin' / 'python') if args.backend == 'llama' else 'python3'
    worker_command = [python, '-m', 'worker.server', '--backend', args.backend, '--host', '127.0.0.1']
    services = [
        ('worker-a', worker_command + ['--port', '8001'], 8001),
        ('worker-b', worker_command + ['--port', '8003'], 8003),
        ('gateway', [str(runtime / 'gateway'), '--workers', 'http://127.0.0.1:8001,http://127.0.0.1:8003', '--listen', '127.0.0.1:8000'], 8000),
        ('scheduler', [str(runtime / 'scheduler'), '--listen', '127.0.0.1:8002'], 8002),
    ]
    processes, logs = [], []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        for name, command, port in services:
            log = open(runtime / (name + '.log'), 'w')
            logs.append(log)
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log)
            processes.append(process)
            for _ in range(1200):
                if process.poll() is not None:
                    raise RuntimeError(name + ' exited; see ' + str(runtime / (name + '.log')))
                try:
                    with urllib.request.urlopen('http://127.0.0.1:%d/healthz' % port, timeout=.5) as response:
                        assert response.status == 200
                    break
                except Exception:
                    time.sleep(.1)
            else:
                raise RuntimeError(name + ' did not become ready')
            print(name + ' ready', flush=True)
        (runtime / 'pids.json').write_text(json.dumps({name: p.pid for (name, _, _), p in zip(services, processes)}, indent=2))
        print('Open http://127.0.0.1:8000 — Ctrl+C stops all services.', flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
