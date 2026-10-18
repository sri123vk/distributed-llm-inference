"""Produce a source archive and a runnable free-Colab GPU validation notebook."""
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
out = root / 'artifacts'
out.mkdir(exist_ok=True)
with zipfile.ZipFile(out / 'inference-gpu-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for directory in ('worker', 'cuda', 'tests', 'benchmarks', 'scripts'):
        for path in (root / directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.write(path, 'inference/' + str(path.relative_to(root)))
    archive.write(root / 'requirements.txt', 'inference/requirements.txt')


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': text.splitlines(True)}


cells = [markdown('''# Distributed inference: real GPU validation
Choose **Runtime → Change runtime type → GPU**. Free GPU availability varies.
Upload `inference-gpu-source.zip` when prompted. This notebook compiles the custom
CUDA kernel, checks its numerical output, validates the actual pretrained model,
and compares continuous and static batching on an identical workload.
It does not create a Kubernetes GPU cluster or host a public service.
'''), code('''import subprocess
subprocess.run(['nvidia-smi'], check=True)
subprocess.run(['nvcc', '--version'], check=True)
'''), code('''from google.colab import files
from pathlib import Path
import zipfile
uploaded = files.upload()
archive = next(name for name in uploaded if name.endswith('.zip'))
with zipfile.ZipFile(archive) as z:
    destination = Path('/content').resolve()
    for member in z.infolist():
        if not (destination / member.filename).resolve().is_relative_to(destination):
            raise ValueError('Unsafe archive path')
    z.extractall(destination)
'''), code('''import os, subprocess
os.chdir('/content/inference')
os.environ['MAX_JOBS'] = '2'
subprocess.run(['python', '-m', 'venv', '/content/inference-venv'], check=True)
PY = '/content/inference-venv/bin/python'
subprocess.run([PY, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
subprocess.run([PY, 'scripts/download_model.py'], check=True)
'''), code('''subprocess.run([PY, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
subprocess.run([PY, 'scripts/validate_model.py', '--device', 'cuda', '--attention', 'cuda',
                '--output', '.run/gpu-model-validation.json'], check=True)
'''), code('''subprocess.run([PY, 'benchmarks/compare.py', '--device', 'cuda', '--attention', 'cuda',
                '--requests', '100', '--repeats', '3', '--batch', '8',
                '--output', '.run/gpu-batching-comparison.json'], check=True)
'''), code('''import shutil
shutil.make_archive('/content/inference-gpu-results', 'zip', '.run')
files.download('/content/inference-gpu-results.zip')
''')]
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
    'accelerator': 'GPU', 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}, 'cells': cells}
(out / 'GPU-validation.ipynb').write_text(json.dumps(notebook, indent=2))
print(out)
