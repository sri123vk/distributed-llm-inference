"""Download the public safetensors checkpoint and record the exact revision."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = 'HuggingFaceTB/SmolLM2-135M-Instruct'
revision = '12fd25f77366fa6b3b4b768ec3050bf629380bac'
destination = ROOT / 'models' / 'smollm2-135m-instruct'
snapshot_download(REPO, revision=revision, local_dir=str(destination),
                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'], max_workers=4)
(destination / 'source.json').write_text(json.dumps({'repo': REPO, 'revision': revision}, indent=2))
print(destination)
