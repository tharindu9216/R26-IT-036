"""Run Stress Header BERTopic training on a Modal GPU."""

import os
from pathlib import Path

import modal


def find_project_root():
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (
            (candidate / 'ai_components').exists()
            and (candidate / 'data').exists()
        ):
            return candidate
    remote_root = Path('/root/stress_bertopic')
    return remote_root if remote_root.exists() else here.parent


PROJECT_ROOT = find_project_root()
ENV_PATH = PROJECT_ROOT / '.env'


def load_env_value(key, env_path=ENV_PATH):
    if key in os.environ:
        return os.environ[key]
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        name, value = line.split('=', 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return None


hf_token = load_env_value('HF_TOKEN')
modal_gpu = (
    load_env_value('BERTOPIC_GPU')
    or load_env_value('MODAL_GPU')
    or 'B200'
)
modal_secrets = (
    [modal.Secret.from_dict({'HF_TOKEN': hf_token})]
    if hf_token
    else []
)

image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install(
        'torch==2.7.1',
        index_url='https://download.pytorch.org/whl/cu128',
    )
    .pip_install(
        'transformers==4.40.0',
        'sentence-transformers==3.0.1',
        # BERTopic 0.17.4 imports StaticEmbedding, which is unavailable in
        # sentence-transformers 3.0.1. This compatible pin avoids that crash.
        'bertopic==0.16.4',
        'umap-learn==0.5.6',
        'hdbscan==0.8.40',
        'scikit-learn==1.4.2',
        'pandas==2.2.2',
        'numpy==1.26.4',
        'scipy==1.13.0',
        'matplotlib==3.8.4',
        'seaborn==0.13.2',
        'plotly==5.22.0',
        'safetensors==0.4.3',
        'huggingface-hub==0.23.0',
    )
    .run_commands(
        'python -c "from bertopic.backend._sentencetransformers import '
        "SentenceTransformerBackend; print('BERTopic backend OK')\""
    )
    .run_commands(
        'python -c "from sentence_transformers import SentenceTransformer; '
        "SentenceTransformer('sentence-transformers/all-mpnet-base-v2'); "
        "print('all-mpnet-base-v2 cached')\""
    )
    .add_local_dir(
        PROJECT_ROOT,
        remote_path='/root/stress_bertopic',
        ignore=[
            '__pycache__/**',
            '.ipynb_checkpoints/**',
            '.git/**',
            '.env',
            '.env.*',
            '**/.env',
            '**/.env.*',
            '.agents/**',
            '.codex/**',
            '.venv/**',
            'venv/**',
            'models/**',
            'reports/**',
            'data/**/raw/**',
            'frontend/node_modules/**',
            '*.parquet',
        ],
    )
)

volume = modal.Volume.from_name(
    'stress-bertopic-outputs', create_if_missing=True)
app = modal.App('stress-header-bertopic')


@app.function(
    gpu=modal_gpu,
    image=image,
    timeout=3600 * 3,
    secrets=modal_secrets,
    volumes={'/results': volume},
)
def run_training():
    import json
    import os
    import subprocess
    import sys
    import time

    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if 'B200' in gpu_name or 'B300' in gpu_name:
            batch_size = 512
        elif any(
                name in gpu_name
                for name in ('H200', 'H100', 'A100')
        ):
            batch_size = 256
        else:
            batch_size = 64
        print(f'GPU: {gpu_name} ({vram:.1f} GB)')
    else:
        batch_size = 32
        print('WARNING: BERTopic is running without a GPU')

    remote_root = '/root/stress_bertopic'
    train_dir = (
        remote_root
        + '/ai_components/c3_text_stressor_distortion/'
        + 'Stress_header_BERTopic'
    )
    data_dir = remote_root + '/data/Stress header/processed'
    output_dir = '/results/stress_bertopic_outputs'
    os.environ['DATA_DIR'] = data_dir
    os.environ['OUTPUT_DIR'] = output_dir
    os.environ['BERTOPIC_BATCH_SIZE'] = str(batch_size)
    os.makedirs(output_dir, exist_ok=True)

    required = [
        os.path.join(data_dir, 'dreaddit_train.csv'),
        os.path.join(data_dir, 'dreaddit_val.csv'),
        os.path.join(data_dir, 'dreaddit_test.csv'),
    ]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing Stress data: {path}')

    print(f'Embedding batch size: {batch_size}')
    started = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(train_dir, 'train.py')],
        cwd=train_dir,
        capture_output=False,
    )
    elapsed = time.time() - started
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(
            f'Stress BERTopic failed with exit code {result.returncode}. '
            'Partial outputs were committed to stress-bertopic-outputs.'
        )

    print('\nSaved files:')
    for root, _, filenames in os.walk(output_dir):
        for filename in sorted(filenames):
            path = os.path.join(root, filename)
            relative = os.path.relpath(path, output_dir)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f'  {relative:60s} {size_mb:.2f} MB')

    metrics_path = os.path.join(output_dir, 'topic_metrics.json')
    with open(metrics_path, encoding='utf-8') as metrics_file:
        metrics = json.load(metrics_file)
    filter_path = os.path.join(output_dir, 'fit_filter_summary.json')
    with open(filter_path, encoding='utf-8') as filter_file:
        fit_filter = json.load(filter_file)
    print('\nTopic summary:')
    print(
        f'  Topics: {metrics["topic_count_excluding_outliers"]} | '
        f'Outliers: {metrics["outlier_rate"]:.2%} | '
        f'Diversity: {metrics["topic_diversity_top_10"]:.4f}')
    print(
        f'  Fit quality filter: {fit_filter["included_documents"]} included | '
        f'{fit_filter["excluded_documents"]} excluded')
    return {
        'status': 'success',
        'minutes': round(elapsed / 60, 2),
        'topics': metrics['topic_count_excluding_outliers'],
    }


@app.local_entrypoint()
def main():
    print(f'Submitting Stress BERTopic to Modal GPU: {modal_gpu}')
    result = run_training.spawn().get()
    print(
        f'Done: {result["topics"]} topics in '
        f'{result["minutes"]} minutes')
    print(
        'Download: modal volume get stress-bertopic-outputs '
        'stress_bertopic_outputs ./stress_bertopic_results/')
