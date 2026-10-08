"""Modal cloud runner for Binary V4 Distortion/No-Distortion training.

This script prepares a Modal cloud environment, installs the required packages,
uploads the project files, requests a GPU, and runs train.py on Modal.

The trained models, logs, metrics, and plots are saved into a Modal volume so
they can be downloaded after training.
"""
import modal, os
from pathlib import Path


def find_project_root():
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / '.env').exists() and (candidate / 'ai_components').exists():
            return candidate

    modal_project_root = Path('/root/cbt_distortion_detection')
    if modal_project_root.exists():
        return modal_project_root

    return here.parent


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
if not hf_token:
    raise RuntimeError(
        f'HF_TOKEN not found in environment or {ENV_PATH}. '
        'Add HF_TOKEN=... to .env before running Modal.'
    )

modal_gpu = load_env_value('MODAL_GPU') or 'A100-80GB'

image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install(
        'torch==2.3.0', 'transformers==4.40.0',
        'sentencepiece', 'protobuf', 'huggingface_hub',
        'optuna==3.6.1', 'scikit-learn==1.4.2',
        'pandas==2.2.2', 'numpy==1.26.4',
        'matplotlib==3.8.4', 'seaborn==0.13.2',
    )
    .add_local_dir(
        PROJECT_ROOT,
        remote_path='/root/cbt_distortion_detection',
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
            'local_results/**',
            'models/**',
            'reports/**',
            'data/**/raw/**',
            'frontend/node_modules/**',
            '*.parquet',
        ],
    )
)

volume = modal.Volume.from_name('cbt-outputs', create_if_missing=True)
app    = modal.App('cbt-distortion-detection')


@app.function(
    gpu     = modal_gpu,     # set MODAL_GPU=A100 after adding a payment method
    image   = image,
    timeout = 3600 * 8,      # 8 hour max
    secrets = [modal.Secret.from_dict({'HF_TOKEN': hf_token})],
    volumes = {'/results': volume},
)
def run_training():
    import subprocess, sys, os, json, time, torch

    print('=' * 60)
    print('  CBT BINARY V4 — DISTORTION VS NO DISTORTION')
    print('  MUNASINGHE M.A.C.D | IT22252586')
    print('=' * 60)

    # GPU info + Optuna batch-size search space
    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'GPU  : {gpu}')
        print(f'VRAM : {vram:.1f} GB')
        batch_choices = '16,32' if (
            'H100' in gpu or 'A100' in gpu
        ) else '8,16'
    else:
        print('WARNING: No GPU!')
        batch_choices = '4,8'

    remote_project_root = '/root/cbt_distortion_detection'
    remote_shared_train_dir = (
        remote_project_root
        + '/ai_components/c3_text_stressor_distortion/CBT header/train'
    )
    remote_train_dir = (
        remote_shared_train_dir + '/binary_classification'
    )
    remote_data_dir = remote_project_root + '/data/CBT header/processed'

    # Set paths
    os.environ['DATA_DIR']   = remote_data_dir
    os.environ['CBT_BATCH_SIZE_CHOICES'] = batch_choices
    # Binary V4 uses a separate directory so all multiclass experiments
    # remain preserved for comparison.
    os.environ['OUTPUT_DIR'] = '/results/binary_v4_outputs/'
    os.makedirs('/results/binary_v4_outputs/', exist_ok=True)

    print(f'Optuna batch-size choices → {batch_choices}')

    # Verify files
    required = [
        os.path.join(remote_data_dir, 'cbt_train.csv'),
        os.path.join(remote_data_dir, 'cbt_val.csv'),
        os.path.join(remote_data_dir, 'cbt_test.csv'),
        os.path.join(remote_data_dir, 'metadata.json'),
    ]
    print('\nVerifying:')
    for p in required:
        ok   = os.path.exists(p)
        size = os.path.getsize(p) / 1024 if ok else 0
        print(f'  {"" if ok else ""} {os.path.basename(p):35s} {size:.1f} KB')
        if not ok:
            raise FileNotFoundError(f'Missing: {p}')

    # Run
    print('\n' + '=' * 60)
    print('STARTING train.py ...')
    print('=' * 60 + '\n')

    t0     = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(remote_train_dir, 'train.py')],
        capture_output=False,
        cwd=remote_train_dir,
    )
    elapsed = time.time() - t0
    h, m    = int(elapsed // 3600), int((elapsed % 3600) // 60)
    print(f'\nFinished in {h}h {m}m | Exit: {result.returncode}')

    volume.commit()
    print(' Saved to Modal Volume: cbt-outputs')

    if result.returncode != 0:
        raise RuntimeError(
            f'CBT training failed with exit code {result.returncode}. '
            'Partial logs and outputs were committed to cbt-outputs.'
        )

    # List saved files
    out = '/results/binary_v4_outputs/'
    if os.path.exists(out):
        files = sorted(os.listdir(out))
        print(f'\nSaved ({len(files)} files):')
        for fn in files:
            sz = os.path.getsize(os.path.join(out, fn)) / (1024*1024)
            print(f'  {fn:50s} {sz:.1f} MB')

    # Results summary
    rp = '/results/binary_v4_outputs/all_results.json'
    if os.path.exists(rp):
        data = json.load(open(rp))
        print('\n' + '=' * 97)
        print(f'{"Model":25s} {"Bal Acc":>10s} {"F1":>10s} '
              f'{"MCC":>10s} {"PR-AUC":>10s} {"ROC-AUC":>10s}')
        print('-' * 97)
        for name, res in sorted(
                data.items(),
                key=lambda x: x[1].get(
                    'test_metrics', {}).get('f1_macro', 0),
                reverse=True):
            m = res.get('test_metrics', {})
            print(f'{name:25s} '
                  f'{m.get("balanced_accuracy",0):>10.4f} '
                  f'{m.get("f1_macro",0):>10.4f} '
                  f'{m.get("mcc",0):>10.4f} '
                  f'{m.get("pr_auc",0):>10.4f} '
                  f'{m.get("roc_auc",0):>10.4f}')

    return {'status': 'success', 'hours': round(elapsed / 3600, 2)}


@app.local_entrypoint()
def main():
    print('\nSubmitting to Modal...')
    r = run_training.spawn().get()
    print(f'\nDone! Status={r["status"]} Time={r["hours"]}h')
    print('Download: mkdir -p binary_v4_results && '
          'modal volume get cbt-outputs binary_v4_outputs '
          './binary_v4_results/')
