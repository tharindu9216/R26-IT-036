"""Modal cloud runner for the stress classification training pipeline.

This script prepares a Modal cloud environment, installs the required packages,
uploads the project files, requests a GPU, and runs train.py on Modal.

The trained models, logs, metrics, and plots are saved into a Modal volume so
they can be downloaded after training.
"""
import modal, os, re
from pathlib import Path


def find_project_root():
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / '.env').exists() and (candidate / 'ai_components').exists():
            return candidate

    modal_project_root = Path('/root/stress_detection')
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
        remote_path='/root/stress_detection',
        ignore=[
            '__pycache__/**',
            '.ipynb_checkpoints/**',
            '.git/**',
            'local_results/**',
            '*.parquet',
        ],
    )
)

volume = modal.Volume.from_name('stress-outputs', create_if_missing=True)
app    = modal.App('stress-detection')


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
    print('  STRESS DETECTION — MODAL CLOUD')
    print('  MUNASINGHE M.A.C.D | IT22252586')
    print('=' * 60)

    # GPU info + auto batch size
    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'GPU  : {gpu}')
        print(f'VRAM : {vram:.1f} GB')
        batch = 64 if 'H100' in gpu else 32 if 'A100' in gpu else 32  # L4
    else:
        print('WARNING: No GPU!')
        batch = 8

    remote_project_root = '/root/stress_detection'
    remote_train_dir = (
        remote_project_root
        + '/ai_components/c3_text_stressor_distortion/Stress header/train'
    )
    remote_data_dir = remote_project_root + '/data/Stress header/processed'

    # Set paths
    os.environ['DATA_DIR']   = remote_data_dir
    os.environ['OUTPUT_DIR'] = '/results/training_outputs/'
    os.makedirs('/results/training_outputs/', exist_ok=True)

    # Patch batch size in config
    cfg_path = os.path.join(remote_train_dir, 'config.py')
    cfg      = open(cfg_path).read()
    cfg      = re.sub(r"'batch_size'\s*:\s*\d+,",
                      f"'batch_size'       : {batch},",
                      cfg)
    open(cfg_path, 'w').write(cfg)
    print(f'batch_size → {batch}')

    # Verify files
    required = [
        os.path.join(remote_data_dir, 'dreaddit_train.csv'),
        os.path.join(remote_data_dir, 'dreaddit_val.csv'),
        os.path.join(remote_data_dir, 'dreaddit_test.csv'),
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
    print(' Saved to Modal Volume: stress-outputs')

    # List saved files
    out = '/results/training_outputs/'
    if os.path.exists(out):
        files = sorted(os.listdir(out))
        print(f'\nSaved ({len(files)} files):')
        for fn in files:
            sz = os.path.getsize(os.path.join(out, fn)) / (1024*1024)
            print(f'  {fn:50s} {sz:.1f} MB')

    # Results summary
    rp = '/results/training_outputs/all_results.json'
    if os.path.exists(rp):
        data = json.load(open(rp))
        print('\n' + '=' * 65)
        print(f'{"Model":22s} {"Test F1":>9s} {"Acc":>8s} {"MCC":>8s}')
        print('-' * 50)
        for name, res in sorted(
                data.items(),
                key=lambda x: x[1].get('test_metrics', {}).get('f1_macro', 0),
                reverse=True):
            m = res.get('test_metrics', {})
            print(f'{name:22s} '
                  f'{m.get("f1_macro",0):>9.4f} '
                  f'{m.get("accuracy",0):>8.4f} '
                  f'{m.get("mcc",0):>8.4f}')

    return {'status': 'success' if result.returncode == 0 else 'failed',
            'hours' : round(elapsed / 3600, 2)}


@app.local_entrypoint()
def main():
    print('\nSubmitting to Modal...')
    r = run_training.spawn().get()
    print(f'\nDone! Status={r["status"]} Time={r["hours"]}h')
    print('Download: mkdir -p local_results && '
          'modal volume get stress-outputs training_outputs ./local_results/')
