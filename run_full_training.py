"""Run both full-data experiments sequentially and persist progress to disk."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / '案例5-自动图像描述/案例5-代码和数据/fullname_lab.py'
STATUS = ROOT / 'runs/full-training-status.json'


def save_status(state):
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temporary.replace(STATUS)


def main():
    STATUS.parent.mkdir(exist_ok=True)
    state = {'status': 'running', 'experiments': {}}
    environment = dict(os.environ, TF_CPP_MIN_LOG_LEVEL='2', TF_NUM_INTRAOP_THREADS='8', TF_NUM_INTEROP_THREADS='2', PYTHONIOENCODING='utf-8')
    for backbone in ['efficientnet', 'densenet']:
        folder = ROOT / 'runs' / f'full-{backbone}'
        folder.mkdir(exist_ok=True)
        state['current_model'] = backbone
        state['experiments'][backbone] = {'status': 'training', 'output': str(folder)}
        save_status(state)
        command = [sys.executable, '-u', str(SCRIPT), 'train', '--backbone', backbone, '--epochs', '30', '--batch-size', '32', '--cache-features', '--output', str(folder)]
        with (folder / 'training.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            state['status'] = 'failed'
            state['experiments'][backbone]['status'] = 'failed'
            state['experiments'][backbone]['exit_code'] = result.returncode
            save_status(state)
            return result.returncode
        results = json.loads((folder / 'results.json').read_text())
        state['experiments'][backbone].update(status='trained', results=results)
        save_status(state)
        split = json.loads((folder / 'split.json').read_text())
        samples = []
        for image in split['validation'][:3]:
            prediction = subprocess.run([sys.executable, str(SCRIPT), 'predict', '--output', str(folder), '--image', image], cwd=ROOT, env=environment, capture_output=True, text=True, encoding='utf-8')
            if prediction.returncode:
                (folder / 'prediction-error.log').write_text(prediction.stderr, encoding='utf-8')
                state['status'] = 'failed'
                state['experiments'][backbone]['status'] = 'prediction_failed'
                save_status(state)
                return prediction.returncode
            samples.append({'image': image, 'caption': prediction.stdout.strip()})
        (folder / 'predictions.json').write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding='utf-8')
        state['experiments'][backbone]['status'] = 'complete'
        save_status(state)
    state['status'] = 'complete'
    save_status(state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
