"""Run two case-faithful baseline replications and matched BLIP evaluation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def dump(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'running', 'pid': os.getpid(), 'jobs': {},
             'started_at': datetime.now(timezone.utc).isoformat()}
    environment = dict(os.environ, PYTHONIOENCODING='utf-8', KERAS_BACKEND='torch')
    blip_python = ROOT / '.venv-blip/Scripts/python.exe'
    blip_experiment = ROOT / 'runs/blip-tuning-20261022/lr-5em06'
    common_evaluation = [str(blip_python), '-u', str(ROOT / 'evaluate_blip.py'),
                         '--experiment', str(blip_experiment), '--feature-cache-dir',
                         str(ROOT / 'runs/blip-full-random80-20261022'),
                         '--max-new-tokens', '24', '--batch-size', '16']
    jobs = []
    for backbone in ['efficientnet', 'densenet']:
        jobs.append((backbone, [sys.executable, '-u', str(ROOT / 'train_cnn_comparison.py'),
                               '--backbone', backbone, '--output', str(args.output / backbone)]))
    jobs.extend([
        ('blip-pretrained-greedy', common_evaluation + ['--checkpoint', str(ROOT / 'models/blip-base-source-82a3776'),
                                                      '--num-beams', '1', '--output', str(args.output / 'blip-pretrained-greedy')]),
        ('blip-finetuned-greedy', common_evaluation + ['--num-beams', '1', '--output', str(args.output / 'blip-finetuned-greedy')]),
        ('blip-finetuned-beam3', common_evaluation + ['--num-beams', '3', '--batch-size', '4',
                                                    '--output', str(args.output / 'blip-finetuned-beam3')]),
    ])
    try:
        for name, command in jobs:
            state['current_job'] = name
            state['jobs'][name] = {'status': 'running', 'command': command}
            dump(args.output / 'status.json', state)
            print(f'START {name}', flush=True)
            with (args.output / f'{name}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{name} failed with code {result.returncode}; see {name}.log')
            state['jobs'][name]['status'] = 'complete'
            dump(args.output / 'status.json', state)
            print(f'COMPLETE {name}', flush=True)
        state['status'] = 'complete'
        state['finished_at'] = datetime.now(timezone.utc).isoformat()
        dump(args.output / 'status.json', state)
    except Exception as error:
        state['status'] = 'failed'
        state['error'] = str(error)
        dump(args.output / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
