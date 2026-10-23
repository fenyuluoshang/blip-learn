"""Monitor and finish BLIP training, adapt learning rates, and verify the winner."""
import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from blip_tuning_policy import next_learning_rate
from blip_training_data import manifest_hash

ROOT = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=8))
TRAINING_KEYS = ['seed', 'validation_fraction', 'limit', 'epochs', 'patience', 'batch_size',
                 'gradient_accumulation', 'eval_batch_size', 'cache_batch_size', 'workers',
                 'learning_rate', 'max_length', 'max_new_tokens', 'checkpoint_every', 'log_every', 'loss_group_size']


def now():
    return datetime.now(TZ).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def process_information(pid):
    """An observation error is not evidence that a process has stopped."""
    if not pid:
        return {'known': True, 'alive': False}
    if os.name == 'nt':
        command = (f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}'; "
                   "if ($p) { $p | Select-Object ProcessId,CommandLine,CreationDate | ConvertTo-Json -Compress }")
        result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)
        if result.returncode:
            return {'known': False, 'error': result.stderr}
        if not result.stdout.strip():
            return {'known': True, 'alive': False}
        record = json.loads(result.stdout)
        return {'known': True, 'alive': True, 'pid': record['ProcessId'],
                'command': record['CommandLine'], 'creation': record['CreationDate']}
    proc = Path('/proc') / str(pid)
    if not proc.exists():
        return {'known': True, 'alive': False}
    try:
        return {'known': True, 'alive': True, 'pid': pid,
                'command': (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()}
    except OSError as error:
        return {'known': False, 'error': str(error)}


def matching_training_process(observation, output):
    if not observation.get('known') or not observation.get('alive'):
        return observation
    command = (observation.get('command') or '').casefold()
    if 'train_blip.py' not in command or str(output.resolve()).casefold() not in command:
        return {'known': True, 'alive': False, 'pid_reused': True}
    return observation


def find_training_process(output):
    if os.name != 'nt':
        return None
    command = ("Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
               "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)
    if result.returncode:
        raise RuntimeError(f'Unable to observe training processes: {result.stderr}')
    records = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(records, dict):
        records = [records]
    for record in records:
        command_line = (record.get('CommandLine') or '').casefold()
        if 'train_blip.py' in command_line and str(output.resolve()).casefold() in command_line:
            return {'pid': record['ProcessId'], 'observed_at': now(), 'log': None}
    return None


def training_command(output, config, baseline, resume=False):
    command = [sys.executable, '-u', str(ROOT / 'train_blip.py'), '--output', str(output)]
    for key in TRAINING_KEYS:
        value = config.get(key)
        if value is not None:
            command += ['--' + key.replace('_', '-'), str(value)]
    if output.resolve() != baseline.resolve():
        command += ['--feature-cache-dir', str(baseline)]
    if resume:
        command.append('--resume')
    return command


def launch_training(output, config, baseline, resume=False):
    stamp = datetime.now(TZ).strftime('%Y%m%d-%H%M%S-%f')
    log = output.parent / (output.name + f'.tuning-{stamp}.log')
    environment = dict(os.environ, PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1')
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(training_command(output, config, baseline, resume), cwd=ROOT,
                                   env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    return {'pid': process.pid, 'log': str(log), 'launched_at': now(), 'resume': resume}


def audit_completed_run(output, expected):
    config, results, selection = read(output / 'config.json'), read(output / 'results.json'), read(output / 'best/selection.json')
    if results['status'] != 'complete':
        raise ValueError('Training result is not complete')
    for key in ['split_sha256', 'captions_sha256', 'source_weights_sha256', 'model_revision',
                'train_images', 'validation_images', 'train_pairs', 'validation_pairs', 'metric_protocol', 'effective_batch_size']:
        if config[key] != expected[key]:
            raise ValueError(f'Training trial changed comparison contract: {key}')
    if (config.get('loss_group_size') or config['batch_size']) != (expected.get('loss_group_size') or expected['batch_size']):
        raise ValueError('Trial changed the original loss weighting groups')
    history = results['history']
    if [row['epoch'] for row in history] != list(range(results['completed_epochs'] + 1)):
        raise ValueError('Training history is missing one or more complete epochs')
    if results['completed_epochs'] < 1:
        raise ValueError('No training epoch was completed')
    if results['completed_epochs'] < config['epochs'] and not results['early_stopped']:
        raise ValueError('Training ended without its declared termination condition')
    best = max(history, key=lambda row: row['cider'])
    if (best['epoch'] != results['best_epoch'] or selection['epoch'] != results['best_epoch']
            or not math.isclose(best['cider'], results['best_cider'], rel_tol=1e-8)
            or not math.isclose(selection['metrics']['cider'], best['cider'], rel_tol=1e-8)):
        raise ValueError('Best checkpoint selection differs from the actual validation history')
    split = read(output / 'split.json')
    if (manifest_hash(split) != expected['split_sha256']
            or len(split['train']) != expected['train_images']
            or len(split['validation']) != expected['validation_images']
            or set(split['train']) & set(split['validation'])):
        raise ValueError('Actual image split differs from the declared comparison contract')
    for row in history:
        predictions = read(output / 'predictions' / f'epoch-{row["epoch"]:02d}.json')
        if [prediction['image'] for prediction in predictions] != split['validation']:
            raise ValueError('Validation predictions do not cover exactly the declared validation images')
        if sum(len(prediction['references']) for prediction in predictions) != config['validation_pairs']:
            raise ValueError('Validation reference captions are missing')
        if not all(math.isfinite(row[key]) for key in ['cider', 'bleu1', 'bleu4', 'meteor', 'rouge_l', 'validation_loss']):
            raise ValueError('Nonfinite validation metric')
    for name in ['best/model.safetensors', 'best/preprocessor_config.json', 'last/training_state.pt']:
        if not (output / name).is_file():
            raise ValueError(f'Missing training artifact: {name}')
    return {'output': str(output.resolve()), 'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'], 'gradient_accumulation': config['gradient_accumulation'],
            'loss_group_size': config.get('loss_group_size') or config['batch_size'],
            'best_cider': results['best_cider'], 'best_epoch': results['best_epoch'],
            'completed_epochs': results['completed_epochs'], 'early_stopped': results['early_stopped'],
            'best_metrics': best, 'audit': 'passed'}


def throughput_profile(baseline, suite, args):
    destination = args.output / 'batch-throughput.json'
    suite['phase'] = 'batch_throughput_benchmark'
    write(args.output / 'status.json', suite)
    if not destination.exists():
        command = [sys.executable, '-u', str(ROOT / 'benchmark_blip_batches.py'),
                   '--baseline', str(baseline), '--output', str(destination)]
        with (args.output / 'batch-throughput.log').open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=ROOT,
                                       env=dict(os.environ, HF_HUB_OFFLINE='1', PYTHONIOENCODING='utf-8'),
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            suite['benchmark_pid'] = process.pid
            write(args.output / 'status.json', suite)
            while process.poll() is None:
                time.sleep(10)
        if process.returncode:
            raise RuntimeError('CUDA throughput benchmark failed; see batch-throughput.log')
    result = read(destination)
    if result['status'] != 'complete' or result['split_sha256'] != suite['baseline_contract']['split_sha256']:
        raise ValueError('Throughput benchmark belongs to different data or did not complete')
    suite['throughput_benchmark'] = result
    write(args.output / 'status.json', suite)
    return result['selected']


def wait_for_training(output, config, baseline, suite, args, launched=None):
    recoveries = 0
    # A just-launched trainer imports frameworks before writing status.json.
    grace_until = time.monotonic() + 120 if launched else 0
    last_progress = None
    while True:
        state = read(output / 'status.json') if (output / 'status.json').exists() else {}
        pid = state.get('pid')
        if not pid and launched:
            pid = launched['pid']
        if not pid:
            launched = find_training_process(output)
            pid = launched['pid'] if launched else None
        try:
            observation = matching_training_process(process_information(pid), output)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as error:
            observation = {'known': False, 'error': str(error)}
        suite.update(phase='waiting_for_training', active_experiment=str(output), observed_at=now(),
                     observation=observation, active_training=state, pid=os.getpid())
        write(args.output / 'status.json', suite)
        if not observation.get('known'):
            time.sleep(args.poll_seconds)
            continue
        if observation.get('alive'):
            progress = (state.get('epoch'), state.get('global_step'), state.get('phase'))
            if progress != last_progress:
                print(f'{now()} {output.name}: epoch={state.get("epoch")} '
                      f'step={state.get("global_step")} phase={state.get("phase")}', flush=True)
                last_progress = progress
            time.sleep(args.poll_seconds)
            continue
        if state.get('status') == 'complete' and (output / 'results.json').exists():
            return audit_completed_run(output, suite['baseline_contract'])
        if time.monotonic() < grace_until:
            time.sleep(args.poll_seconds)
            continue
        if (output / 'last/training_state.pt').exists() and recoveries < 2:
            recoveries += 1
            launched = launch_training(output, config, baseline, resume=True)
            suite.setdefault('recovery_events', []).append({'experiment': str(output), 'attempt': recoveries, **launched})
            # The old status contains a dead PID until loading completes. Wait for the new launcher instead.
            grace_until = time.monotonic() + 120
            while time.monotonic() < grace_until:
                updated = read(output / 'status.json')
                if updated.get('pid') != pid:
                    break
                observed = matching_training_process(process_information(launched['pid']), output)
                if observed.get('known') and not observed.get('alive'):
                    break
                time.sleep(args.poll_seconds)
            continue
        raise RuntimeError(f'Training process is absent and could not finish/recover: {output}; status={state}')


def run_evaluation(experiment, baseline, destination, beams, expected=None):
    if (destination / 'report.json').exists():
        report = read(destination / 'report.json')
        if report['status'] == 'complete':
            return report
    if destination.exists():
        # Keep evidence of an incomplete evaluation before retrying in a clean directory.
        destination.rename(destination.with_name(destination.name + '.incomplete-' + datetime.now(TZ).strftime('%H%M%S')))
    command = [sys.executable, '-u', str(ROOT / 'evaluate_blip.py'), '--experiment', str(experiment),
               '--feature-cache-dir', str(baseline), '--output', str(destination),
               '--num-beams', str(beams), '--batch-size', '4']
    if expected:
        command += ['--expected-predictions', str(expected)]
    with destination.with_suffix('.log').open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, HF_HUB_OFFLINE='1', PYTHONIOENCODING='utf-8'),
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        while process.poll() is None:
            time.sleep(10)
    if process.returncode:
        raise RuntimeError(f'Independent evaluation failed: {destination.with_suffix(".log")}')
    return read(destination / 'report.json')


def export_and_report(suite, args, baseline):
    winner = max(suite['trials'], key=lambda trial: trial['best_cider'])
    experiment = Path(winner['output'])
    suite.update(phase='independent_greedy_evaluation', selected_experiment=winner['output'])
    write(args.output / 'status.json', suite)
    expected = experiment / 'predictions' / f'epoch-{winner["best_epoch"]:02d}.json'
    greedy = run_evaluation(experiment, baseline, args.output / 'evaluation-greedy', 1, expected)
    if not greedy['training_predictions_match'] or greedy['validation_images'] != suite['baseline_contract']['validation_images']:
        raise RuntimeError('Full validation checkpoint restoration was not verified')
    if not math.isclose(greedy['metrics']['cider'], winner['best_cider'], rel_tol=1e-7):
        raise RuntimeError('Recomputed CIDEr does not match training selection')
    suite['phase'] = 'beam_search_evaluation'
    write(args.output / 'status.json', suite)
    beam = run_evaluation(experiment, baseline, args.output / 'evaluation-beam3', 3)
    selected = max([greedy, beam], key=lambda report: report['metrics']['cider'])
    predictions_path = args.output / ('evaluation-greedy' if selected['generation']['num_beams'] == 1 else 'evaluation-beam3') / 'predictions.json'
    bundle = args.output / 'best_model'
    if not bundle.exists():
        shutil.copytree(experiment / 'best', bundle)
    with (bundle / 'model.safetensors').open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != selected['model_sha256']:
            raise RuntimeError('Exported model differs from the independently evaluated checkpoint')
    write(bundle / 'inference_config.json', selected['generation'])
    write(bundle / 'experiment_provenance.json', {'training_experiment': str(experiment),
          'best_epoch': winner['best_epoch'], 'split_sha256': suite['baseline_contract']['split_sha256'],
          'evaluation': selected, 'note': 'Validation-selected model; no independent test set in this protocol.'})
    rows = read(predictions_path)
    image = DATA_IMAGE_DIR() / rows[0]['image']
    independent = subprocess.run([sys.executable, str(ROOT / 'predict_blip.py'), '--model', str(bundle), '--image', str(image)],
                                 cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=180)
    (args.output / 'export-prediction.log').write_text(independent.stdout + independent.stderr, encoding='utf-8')
    if independent.returncode or not independent.stdout.strip():
        raise RuntimeError('Exported model CLI inference failed')
    bootstrap = paired_cider_bootstrap(baseline / 'predictions/epoch-00.json', predictions_path)
    suite.update(status='complete', phase='finished', completed_at=now(), winner=winner,
                 greedy_evaluation=greedy, beam_evaluation=beam, selected_evaluation=selected,
                 exported_model=str(bundle.resolve()), export_cli='passed', paired_cider_bootstrap=bootstrap)
    contract = suite['baseline_contract']
    lines = ['# BLIP 参数优化与训练结果', '', f'完成时间：{now()}。全部计划实验均已结束，独立加载复核和导出模型 CLI 推理通过。', '',
             f'所有实验使用同一随机 80/20 图片划分，训练 {contract["train_images"]:,} 张、验证 {contract["validation_images"]:,} 张；'
             '各图片全部参考描述留在同一集合。', '',
             '| 学习率 | 物理 batch | 梯度累积 | 完成轮数 | 最佳轮次 | 验证 CIDEr | BLEU-4 | METEOR | ROUGE-L |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for trial in suite['trials']:
        metrics = trial['best_metrics']
        lines.append(f'| {trial["learning_rate"]:.7g} | {trial["batch_size"]} | {trial["gradient_accumulation"]} | '
                     f'{trial["completed_epochs"]} | {trial["best_epoch"]} | '
                     f'{metrics["cider"]:.4f} | {metrics["bleu4"]:.4f} | {metrics["meteor"]:.4f} | {metrics["rouge_l"]:.4f} |')
    benchmark = suite['throughput_benchmark']
    lines += ['', f'物理批次实测选择 {benchmark["selected"]["batch_size"]}，有效 batch size 保持 16，'
              f'按原始 {benchmark["selected"]["loss_group_size"]} 条描述分组保持损失加权。'
              f'短时 GPU 计算吞吐量相对最小测试批次为 {benchmark["measured_speedup"]:.2f} 倍；完整训练耗时另见各组日志。', '',
              f'停止搜索依据：{suite["search_stop_reason"]}。单次训练最多 {contract["epochs"]} 轮，'
              f'验证 CIDEr 连续 {contract["patience"]} 轮不改善时早停。', '',
              f'选择的学习率为 `{winner["learning_rate"]:.7g}`，最佳轮次 {winner["best_epoch"]}。'
              f'最终解码使用 {selected["generation"]["num_beams"]} beams，CIDEr 为 {selected["metrics"]["cider"]:.4f}；'
              f'同一模型贪心 CIDEr 为 {greedy["metrics"]["cider"]:.4f}，beam=3 为 {beam["metrics"]["cider"]:.4f}。', '',
              f'相对原始零样本模型的 CIDEr 提升为 {bootstrap["mean_delta"]:.4f}，图片级配对 bootstrap 95% 区间为 '
              f'[{bootstrap["ci95"][0]:.4f}, {bootstrap["ci95"][1]:.4f}]。该区间固定当前验证参考语料的 IDF，未校正调参选择偏差。', '',
              f'独立进程重新加载最佳模型后，全部 {contract["validation_images"]:,} 张验证图的贪心预测与训练记录一致；另行检查原始 JPEG 推理。'
              '视觉缓存为 FP16，原始图片直接推理可能因量化边界有少量文本差异。', '',
              f'模型目录：`{bundle}`。完整结果：`{args.output / "results.json"}`。', '',
              '```powershell',
              f'.\\.venv-blip\\Scripts\\python.exe predict_blip.py --model "{bundle}" --image "本地图片.jpg"',
              '```', '',
              '指标使用本项目记录的共同分词协议及原生数值尺度。验证集用于早停、参数与解码选择，结果属于验证集；'
              '没有独立测试集，也不表示找到了全局最优参数。额外预训练数据和可能的数据重叠限制仍适用。', '']
    args.report_path.write_text('\n'.join(lines), encoding='utf-8')
    write(args.output / 'results.json', suite)
    write(args.output / 'status.json', suite)


def DATA_IMAGE_DIR():
    from caption_data import DATA_DIR
    return DATA_DIR / 'Flicker8k_Dataset'


def paired_cider_bootstrap(original, selected):
    import numpy as np
    from pycocoevalcap.cider.cider import Cider
    from caption_metrics import tokens
    before, after = read(original), read(selected)
    if [row['image'] for row in before] != [row['image'] for row in after]:
        raise ValueError('Bootstrap predictions have different images/order')
    if [row['references'] for row in before] != [row['references'] for row in after]:
        raise ValueError('Bootstrap predictions have different reference captions')
    references = {i: [' '.join(tokens(text)) for text in row['references']] for i, row in enumerate(before)}
    def score(rows):
        _, scores = Cider().compute_score(references, {i: [' '.join(tokens(row['caption']))] for i, row in enumerate(rows)})
        return scores
    delta = score(after) - score(before)
    random = np.random.default_rng(111)
    samples = np.array([delta[random.integers(0, len(delta), len(delta))].mean() for _ in range(2000)])
    return {'mean_delta': float(delta.mean()), 'ci95': np.quantile(samples, [.025, .975]).tolist(),
            'replicates': 2000, 'unit': 'paired images; fixed validation-corpus IDF', 'seed': 111}


def run(args):
    args.baseline, args.output = args.baseline.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'results.json').exists():
        if read(args.output / 'results.json')['status'] == 'complete':
            print('Tuning already completed:', args.output, flush=True)
            return
    if (args.output / 'status.json').exists():
        previous = read(args.output / 'status.json')
        observed = process_information(previous.get('pid'))
        if observed.get('known') and observed.get('alive') and 'run_blip_tuning.py' in (observed.get('command') or ''):
            raise RuntimeError('Another tuning supervisor is already running')
        suite = previous
    else:
        suite = {'status': 'running', 'started_at': now(), 'pid': os.getpid(), 'trials': [],
                 'baseline': str(args.baseline), 'baseline_contract': read(args.baseline / 'config.json'),
                 'search_policy': {'max_trials': args.max_trials, 'min_gain': args.min_gain,
                                   'initial_learning_rates': [1e-5, 5e-6, 2e-5],
                                   'learning_rate_bounds': [1e-6, 8e-5],
                                   'stop': 'two successive candidates below minimum gain, narrow interval, or maximum trials'},
                 'recovery_events': []}
        snapshot = args.output / 'source'
        snapshot.mkdir(exist_ok=True)
        for filename in ['run_blip_tuning.py', 'blip_tuning_policy.py', 'benchmark_blip_batches.py',
                         'blip_loss.py', 'evaluate_blip.py', 'predict_blip.py']:
            shutil.copy2(ROOT / filename, snapshot / filename)
    suite.update(status='running', pid=os.getpid())
    write(args.output / 'status.json', suite)
    baseline_config = suite['baseline_contract']
    if not suite['trials']:
        trial = wait_for_training(args.baseline, baseline_config, args.baseline, suite, args)
        trial['gain_over_previous_best'] = 0.
        suite['trials'].append(trial)
        write(args.output / 'status.json', suite)
    if 'throughput_benchmark' in suite:
        physical = suite['throughput_benchmark']['selected']
    else:
        physical = throughput_profile(args.baseline, suite, args)
    while True:
        learning_rate, reason = next_learning_rate(suite['trials'], args.min_gain, args.max_trials)
        if learning_rate is None:
            suite['search_stop_reason'] = reason
            break
        key = f'{learning_rate:.8g}'.replace('.', 'p').replace('-', 'm')
        output = args.output / f'lr-{key}'
        config = dict(baseline_config, learning_rate=learning_rate)
        config.update(batch_size=physical['batch_size'], gradient_accumulation=physical['gradient_accumulation'])
        if config['batch_size'] != baseline_config['batch_size']:
            config['loss_group_size'] = physical['loss_group_size']
        suite.update(phase='starting_candidate', candidate_learning_rate=learning_rate, candidate_reason=reason)
        write(args.output / 'status.json', suite)
        launched = None if output.exists() else launch_training(output, config, args.baseline)
        suite['active_launch'] = launched
        write(args.output / 'status.json', suite)
        trial = wait_for_training(output, config, args.baseline, suite, args, launched)
        trial['gain_over_previous_best'] = trial['best_cider'] - max(row['best_cider'] for row in suite['trials'])
        trial['candidate_reason'] = reason
        suite['trials'].append(trial)
        write(args.output / 'status.json', suite)
    export_and_report(suite, args, args.baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'runs/blip-full-random80-20261022')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/blip-tuning-20261022')
    parser.add_argument('--poll-seconds', type=int, default=20)
    parser.add_argument('--max-trials', type=int, default=7)
    parser.add_argument('--min-gain', type=float, default=.005)
    parser.add_argument('--report-path', type=Path, default=ROOT / 'BLIP_OPTIMIZATION_RESULTS.md')
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.max_trials < 3 or args.min_gain <= 0:
        parser.error('Positive poll/gain values and at least three trials are required')
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        run(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        failed = read(args.output / 'status.json') if (args.output / 'status.json').exists() else {}
        if failed.get('pid') and failed['pid'] != os.getpid():
            observed = process_information(failed['pid'])
            if not observed.get('known') or observed.get('alive'):
                raise
        failed.update(status='failed', error=traceback.format_exc(), failed_at=now(), pid=os.getpid())
        write(args.output / 'status.json', failed)
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
