import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from blip_tuning_policy import next_learning_rate
from run_blip_tuning import audit_completed_run, matching_training_process, training_command


class TuningTests(unittest.TestCase):
    def test_search_compares_both_sides_and_stops_on_plateau(self):
        trials = [{'learning_rate': 1e-5, 'best_cider': 1., 'gain_over_previous_best': 0.}]
        self.assertEqual(next_learning_rate(trials)[0], 5e-6)
        trials.append({'learning_rate': 5e-6, 'best_cider': .99, 'gain_over_previous_best': -.01})
        self.assertEqual(next_learning_rate(trials)[0], 2e-5)
        trials.append({'learning_rate': 2e-5, 'best_cider': 1.001, 'gain_over_previous_best': .001})
        self.assertIsNone(next_learning_rate(trials)[0])

    def test_search_follows_an_improved_boundary(self):
        trials = [{'learning_rate': 1e-5, 'best_cider': 1., 'gain_over_previous_best': 0.},
                  {'learning_rate': 5e-6, 'best_cider': 1.02, 'gain_over_previous_best': .02},
                  {'learning_rate': 2e-5, 'best_cider': 1.04, 'gain_over_previous_best': .02}]
        self.assertEqual(next_learning_rate(trials)[0], 4e-5)

    def test_unknown_process_observation_is_not_a_missing_process(self):
        record = {'known': False, 'error': 'observation timed out'}
        self.assertEqual(matching_training_process(record, Path('runs/example')), record)
        reused = {'known': True, 'alive': True, 'command': 'python unrelated.py'}
        self.assertFalse(matching_training_process(reused, Path('runs/example'))['alive'])

    def test_candidate_commands_preserve_data_and_effective_batch(self):
        baseline = Path('runs/blip-full-random80-20261022').resolve()
        config = json.loads((baseline / 'config.json').read_text(encoding='utf-8'))
        config['learning_rate'] = 5e-6
        command = training_command(Path('runs/candidate').resolve(), config, baseline)
        self.assertEqual(command[command.index('--seed') + 1], '111')
        self.assertEqual(command[command.index('--validation-fraction') + 1], '0.2')
        self.assertEqual(command[command.index('--batch-size') + 1], '2')
        self.assertEqual(command[command.index('--gradient-accumulation') + 1], '8')
        self.assertEqual(command[command.index('--feature-cache-dir') + 1], str(baseline))
        self.assertNotIn('--limit', command)

    def test_run_audit_rejects_inconsistent_best_checkpoint(self):
        source = Path('runs/blip-training-preflight-20261022')
        expected = json.loads((source / 'config.json').read_text(encoding='utf-8'))
        result = audit_completed_run(source, expected)
        self.assertEqual(result['audit'], 'passed')
        with TemporaryDirectory() as folder:
            destination = Path(folder)
            (destination / 'best').mkdir()
            for filename in ['config.json', 'results.json']:
                (destination / filename).write_bytes((source / filename).read_bytes())
            selection = json.loads((source / 'best/selection.json').read_text(encoding='utf-8'))
            selection['epoch'] += 1
            (destination / 'best/selection.json').write_text(json.dumps(selection), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Best checkpoint selection'):
                audit_completed_run(destination, expected)


if __name__ == '__main__':
    unittest.main()
