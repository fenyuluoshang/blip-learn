"""A bounded adaptive learning-rate search with an explicit plateau rule."""
import math


def next_learning_rate(trials, min_gain=.005, max_trials=7):
    """Try both neighbors first, then refine around the strongest validation run."""
    rates = [trial['learning_rate'] for trial in trials]
    def attempted(rate):
        return any(math.isclose(rate, previous, rel_tol=1e-6) for previous in rates)
    for rate in [5e-6, 2e-5]:
        if not attempted(rate):
            return rate, 'initial lower/upper learning-rate comparison'
    if len(trials) >= max_trials:
        return None, 'predeclared maximum number of complete training experiments reached'
    if len(trials) >= 3 and all(trial['gain_over_previous_best'] < min_gain for trial in trials[-2:]):
        return None, 'two consecutive candidates improved validation CIDEr by less than the declared threshold'
    ordered = sorted(trials, key=lambda trial: trial['learning_rate'])
    best = max(trials, key=lambda trial: trial['best_cider'])
    index = ordered.index(best)
    if index == 0 and best['learning_rate'] / 2 >= 1e-6:
        return best['learning_rate'] / 2, 'expand toward the better lower learning rate'
    if index == len(ordered) - 1 and best['learning_rate'] * 2 <= 8e-5:
        return best['learning_rate'] * 2, 'expand toward the better higher learning rate'
    neighbors = [ordered[i] for i in [index - 1, index + 1] if 0 <= i < len(ordered)]
    neighbors.sort(key=lambda trial: trial['best_cider'], reverse=True)
    for neighbor in neighbors:
        ratio = max(neighbor['learning_rate'], best['learning_rate']) / min(neighbor['learning_rate'], best['learning_rate'])
        if ratio > 1.2:
            rate = math.sqrt(neighbor['learning_rate'] * best['learning_rate'])
            if not attempted(rate):
                return rate, 'geometric refinement between the best run and a neighboring learning rate'
    return None, 'neighboring learning-rate intervals are sufficiently narrow'
