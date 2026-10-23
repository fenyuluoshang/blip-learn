"""Keep the original small-batch token weighting when batching more examples."""
import math

import torch.nn.functional as F


def grouped_caption_loss(logits, labels, group_size, label_smoothing=0.):
    targets = labels[:, 1:]
    scores = logits[:, :-1, :]
    losses = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), targets.reshape(-1),
                             reduction='none', ignore_index=-100, label_smoothing=label_smoothing)
    per_example = losses.reshape_as(targets).sum(dim=1)
    token_counts = (targets != -100).sum(dim=1)
    padding = (-len(labels)) % group_size
    if padding:
        per_example = F.pad(per_example, (0, padding))
        token_counts = F.pad(token_counts, (0, padding))
    numerator = per_example.reshape(-1, group_size).sum(dim=1)
    denominator = token_counts.reshape(-1, group_size).sum(dim=1)
    if (denominator == 0).any().item():
        raise ValueError('A loss group has no target tokens')
    return (numerator / denominator).mean(), losses.sum()


def accumulation_weight(examples, batch_index, batch_size, accumulation, group_size):
    start_batch = (batch_index // accumulation) * accumulation
    window_examples = min(batch_size * accumulation, examples - start_batch * batch_size)
    current_examples = min(batch_size, examples - batch_index * batch_size)
    return math.ceil(current_examples / group_size) / math.ceil(window_examples / group_size)
