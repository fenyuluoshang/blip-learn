"""Pinned-library caption metrics with explicit, shared text normalization."""
import re
from pathlib import Path


def tokens(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def compute_metrics(predictions):
    import nltk
    nltk.data.path.insert(0, str(Path(__file__).resolve().parent / 'models/nltk_data'))
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from pycocoevalcap.cider.cider import Cider
    from rouge_score.rouge_scorer import RougeScorer

    if not predictions:
        raise ValueError('No generated captions to score')
    refs = [[tokens(text) for text in row['references']] for row in predictions]
    hyps = [tokens(row['caption']) for row in predictions]
    scorer = RougeScorer(['rougeL'], use_stemmer=False)
    rouge = sum(max(scorer.score(' '.join(ref), ' '.join(hyp))['rougeL'].fmeasure
                    for ref in group) for group, hyp in zip(refs, hyps)) / len(hyps)
    meteor = sum(meteor_score(group, hyp) if hyp else 0. for group, hyp in zip(refs, hyps)) / len(hyps)
    cider, _ = Cider().compute_score(
        {i: [' '.join(ref) for ref in group] for i, group in enumerate(refs)},
        {i: [' '.join(hyp)] for i, hyp in enumerate(hyps)})
    return {'bleu1': float(corpus_bleu(refs, hyps, weights=(1., 0., 0., 0.))),
            'bleu4': float(corpus_bleu(refs, hyps, weights=(.25,) * 4)),
            'meteor': float(meteor), 'rouge_l': float(rouge), 'cider': float(cider)}


PROTOCOL = {
    'tokenization': 'lowercase ASCII alphanumeric runs: re.findall([a-z0-9]+); shared across metrics',
    'bleu': 'NLTK corpus BLEU, no smoothing, BLEU1 and BLEU4',
    'meteor': 'NLTK METEOR, default Porter stemming and WordNet matching, best reference per image',
    'rouge_l': 'rouge-score rougeL F1, no stemming, best reference per image, macro mean',
    'cider': 'pycocoevalcap Cider(n=4, sigma=6), validation-corpus document frequencies',
    'scales': 'BLEU/METEOR/ROUGE-L in 0..1; CIDEr native library scale (includes factor 10)',
    'comparability': 'No Java PTBTokenizer; these numbers require the same normalization for comparisons.',
}
