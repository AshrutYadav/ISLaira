"""
Evaluation metrics for Sign Language Transformer.

  - BLEU-1/2/3/4  (translation quality — paper Table 1)
  - WER           (recognition quality — paper Table 1)

These are pure-Python implementations compatible with Python 3.9,
requiring only the standard library + collections.

For production use, the `sacrebleu` package is recommended (pip install sacrebleu),
and this module will automatically use it if available.
"""

import math
from collections import Counter
from typing import Dict, List


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def _clip_count(hyp_counts: Counter, ref_counts_list: List[Counter]) -> int:
    clipped = 0
    for ngram, cnt in hyp_counts.items():
        max_ref = max(rc.get(ngram, 0) for rc in ref_counts_list)
        clipped += min(cnt, max_ref)
    return clipped


def compute_bleu(
    hypotheses: List[List[str]],
    references: List[List[List[str]]],
    max_n: int = 4,
    smooth: bool = True,
) -> Dict[str, float]:
    """
    Corpus-level BLEU score.

    Args:
        hypotheses: list of tokenised hypothesis sentences
        references: list of lists of tokenised reference sentences
                    (each hypothesis can have multiple references)
        max_n:      maximum n-gram order (default 4 for BLEU-4)
        smooth:     add-1 smoothing for zero counts (Lin & Och 2004)

    Returns:
        dict with keys 'bleu1' … 'bleu{max_n}', 'bleu4'
    """
    # Try sacrebleu first (more accurate)
    try:
        import sacrebleu
        flat_refs = [[" ".join(r[0]) for r in references]]
        flat_hyps = [" ".join(h) for h in hypotheses]
        result = sacrebleu.corpus_bleu(flat_hyps, flat_refs)
        return {
            "bleu1": result.precisions[0],
            "bleu2": result.precisions[1],
            "bleu3": result.precisions[2],
            "bleu4": result.score,
        }
    except ImportError:
        pass  # fall through to manual implementation

    assert len(hypotheses) == len(references), "Mismatch between hyps and refs"

    total_hyp_len = 0
    total_ref_len = 0
    clipped_counts = [0] * max_n
    total_counts   = [0] * max_n

    for hyp, refs in zip(hypotheses, references):
        total_hyp_len += len(hyp)
        # Closest reference length (standard BLEU brevity penalty rule)
        ref_lens = [len(r) for r in refs]
        closest  = min(ref_lens, key=lambda rl: (abs(rl - len(hyp)), rl))
        total_ref_len += closest

        ref_ngram_lists = [_ngram_counts(r, n + 1) for r in refs for n in range(max_n)]

        for n in range(max_n):
            hyp_counts = _ngram_counts(hyp, n + 1)
            ref_counts = [_ngram_counts(r, n + 1) for r in refs]
            clipped_counts[n] += _clip_count(hyp_counts, ref_counts)
            total_counts[n]   += max(sum(hyp_counts.values()), 0)

    # Brevity penalty
    if total_hyp_len < total_ref_len:
        bp = math.exp(1 - total_ref_len / max(total_hyp_len, 1))
    else:
        bp = 1.0

    scores = {}
    log_avg = 0.0
    for n in range(max_n):
        c = clipped_counts[n]
        t = total_counts[n]
        if smooth:
            precision = (c + 1) / (t + 1)
        else:
            precision = c / max(t, 1)
        scores[f"bleu{n+1}"] = precision * 100
        log_avg += math.log(max(precision, 1e-10)) / max_n

    scores["bleu4"] = bp * math.exp(log_avg) * 100
    return scores


# ---------------------------------------------------------------------------
# WER (Word Error Rate — for gloss recognition, Table 1)
# ---------------------------------------------------------------------------

def _edit_distance(a: List, b: List) -> int:
    """Levenshtein edit distance between two sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_wer(
    hypotheses: List[List[str]],
    references: List[List[str]],
) -> float:
    """
    Corpus-level Word Error Rate (WER).

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference length.

    Returns percentage (0–100).
    """
    total_dist = 0
    total_ref  = 0
    for hyp, ref in zip(hypotheses, references):
        total_dist += _edit_distance(hyp, ref)
        total_ref  += len(ref)
    return (total_dist / max(total_ref, 1)) * 100
