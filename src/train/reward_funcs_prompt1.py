import re
import math
from typing import List, Optional, Sequence, Set

W_NDCG = 1.0
W_FORMAT = 0.1


_STRICT_LIST_PATTERN = re.compile(r"^\[(\d+), (\d+), (\d+)\]$")


def _parse_strict_region_ids(text: str) -> Optional[List[int]]:
    """
    Parse completion in strict format: [1, 2, 3]
    Returns None if format/range/uniqueness constraints are violated.
    """
    match = _STRICT_LIST_PATTERN.fullmatch(text.strip())
    if not match:
        return None

    ids = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    if any(rid < 1 or rid > 16 for rid in ids):
        return None
    if len(set(ids)) != 3:
        return None
    return ids


def _parse_gt_ids(text: str) -> List[int]:
    """
    Parse GT ids from either JSON-like list string or comma-separated values.
    Non-integer tokens are ignored.
    """
    tokens = re.findall(r"\d+", text)
    return [int(tok) for tok in tokens]

def _ndcg_at_3(pred: Sequence[int], gt: Sequence[int]) -> float:
    """
    nDCG@3 for binary relevance with formula aligned to metrics/localization_metrics.py.
    """
    G: Set[int] = set(gt)
    dcg = 0.0
    for j in range(1, 4):
        rel = 1 if pred[j - 1] in G else 0
        dcg += (2**rel - 1) / math.log2(j + 1)

    m = min(3, len(G))
    if m == 0:
        return 0.0

    idcg = 0.0
    for j in range(1, m + 1):
        idcg += 1.0 / math.log2(j + 1)

    return dcg / idcg


def _strict_format_score(completions: Sequence[str], **kwargs) -> List[float]:
    """
    Reward 1.0 only when completion is exactly in the form:
      [x, y, z]
    where x,y,z are unique integers in [1, 16].
    """
    rewards = []
    for completion in completions:
        pred_ids = _parse_strict_region_ids(completion)
        rewards.append(1.0 if pred_ids is not None else 0.0)
    return rewards


def _ndcg3_score(completions: Sequence[str], assistant: Sequence[str], **kwargs) -> List[float]:
    """
    nDCG@3 using strict prediction parsing.
    Invalid predictions receive 0.0.
    """
    rewards = []
    for completion, gt in zip(completions, assistant):
        pred_ids = _parse_strict_region_ids(completion)
        if pred_ids is None:
            rewards.append(0.0)
            continue

        gt_ids = _parse_gt_ids(gt)
        if not gt_ids:
            rewards.append(0.0)
            continue

        rewards.append(float(_ndcg_at_3(pred_ids, gt_ids)))
    return rewards


def prompt1_reward(completions: Sequence[str], assistant: Sequence[str], **kwargs) -> List[float]:
    """
    Weighted Prompt-1 reward:
      w_ndcg * nDCG@3 + w_format * strict_format
    """
    ndcg_scores = _ndcg3_score(completions, assistant, **kwargs)
    format_scores = _strict_format_score(completions, **kwargs)
    return [
        float(W_NDCG * n + W_FORMAT * f)
        for n, f in zip(ndcg_scores, format_scores)
    ]
