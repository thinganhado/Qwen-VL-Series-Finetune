import re
import math
from typing import Dict, List, Optional, Sequence

W_NDCG = 1.0
W_FORMAT = 0.1


def _parse_region_ids_any(text: str) -> Optional[List[int]]:
    """
    Parse completion with flexible format as long as it contains exactly 3 integers.
    Examples accepted:
      [1, 2, 3]
      [1,2,3]
      1,2,3
      C1 C2 C3
      regions: 1 2 3
    Returns None if count/range/uniqueness constraints are violated.
    """
    ids = [int(tok) for tok in re.findall(r"\d+", text)]
    if len(ids) != 3:
        return None

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
    Graded nDCG@3 with GT rank-aware relevance.

    GT is already ranked by prominence (most prominent first):
      GT rank 1 -> rel 3
      GT rank 2 -> rel 2
      GT rank 3 -> rel 1
      non-GT    -> rel 0

    DCG@3 = sum_{j=1..3} (2^rel_j - 1) / log2(j + 1)
    IDCG@3 is computed from ideal GT ordering (top-3 GT ranks in order).
    """
    rel_by_id: Dict[int, int] = {}
    # Keep first occurrence only; GT order encodes relevance.
    for rank, rid in enumerate(gt[:3]):
        if rid in rel_by_id:
            continue
        rel_by_id[rid] = 3 - rank

    dcg = 0.0
    for j in range(1, 4):
        rel = rel_by_id.get(pred[j - 1], 0)
        dcg += (2**rel - 1) / math.log2(j + 1)

    ideal_rels = sorted(rel_by_id.values(), reverse=True)
    m = min(3, len(ideal_rels))
    if m == 0:
        return 0.0

    idcg = 0.0
    for j in range(1, m + 1):
        rel = ideal_rels[j - 1]
        idcg += (2**rel - 1) / math.log2(j + 1)

    return dcg / idcg


def _strict_format_score(completions: Sequence[str], **kwargs) -> List[float]:
    """
    Reward 1.0 when completion can be parsed into exactly 3 unique region IDs
    in [1, 16], regardless of delimiter/wrapping format.
    """
    rewards = []
    for completion in completions:
        pred_ids = _parse_region_ids_any(completion)
        rewards.append(1.0 if pred_ids is not None else 0.0)
    return rewards


def _ndcg3_score(completions: Sequence[str], assistant: Sequence[str], **kwargs) -> List[float]:
    """
    nDCG@3 using flexible 3-number prediction parsing.
    Invalid predictions receive 0.0.
    """
    rewards = []
    for completion, gt in zip(completions, assistant):
        pred_ids = _parse_region_ids_any(completion)
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
