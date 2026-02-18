import math
from typing import List, Sequence, Set, Optional

# ------------------------------------------------------------
# Ranking-sensitive metrics for localization tasks 
# ------------------------------------------------------------

def recall_at_k(pred: Sequence[int], gt: Sequence[int], k: int) -> float:
    """
    Recall@k = |TopK(R) ∩ G| / k
    """
    topk = pred[:k]
    G: Set[int] = set(gt)
    return len([r for r in topk if r in G]) / k


def dcg_at_k(pred: Sequence[int], gt: Sequence[int], k: int) -> float:
    """
    DCG@k = Σ_{j=1..k} (2^{rel(j)} - 1) / log2(j + 1)
    where rel(j) = 1 if item at rank j is in G, else 0.
    """
    G: Set[int] = set(gt)
    dcg = 0.0
    for j in range(1, k + 1):
        rel = 1 if pred[j - 1] in G else 0
        dcg += (2**rel - 1) / math.log2(j + 1)
    return dcg


def ndcg_at_k(pred: Sequence[int], gt: Sequence[int], k: int) -> float:
    """
    nDCG@k = DCG@k / IDCG@k
    IDCG@k is DCG@k for an ideal ranking (all relevant items first).
    For binary relevance with |G| relevant items:
      IDCG@k = Σ_{j=1..min(k, |G|)} (2^1 - 1) / log2(j + 1)
             = Σ_{j=1..min(k, |G|)} 1 / log2(j + 1)
    """
    dcg = dcg_at_k(pred, gt, k)
    m = min(k, len(set(gt)))
    if m == 0:
        return 0.0
    idcg = 0.0
    for j in range(1, m + 1):
        idcg += 1.0 / math.log2(j + 1)
    return dcg / idcg


def average_precision(pred: Sequence[int], gt: Sequence[int]) -> float:
    """
    AP = (1 / k) * Σ_{j=1..N} Precision@j * 1[R_j ∈ G]
    where:
      - N = len(pred)
      - k = |G| (number of relevant GT items)
      - Precision@j = |Top-j(R) ∩ G| / j
    """
    G: Set[int] = set(gt)
    k = len(G)
    if k == 0:
        return 0.0

    hits = 0
    s = 0.0
    for j, rj in enumerate(pred, start=1):
        if rj in G:
            hits += 1
            precision_j = hits / j
            s += precision_j
    return s / k


def mean_average_precision(preds: List[Sequence[int]], gts: List[Sequence[int]]) -> float:
    """
    MAP = mean(AP) across samples.
    """
    return sum(average_precision(p, g) for p, g in zip(preds, gts)) / len(preds)
