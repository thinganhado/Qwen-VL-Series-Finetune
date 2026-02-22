import re
from typing import Dict, List, Optional, Sequence

# No field-accuracy reward. Keep other weights unchanged.
W_ACC = 0.0
W_CONS = 0.0
W_FMT = 0.1


TIME_LABELS = {"speech", "non-speech"}
FREQ_LABELS = {"low", "mid", "high"}
PHON_LABELS = {"consonant", "vowel", "unvoiced"}

TIME_LEXICON = {
    "speech": [r"\bspeech\b", r"\bvoiced\b", r"\bspoken\b"],
    "non-speech": [
        r"\bsilence\b",
        r"\bpause\b",
        r"\bnon[- ]?speech\b",
        r"\bbackground\b",
        r"\bnoise[- ]?only\b",
        r"\bunvoiced\b",
    ],
}

FREQ_LEXICON = {
    "low": [r"\blow\b"],
    "mid": [r"\bmid\b", r"\bmiddle\b"],
    "high": [r"\bhigh\b"],
}

PHON_LEXICON = {
    "vowel": [r"\bvowel\b", r"\bformant\b"],
    "consonant": [r"\bconsonant\b", r"\bstop\b", r"\bfricative\b"],
    "unvoiced": [r"\bunvoiced\b", r"\bvoiceless\b", r"\baspiration\b", r"\bburst\b"],
}


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_time(value: str) -> Optional[str]:
    v = _normalize_space(value)
    if v in TIME_LABELS:
        return v
    if v in {"nonspeech", "non speech"}:
        return "non-speech"
    return None


def _normalize_freq(value: str) -> Optional[str]:
    v = _normalize_space(value)
    if v in FREQ_LABELS:
        return v
    if v == "middle":
        return "mid"
    return None


def _normalize_phon(value: str) -> Optional[str]:
    v = _normalize_space(value)
    if v in PHON_LABELS:
        return v
    if v == "voiceless":
        return "unvoiced"
    return None


_REGION_PATTERN = re.compile(
    r"\(\s*(\d+)\s*,\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*(.*?)\s*\)",
    flags=re.DOTALL,
)


def _parse_region_tuples(text: str) -> Optional[List[Dict[str, str]]]:
    matches = _REGION_PATTERN.findall(text)
    if len(matches) != 3:
        return None

    parsed: List[Dict[str, str]] = []
    seen_cn = set()
    for cn_raw, t_raw, f_raw, p_raw, en_raw in matches:
        cn = int(cn_raw)
        t = _normalize_time(t_raw)
        f = _normalize_freq(f_raw)
        p = _normalize_phon(p_raw)
        en = en_raw.strip()

        if cn <= 0 or cn in seen_cn or t is None or f is None or p is None or len(en) == 0:
            return None
        seen_cn.add(cn)

        parsed.append({"Cn": str(cn), "T": t, "F": f, "P": p, "En": en})
    return parsed


def _extract_label_by_lexicon(en_text: str, lexicon: Dict[str, List[str]]) -> Optional[str]:
    text = en_text.lower()
    best_label = None
    best_count = 0
    best_pos = 10**9

    for label, patterns in lexicon.items():
        count = 0
        first_pos = 10**9
        for pat in patterns:
            matches = list(re.finditer(pat, text, flags=re.IGNORECASE))
            if not matches:
                continue
            count += len(matches)
            first_pos = min(first_pos, matches[0].start())
        if count > best_count or (count == best_count and count > 0 and first_pos < best_pos):
            best_label = label
            best_count = count
            best_pos = first_pos
    return best_label


def _independent_extract_from_en(en_text: str) -> Dict[str, Optional[str]]:
    return {
        "T": _extract_label_by_lexicon(en_text, TIME_LEXICON),
        "F": _extract_label_by_lexicon(en_text, FREQ_LEXICON),
        "P": _extract_label_by_lexicon(en_text, PHON_LEXICON),
    }


def _region_field_accuracy(pred: Dict[str, str], gt: Optional[Dict[str, str]]) -> float:
    if gt is None:
        return 0.0
    score = 0.0
    score += 1.0 if pred["T"] == gt["T"] else 0.0
    score += 1.0 if pred["F"] == gt["F"] else 0.0
    score += 1.0 if pred["P"] == gt["P"] else 0.0
    return score / 3.0


def _region_consistency(pred: Dict[str, str]) -> float:
    en_extracted = _independent_extract_from_en(pred["En"])
    score = 0.0
    score += 1.0 if en_extracted["T"] is not None and en_extracted["T"] == pred["T"] else 0.0
    score += 1.0 if en_extracted["F"] is not None and en_extracted["F"] == pred["F"] else 0.0
    score += 1.0 if en_extracted["P"] is not None and en_extracted["P"] == pred["P"] else 0.0
    return score / 3.0


def _format_score(text: str) -> float:
    return 1.0 if _parse_region_tuples(text) is not None else 0.0


def prompt2_no_field_accuracy_reward(completions: Sequence[str], assistant: Sequence[str], **kwargs) -> List[float]:
    """
    Prompt-2 reward with field-accuracy disabled (W_ACC=0.0).
    Combined reward (per-region, then average over 3 regions):
      0.0 * region_field_accuracy + 0.0 * region_consistency + 0.1 * region_format
    """
    rewards = []
    for completion, gt_text in zip(completions, assistant):
        pred_regions = _parse_region_tuples(completion)
        gt_regions = _parse_region_tuples(gt_text)

        if pred_regions is None or gt_regions is None:
            rewards.append(0.0)
            continue

        gt_by_cn = {r["Cn"]: r for r in gt_regions}
        fmt = _format_score(completion)
        reward = 0.0
        for pred in pred_regions:
            gt = gt_by_cn.get(pred["Cn"])
            region_acc = _region_field_accuracy(pred, gt)
            region_cons = _region_consistency(pred)
            reward += W_ACC * region_acc + W_CONS * region_cons + W_FMT * fmt

        rewards.append(float(reward / 3.0))
    return rewards
