import re
from typing import Dict, List, Optional, Sequence

W_ACC = 0.75
W_CONS = 0.25


# Allowed closed sets
TIME_LABELS = {"speech", "non-speech"}
FREQ_LABELS = {"low", "mid", "high"}
PHON_LABELS = {"consonant", "vowel", "unvoiced"}


# Regex + lexicon extraction rules for En
TIME_LEXICON = {
    "speech": [
        r"\bspeech\b",
        r"\bvoiced\b",
        r"\bspoken\b",
    ],
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
    "vowel": [
        r"\bvowel\b",
        r"\bformant\b",
    ],
    "consonant": [
        r"\bconsonant\b",
        r"\bstop\b",
        r"\bfricative\b",
    ],
    "unvoiced": [
        r"\bunvoiced\b",
        r"\bvoiceless\b",
        r"\baspiration\b",
        r"\bburst\b",
    ],
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


def _extract_field_value(text: str, field: str) -> Optional[str]:
    """
    Extract single-line field value:
      T: ...
      F: ...
      P: ...
      En: ...
    """
    pattern = re.compile(
        rf"(?:^|\n)\s*{field}\s*:\s*(.*?)(?=\n[A-Za-z]{{1,2}}\s*:|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return None
    return m.group(1).strip()


def _parse_tfp_from_text(text: str) -> Dict[str, Optional[str]]:
    raw_t = _extract_field_value(text, "T")
    raw_f = _extract_field_value(text, "F")
    raw_p = _extract_field_value(text, "P")
    return {
        "T": _normalize_time(raw_t) if raw_t is not None else None,
        "F": _normalize_freq(raw_f) if raw_f is not None else None,
        "P": _normalize_phon(raw_p) if raw_p is not None else None,
    }


def _extract_en_text(text: str) -> str:
    en = _extract_field_value(text, "En")
    if en is None:
        # Fallback: use whole text if En field is absent.
        return text
    return en


def _extract_label_by_lexicon(en_text: str, lexicon: Dict[str, List[str]]) -> Optional[str]:
    """
    Weighted counts by mention frequency; tie-break by first mention position.
    """
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


def _field_accuracy_score(pred: Dict[str, Optional[str]], gt: Dict[str, Optional[str]]) -> float:
    """
    Macro average over T/F/P:
      (1[T=gtT] + 1[F=gtF] + 1[P=gtP]) / 3
    Missing/invalid labels score 0 for that field.
    """
    score = 0.0
    score += 1.0 if pred["T"] is not None and pred["T"] == gt["T"] else 0.0
    score += 1.0 if pred["F"] is not None and pred["F"] == gt["F"] else 0.0
    score += 1.0 if pred["P"] is not None and pred["P"] == gt["P"] else 0.0
    return score / 3.0


def _consistency_score(pred: Dict[str, Optional[str]], en_extracted: Dict[str, Optional[str]]) -> float:
    """
    Fields <-> En consistency:
      (1[T_en=T] + 1[F_en=F] + 1[P_en=P]) / 3
    Missing En extraction on a field gives 0 for that field.
    """
    score = 0.0
    score += 1.0 if en_extracted["T"] is not None and en_extracted["T"] == pred["T"] else 0.0
    score += 1.0 if en_extracted["F"] is not None and en_extracted["F"] == pred["F"] else 0.0
    score += 1.0 if en_extracted["P"] is not None and en_extracted["P"] == pred["P"] else 0.0
    return score / 3.0


def prompt2_reward(completions: Sequence[str], assistant: Sequence[str], **kwargs) -> List[float]:
    """
    Dominant metric: field accuracy on T/F/P.
    Secondary metric: fields <-> En consistency.

    Combined reward:
      0.75 * field_accuracy + 0.25 * consistency
    """
    rewards = []
    for completion, gt_text in zip(completions, assistant):
        pred_tfp = _parse_tfp_from_text(completion)
        gt_tfp = _parse_tfp_from_text(gt_text)

        field_acc = _field_accuracy_score(pred_tfp, gt_tfp)

        en_text = _extract_en_text(completion)
        en_tfp = _independent_extract_from_en(en_text)
        consistency = _consistency_score(pred_tfp, en_tfp)

        reward = W_ACC * field_acc + W_CONS * consistency
        rewards.append(float(reward))

    return rewards
