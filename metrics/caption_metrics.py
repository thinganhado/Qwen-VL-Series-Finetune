from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice


def meteor_score(gts, res):
    """
    METEOR:
    score, scores = Meteor().compute_score(gts, res)
    """
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    return score, scores


def rouge_l_score(gts, res):
    """
    ROUGE-L:
    score, scores = Rouge().compute_score(gts, res)
    """
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    return score, scores


def spice_score(gts, res):
    """
    SPICE:
    score, scores = Spice().compute_score(gts, res)
    """
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    return score, scores


def caption_metrics(gts, res):
    """
    Aggregate caption metrics using pycocoevalcap scorers.
    Expects caller-prepared gts/res in scorer-compatible format.
    """
    meteor, meteor_per_sample = meteor_score(gts, res)
    rouge_l, rouge_l_per_sample = rouge_l_score(gts, res)
    spice, spice_per_sample = spice_score(gts, res)

    return {
        "METEOR": meteor,
        "ROUGE_L": rouge_l,
        "SPICE": spice,
        "METEOR_per_sample": meteor_per_sample,
        "ROUGE_L_per_sample": rouge_l_per_sample,
        "SPICE_per_sample": spice_per_sample,
    }


__all__ = [
    "meteor_score",
    "rouge_l_score",
    "spice_score",
    "caption_metrics",
]

