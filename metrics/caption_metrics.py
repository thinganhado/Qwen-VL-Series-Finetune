from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice


class RobustMeteor(Meteor):
    """
    METEOR wrapper that tolerates occasional non-scalar stdout lines.
    """

    @staticmethod
    def _parse_scalar_line(raw_line):
        line = raw_line.decode("utf-8", errors="ignore").strip() if isinstance(raw_line, bytes) else str(raw_line).strip()
        try:
            return float(line)
        except ValueError:
            return None

    def _read_scalar_score(self, max_reads=64):
        for _ in range(max_reads):
            raw = self.meteor_p.stdout.readline()
            if not raw:
                break
            val = self._parse_scalar_line(raw)
            if val is not None:
                return val
        raise RuntimeError("METEOR produced no scalar score line")

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()
        img_ids = list(gts.keys())
        scores = []

        eval_line = "EVAL"
        self.lock.acquire()
        try:
            for i in img_ids:
                assert len(res[i]) == 1
                stat = self._stat(res[i][0], gts[i])
                eval_line += " ||| {}".format(stat)

            self.meteor_p.stdin.write("{}\n".format(eval_line).encode())
            self.meteor_p.stdin.flush()

            for _ in img_ids:
                scores.append(self._read_scalar_score())
            score = self._read_scalar_score()
        finally:
            self.lock.release()

        return score, scores


def meteor_score(gts, res):
    """
    METEOR:
    score, scores = Meteor().compute_score(gts, res)
    """
    scorer = RobustMeteor()
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
