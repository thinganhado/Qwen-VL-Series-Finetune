from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from statistics import mean


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

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        try:
            hypothesis_str = hypothesis_str.replace("|||", "").replace("  ", " ")
            score_line = " ||| ".join(("SCORE", " ||| ".join(reference_list), hypothesis_str))
            self.meteor_p.stdin.write("{}\n".format(score_line).encode())
            self.meteor_p.stdin.flush()
            stats = self.meteor_p.stdout.readline().decode("utf-8", errors="ignore").strip()
            eval_line = "EVAL ||| {}".format(stats)
            self.meteor_p.stdin.write("{}\n".format(eval_line).encode())
            self.meteor_p.stdin.flush()
            # Some jars emit two scalar lines here (avg + all); use the last valid one.
            first = self._read_scalar_score()
            try:
                second = self._read_scalar_score(max_reads=1)
                return second
            except Exception:
                return first
        finally:
            self.lock.release()


def meteor_score(gts, res):
    """
    METEOR:
    score, scores = Meteor().compute_score(gts, res)
    """
    # Try batch pycocoevalcap first.
    try:
        scorer = RobustMeteor()
        return scorer.compute_score(gts, res)
    except Exception:
        pass

    # Fallback 1: per-sample with a fresh METEOR process each call.
    try:
        img_ids = list(gts.keys())
        per_sample = []
        for i in img_ids:
            assert len(res[i]) == 1
            scorer_i = RobustMeteor()
            try:
                per_sample.append(float(scorer_i._score(res[i][0], gts[i])))
            finally:
                try:
                    scorer_i.__del__()
                except Exception:
                    pass
        agg = mean(per_sample) if per_sample else 0.0
        return float(agg), per_sample
    except Exception:
        pass

    # Fallback 2: NLTK METEOR without Java subprocess.
    from nltk.translate.meteor_score import meteor_score as nltk_meteor_score

    img_ids = list(gts.keys())
    per_sample = []
    for i in img_ids:
        hyp = (res[i][0] or "").replace("|||", " ").strip().split()
        refs = [(r or "").replace("|||", " ").strip().split() for r in gts[i]]
        per_sample.append(float(nltk_meteor_score(refs, hyp)))
    agg = mean(per_sample) if per_sample else 0.0
    return float(agg), per_sample


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
