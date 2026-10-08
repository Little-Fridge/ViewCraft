"""Example custom reward: use scorer: score.py:score in task.yaml."""


def score(prediction, answer, item):
    """Return a finite float in [0, 1]. `item` is the public question/media."""
    return float(str(prediction).strip() == str(answer).strip())
