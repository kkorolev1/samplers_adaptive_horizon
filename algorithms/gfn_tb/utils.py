import math


def get_invtemp(it: int, n_epochs: int, invtemp: float, invtemp_anneal: bool) -> float:
    if not invtemp_anneal:
        return invtemp
    return linear_annealing(
        it, int(0.5 * n_epochs), invtemp, 1.0, descending=False, avoid_zero=True
    )


def linear_annealing(
    current: int,
    n_rounds: int,
    min_val: float,
    max_val: float,
    exp=True,
    descending=False,
    avoid_zero=True,
) -> float:
    assert min_val <= max_val
    if min_val == max_val:
        return min_val

    start_val, end_val = min_val, max_val
    if descending:
        start_val, end_val = end_val, start_val

    if current >= n_rounds:
        return end_val

    num = current + 1 if avoid_zero else current
    denom = n_rounds + 1 if avoid_zero else n_rounds
    if exp:
        return start_val * ((end_val / start_val) ** (num / denom))
    else:
        return start_val + (end_val - start_val) * (num / denom)
