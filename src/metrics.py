from typing import List


def ap_k(pred: List[int], gt: List[int], k: int):
    R = len(gt)
    normalizer = min(R, k)
    hits = 0
    cum_score = 0

    for i, img_id in enumerate(pred[:k], start=1):
        if img_id in gt:
            hits += 1
            prec_at_i = hits / i
            cum_score += prec_at_i

    return cum_score / normalizer


def map_k(all_pred: List[List[int]], all_gt: List[List[int]], k: int):
    # assert len(all_pred) == len(all_gt)

    ap_scores = [ap_k(pred, gt, k) for pred, gt in zip(all_pred, all_gt)]

    return sum(ap_scores) / len(ap_scores)


def evaluate(
    all_pred: List[List[int]],
    all_gt: List[List[int]],
    query_img_ids: List[int],
    ks: List[int] = [5, 10, 25, 50, 100],
):
    if query_img_ids:
        # sanitize: remove self id from retrieved list
        all_pred = [
            [valid for valid in pred if valid != query_id]
            for pred, query_id in zip(all_pred, query_img_ids)
        ]

    results = {}
    for k in ks:
        results[f"mAP@{k}"] = map_k(all_pred, all_gt, k)

    return results
