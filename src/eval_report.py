"""Reporting for `src/evaluate.py`.

Turns a set of predictions + ground truth into: the overall metrics, the optional
`--breakdown` slices (by category / country / continent), the optional
`--marginal` single-attribute scores, a console printout, and the results JSON.
"""

import json

from src.data.countries import display_country
from src.metrics import DEFAULT_KS, evaluate


def effective_k1(args) -> int:
    """Two-step k1 clamped to at least topk (else recall@max(k) is capped by k1)."""
    return max(args.two_step_k1, max(DEFAULT_KS) + 1)


def print_breakdown(label, scores: dict) -> None:
    print(f"\n  [{label}]")
    for group_key, group_scores in sorted(scores.items()):
        line = "  " + str(group_key) + ":"
        for metric, val in group_scores.items():
            line += f"  {metric}={val:.4f}"
        print(line)


def marginal_queries(records, axis, template, min_relevant):
    """One query per distinct `axis` value; relevance = every index row with that value.

    axis: "category" or "country". Measures the model on the single-attribute task
    in isolation, to see whether the joint (category AND country) score is
    bottlenecked by the category axis, the country axis, or their conjunction.
    """
    groups: dict = {}
    for i, rec in enumerate(records):
        key = rec.get(axis)
        if key is None or key == "":
            continue
        groups.setdefault(key, []).append(i)

    disp = display_country if axis == "country" else (lambda x: x)
    out = []
    for key, idxs in sorted(groups.items()):
        if len(idxs) < min_relevant:
            continue
        out.append(
            {axis: key, "text": template.format(disp(key)), "relevant_indices": idxs}
        )
    return out


def _subset_scores(all_pred, all_gt, queries, key_of, keys) -> dict:
    """Per-key metrics over the subset of queries whose `key_of(q)` matches."""
    return {
        k: evaluate(
            [all_pred[i] for i, q in enumerate(queries) if key_of(q) == k],
            [all_gt[i] for i, q in enumerate(queries) if key_of(q) == k],
            ks=DEFAULT_KS,
        )
        for k in keys
    }


def score_query_set(queries, search_fn, axis, label) -> dict:
    """Retrieve + score a marginal query set; print overall + per-key breakdown."""
    if not queries:
        print(f"\n=== Marginal: {label} — no groups meet --min-relevant ===")
        return {}
    pred = search_fn(queries)
    gt = [q["relevant_indices"] for q in queries]

    overall = evaluate(pred, gt, ks=DEFAULT_KS)
    per_key = {
        q[axis]: evaluate([pred[i]], [gt[i]], ks=DEFAULT_KS)
        for i, q in enumerate(queries)
    }
    print(f"\n=== Marginal: {label} ({len(queries)} queries) ===")
    for metric, val in overall.items():
        print(f"  {metric}: {val:.4f}")
    print_breakdown(f"marginal {label}", per_key)
    return {"overall": overall, "per_key": per_key}


def eval_and_report(
    all_pred, all_gt, queries, args, search_fn=None, records=None
) -> dict:
    results = evaluate(all_pred, all_gt, ks=DEFAULT_KS)

    mode = f" · two-step k1={effective_k1(args)}" if args.two_step else ""
    print(f"\n=== Overall ({args.model}{mode}) ===")
    for metric, val in results.items():
        print(f"  {metric}: {val:.4f}")
    results["retrieval"] = {
        "mode": "two_step" if args.two_step else "one_step",
        "k1": effective_k1(args) if args.two_step else None,
    }

    if args.breakdown:
        for name, key_of in (
            ("by_category", lambda q: q["category"]),
            ("by_country", lambda q: q["country"]),
            ("by_continent", lambda q: q.get("region")),
        ):
            keys = sorted({key_of(q) for q in queries if key_of(q)})
            if not keys:
                continue
            results[name] = _subset_scores(all_pred, all_gt, queries, key_of, keys)
            print_breakdown(name.replace("_", " "), results[name])

    if args.marginal and records is not None and search_fn is not None:
        cat_q = marginal_queries(
            records, "category", args.marginal_cat_template, args.min_relevant
        )
        ctr_q = marginal_queries(
            records, "country", args.marginal_country_template, args.min_relevant
        )
        results["marginal_category"] = score_query_set(
            cat_q, search_fn, "category", "category-only"
        )
        results["marginal_country"] = score_query_set(
            ctr_q, search_fn, "country", "country-only"
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results
