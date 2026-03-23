import argparse
import json
import logging
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import polars as pl
from tqdm import tqdm

from src.utils import read_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  BALANCED SEED SAMPLING  (category x country grid)
# ══════════════════════════════════════════════════════════════════════════════


def sample_balanced_seeds(df: pl.DataFrame, total_seeds: int, random_seed: int = 42):
    landmark_representatives = df.group_by(
        *["pred_label", "country", "landmark_id"]
    ).map_groups(lambda g: g.sample(n=1, seed=random_seed, with_replacement=False))

    # Step 3 — quota per (pred_label, country) cell
    n_cells = landmark_representatives.select(["pred_label", "country"]).unique().height
    quota = max(1, total_seeds // n_cells)
    log.info(f"Cells: {n_cells:,}  |  quota per cell: {quota}")

    # Step 4 — sample within each cell from landmark representatives only
    sampled = landmark_representatives.group_by(*["pred_label", "country"]).map_groups(
        lambda g: g.sample(
            n=min(quota, len(g)),
            seed=random_seed,
            with_replacement=False,
        )
    )

    seed_indices = sampled["id"].to_list()

    rng = random.Random(random_seed)
    rng.shuffle(seed_indices)

    log.info(f"Sampled {len(seed_indices):,} seeds (requested {total_seeds:,})")
    return seed_indices


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SUBSET CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════


def build_subset(
    seed_id: str,
    meta: list[dict],
    faiss_index: faiss.IndexHNSWFlat,
    target_k=10,
    min_countries=5,
    topk_search=300,
    max_dup_sim=0.95,
    div_gap=0.002,
):
    """
    Build one subset anchored at seed_row_idx.

    Algorithm (CIRR-adapted for cross-landmark cross-country)
    ----------------------------------------------------------
    1. FAISS: find top-K visually similar images to the seed.
    2. Drop: same landmark_id as seed  (cross-landmark is mandatory).
    3. Drop: cosine_sim >= max_dup_sim  (CIRR near-duplicate rule).
    4. Greedy pass 1 — prefer new countries:
         Add candidate if diversity gap OK and either the country is new
         OR we already have >= min_countries (soft enforcement).
    5. Greedy pass 2 — relax country constraint to fill remaining slots.
    6. Keep subset even if incomplete (fewer than target_k candidates).

    Returns a subset dict or None if no valid candidates exist.
    """
    seed_idx, seed_meta = next((i, d) for i, d in enumerate(meta) if d["id"] == seed_id)
    seed_feat = faiss_index.reconstruct(seed_idx).reshape(1, -1)

    # ── FAISS search ──────────────────────────────────────────────────────────
    sims, idxs = faiss_index.search(seed_feat, topk_search + 1)
    sims = sims[0]
    idxs = idxs[0]

    # ── Build filtered candidate list ─────────────────────────────────────────
    candidates = []
    for idx, sim in zip(idxs, sims):
        idx = int(idx)
        if idx == seed_id:  # skip self
            continue
        if sim >= max_dup_sim:  # near-dup filter (CIRR)
            continue
        if meta[idx]["category"] != seed_meta["category"]:
            continue
        if meta[idx]["country"] == seed_meta["country"]:
            continue
        candidates.append((idx, float(sim)))

    if not candidates:
        return None

    # ── Greedy pass 1: prefer new countries ───────────────────────────────────
    selected_ids = []
    selected_idxs = []
    selected_sims = []
    country_counts = defaultdict(int)
    last_sim = None

    # tracker to ensure category-country pairs diversity
    # occupied_slots = {(meta[seed_idx]["pred_label"], meta[seed_idx]["country"])}

    for cand_idx, cand_sim in candidates:
        if len(selected_ids) >= target_k:
            break

        # Diversity gap filter (CIRR rule)
        if last_sim is not None and abs(cand_sim - last_sim) < div_gap:
            continue

        cand_country = meta[cand_idx]["country"]
        # cand_category = meta[cand_idx]["pred_label"]
        # slot = (cand_country, cand_category)

        # if slot in occupied_slots:
        #     continue

        # Soft country rule: defer repeated countries until min_countries reached
        if country_counts[cand_country] > 0 and len(country_counts) < min_countries:
            continue

        selected_ids.append(meta[cand_idx]["id"])
        selected_idxs.append(cand_idx)
        selected_sims.append(cand_sim)
        country_counts[cand_country] += 1
        last_sim = cand_sim
        # occupied_slots.add(slot)

    # ── Greedy pass 2: fill remaining slots, country constraint relaxed ────────
    if len(selected_ids) < target_k:
        selected_set = set(selected_ids)
        last_sim = selected_sims[-1] if selected_sims else None

        for cand_idx, cand_sim in candidates:
            if len(selected_ids) >= target_k:
                break
            if cand_idx in selected_set:
                continue
            if last_sim is not None and abs(cand_sim - last_sim) < div_gap:
                continue
            
            # slot = (meta[cand_idx]["pred_label"], meta[cand_idx]["country"])
            
            # if slot in occupied_slots:
                # continue

            selected_ids.append(meta[cand_idx]["id"])
            selected_idxs.append(cand_idx)
            selected_sims.append(cand_sim)
            country_counts[meta[cand_idx]["country"]] += 1
            last_sim = cand_sim
            # occupied_slots.add(slot)

    if not selected_ids:
        return None

    all_ids = [seed_id] + selected_ids
    all_idxs = [
        next(i for i, d in enumerate(meta) if d["id"] == seed_id)
    ] + selected_idxs
    all_ctrs = [meta[i]["country"] for i in all_idxs]
    all_categories = [meta[i]["category"] for i in all_idxs]

    return {
        "subset_id": f"sub_{seed_id}",
        "seed_id": seed_id,
        "member_ids": all_ids,
        "member_idxs": all_idxs,
        "countries": all_ctrs,
        "categories": all_categories,
        "n_countries": len(set(all_ctrs[1:])),  # diversity among candidates
        "complete": len(selected_ids) == target_k,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  OVERLAP FILTER  (mirrors CIRR section 4.1)
# ══════════════════════════════════════════════════════════════════════════════


def filter_overlapping_subsets(subsets, max_shared=3, random_seed=42):
    """
    Greedy overlap filter.

    Iterates through subsets in random order. Keeps a subset only if it
    shares <= max_shared image_ids with every already-kept subset.

    max_shared=3 is a good default: allows minor overlap while preventing
    the split leakage problem where the same (ref, tgt) pair appears in
    both train and val because Eiffel Tower seeded sub_A and Tokyo Tower
    seeded sub_B, and both subsets contain the same pair in different roles.
    """
    log.info(f"Overlap filter: max_shared={max_shared} ...")
    rng = random.Random(random_seed)
    shuffled = subsets[:]
    rng.shuffle(shuffled)

    kept = []
    kept_member_sets = []

    for subset in tqdm(shuffled, desc="Overlap filter", unit="subset"):
        member_set = set(subset["member_ids"])
        too_much = any(
            len(member_set & existing) > max_shared for existing in kept_member_sets
        )
        if not too_much:
            kept.append(subset)
            kept_member_sets.append(member_set)

    log.info(
        f"Overlap filter: {len(subsets):,} -> {len(kept):,} subsets  "
        f"({len(subsets) - len(kept):,} removed)"
    )
    return kept


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PAIR GENERATION  —  same category, different country
# ══════════════════════════════════════════════════════════════════════════════


def generate_pairs(subset, meta, max_pairs_per_ref=5, random_seed=42):
    """
    Generate directed (reference -> target) pairs from one subset.

    Rules
    -----
    - Reference and target must have DIFFERENT countries.
    - Reference and target must share the SAME landmark category.
    - Each (reference, target) combination becomes one separate pair with its
      own caption field (to be filled in Stage 4).
      This is one-to-one per CIRR's annotation design — NOT one caption
      pointing to many targets.

    Sampling
    --------
    If a reference image has more valid targets than max_pairs_per_ref,
    we randomly sample max_pairs_per_ref of them to keep pair counts
    manageable.

    target_soft
    -----------
    Populated as {image_id: 1.0} for now (one certain target).
    After a Stage 4 auxiliary annotation pass, additional subset members
    that annotators agree also satisfy the caption can be added here with
    fractional scores, matching the CIRR JSON schema exactly.
    """
    rng = random.Random(random_seed)
    pairs = []
    member_idxs = subset["member_idxs"]
    member_ids = subset["member_ids"]
    subset_id = subset["subset_id"]

    for ref_pos, ref_idx in enumerate(member_idxs):
        ref_country = meta[ref_idx]["country"]
        ref_category = meta[ref_idx]["pred_label"]

        # Collect all valid targets for this reference image
        valid_targets = []  # list of (member_pos, image_id)
        for tgt_pos, tgt_idx in enumerate(member_idxs):
            if tgt_pos == ref_pos:
                continue
            if meta[tgt_idx]["country"] == ref_country:  # must be cross-country
                continue
            if meta[tgt_idx]["pred_label"] != ref_category:  # must be same category
                continue
            valid_targets.append((tgt_pos, member_ids[tgt_pos]))

        if not valid_targets:
            continue

        # Sample down to max_pairs_per_ref
        if len(valid_targets) > max_pairs_per_ref:
            valid_targets = rng.sample(valid_targets, max_pairs_per_ref)

        for tgt_pos, tgt_id in valid_targets:
            pairs.append(
                {
                    "pair_id": f"{subset_id}_r{ref_pos:02d}_t{tgt_pos:02d}",
                    "subset_id": subset_id,
                    "reference": member_ids[ref_pos],
                    "target_hard": tgt_id,
                    "target_soft": {tgt_id: 1.0},  # extend after Stage 4
                    "caption": "",  # filled in Stage 4
                    "img_set": {
                        "members": member_ids,
                        "hard_negatives": [],  # optional Tier-2 mining
                    },
                    "meta": {
                        "ref_country": ref_country,
                        "ref_category": ref_category,
                        "tgt_country": meta[member_idxs[tgt_pos]]["country"],
                        "tgt_category": meta[member_idxs[tgt_pos]]["pred_label"],
                    },
                }
            )

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CORRIDOR CAP  (post-sampling balance guard)
# ══════════════════════════════════════════════════════════════════════════════


def cap_pairs_by_corridor(pairs, cap, random_seed=42):
    """
    Cap pairs per (ref_country -> tgt_country) corridor.

    This is a safety net for cases where one geographic corridor (e.g. IT->FR)
    is genuinely over-represented even after balanced seeding. Only triggers
    when a corridor exceeds `cap` pairs.
    """
    rng = random.Random(random_seed)
    corridor = defaultdict(list)

    for p in pairs:
        key = (p["meta"]["ref_country"], p["meta"]["tgt_country"])
        corridor[key].append(p)

    capped = []
    for key, corridor_pairs in corridor.items():
        if len(corridor_pairs) > cap:
            corridor_pairs = rng.sample(corridor_pairs, cap)
        capped.extend(corridor_pairs)

    removed = len(pairs) - len(capped)
    if removed:
        log.info(f"Corridor cap ({cap}): removed {removed:,} pairs")
    return capped


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SPLIT  (always at subset level — never image level)
# ══════════════════════════════════════════════════════════════════════════════


def split_subsets(subsets, ratios=(0.8, 0.1, 0.1), random_seed=42):
    """
    Assign subsets to train/val/test.
    Splitting at subset level guarantees no image appears in multiple splits.
    """
    rng = random.Random(random_seed)
    shuffled = subsets[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


# ══════════════════════════════════════════════════════════════════════════════
# 9.  STATISTICS
# ══════════════════════════════════════════════════════════════════════════════


def log_stats(subsets, pairs, split_name="all"):
    if not subsets:
        log.info(f"[{split_name}] No subsets.")
        return

    n_sub = len(subsets)
    n_complete = sum(1 for s in subsets if s["complete"])
    n_members = [len(s["member_ids"]) for s in subsets]
    n_ctrs = [s["n_countries"] for s in subsets]
    n_pairs = len(pairs)

    # Top-5 country corridors by pair count
    corridor_counts = defaultdict(int)
    for p in pairs:
        key = f"{p['meta']['ref_country']}->{p['meta']['tgt_country']}"
        corridor_counts[key] += 1
    top5 = sorted(corridor_counts.items(), key=lambda x: -x[1])[:5]
    top5_str = "  ".join(f"{k}:{v}" for k, v in top5)

    log.info(
        f"\n{'─' * 60}\n"
        f"  Split          : {split_name}\n"
        f"  Subsets        : {n_sub:,}  "
        f"(complete: {n_complete:,} = {100 * n_complete / n_sub:.1f}%)\n"
        f"  Avg members    : {np.mean(n_members):.1f}  "
        f"(min {min(n_members)}, max {max(n_members)})\n"
        f"  Avg countries  : {np.mean(n_ctrs):.1f}  "
        f"(>=3: {sum(1 for c in n_ctrs if c >= 3):,})\n"
        f"  Pairs          : {n_pairs:,}\n"
        f"  Top-5 corridors: {top5_str}\n"
        f"{'─' * 60}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build GLDv2-CIR cross-country subsets and pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--faiss-index", required=True)
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for JSON annotation files"
    )
    parser.add_argument("--n_seeds", type=int, default=80_000)
    parser.add_argument(
        "--target_k",
        type=int,
        default=10,
        help="Target candidates per subset (excl. seed)",
    )
    parser.add_argument(
        "--topk_search", type=int, default=300, help="FAISS neighbours fetched per seed"
    )
    parser.add_argument("--max_dup_sim", type=float, default=0.94)
    parser.add_argument("--min_countries", type=int, default=10)
    parser.add_argument("--div_gap", type=float, default=0.002)
    parser.add_argument(
        "--max_shared",
        type=int,
        default=3,
        help="Overlap filter: max shared members between kept subsets",
    )
    parser.add_argument(
        "--max_pairs_per_ref",
        type=int,
        default=3,
        help="Max target samples per reference image per subset",
    )
    parser.add_argument(
        "--pair_cap_per_corridor",
        type=int,
        default=200,
        help="Max pairs per (ref_country, tgt_country) corridor",
    )
    parser.add_argument(
        "--split_ratios",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip seeds already present in subsets_raw.jsonl",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pl.read_csv(args.data_path)
    index, meta = read_index(args.faiss_index)
    meta = meta["metadata"]

    # ── Balanced seed sampling ─────────────────────────────────────────────────
    if not os.path.exists("./seed_indices"):
        seed_indices = sample_balanced_seeds(
            df=df,
            total_seeds=args.n_seeds,
            random_seed=args.random_seed,
        )

        with open("./seed_indices", "wb") as f:
            pickle.dump(seed_indices, f)
    else:
        with open("./seed_indices", "rb") as f:
            seed_indices = pickle.load(f)

    existing_ids = set()
    subset_path = out_dir / "subsets_raw.jsonl"

    # ── Subset construction loop ───────────────────────────────────────────────
    all_subsets = []
    discarded = 0
    skipped = 0

    with open(subset_path, "w") as raw_fh:
        for i, seed_idx in enumerate(tqdm(seed_indices, desc="Building subsets", unit="seed")):
            subset_id = f"sub_{seed_idx}"
            if subset_id in existing_ids:
                skipped += 1
                continue

            subset = build_subset(
                seed_id=seed_idx,
                meta=meta,
                faiss_index=index,
                target_k=args.target_k,
                topk_search=args.topk_search,
                max_dup_sim=args.max_dup_sim,
                min_countries=args.min_countries,
                div_gap=args.div_gap,
            )

            if subset is None:
                discarded += 1
                continue

            all_subsets.append(subset)
            raw_fh.write(json.dumps(subset) + "\n")  # incremental write

            break

    # log.info(
    #     f"Subsets built: {len(all_subsets):,}  |  "
    #     f"discarded: {discarded:,}  |  skipped (resume): {skipped:,}"
    # )
    # if not all_subsets:
    #     log.error("No subsets built. Check metadata/features alignment.")
    #     return

    # # ── Overlap filter ────────────────────────────────────────────────────────
    # all_subsets = filter_overlapping_subsets(
    #     all_subsets,
    #     max_shared=args.max_shared,
    #     random_seed=args.random_seed,
    # )

    # # ── Pair generation ───────────────────────────────────────────────────────
    # log.info("Generating pairs ...")
    # all_pairs = []
    # no_pair_subs = 0

    # for subset in tqdm(all_subsets, desc="Generating pairs", unit="subset"):
    #     pairs = generate_pairs(
    #         subset=subset,
    #         meta=meta,
    #         max_pairs_per_ref=args.max_pairs_per_ref,
    #         random_seed=args.random_seed,
    #     )
    #     if not pairs:
    #         no_pair_subs += 1
    #     all_pairs.extend(pairs)

    # log.info(
    #     f"Pairs generated: {len(all_pairs):,}  |  "
    #     f"subsets with no valid pairs: {no_pair_subs:,}"
    # )

    # # ── Corridor cap ──────────────────────────────────────────────────────────
    # all_pairs = cap_pairs_by_corridor(
    #     all_pairs,
    #     cap=args.pair_cap_per_corridor,
    #     random_seed=args.random_seed,
    # )

    # # ── Split ─────────────────────────────────────────────────────────────────
    # train_subs, val_subs, test_subs = split_subsets(
    #     all_subsets,
    #     ratios=tuple(args.split_ratios),
    #     random_seed=args.random_seed,
    # )
    # split_lookup = (
    #     {s["subset_id"]: "train" for s in train_subs}
    #     | {s["subset_id"]: "val" for s in val_subs}
    #     | {s["subset_id"]: "test" for s in test_subs}
    # )
    # train_pairs = [p for p in all_pairs if split_lookup.get(p["subset_id"]) == "train"]
    # val_pairs = [p for p in all_pairs if split_lookup.get(p["subset_id"]) == "val"]
    # test_pairs = [p for p in all_pairs if split_lookup.get(p["subset_id"]) == "test"]

    # # ── Stats ─────────────────────────────────────────────────────────────────
    # log_stats(train_subs, train_pairs, "train")
    # log_stats(val_subs, val_pairs, "val")
    # log_stats(test_subs, test_pairs, "test")

    # # ── Write outputs ─────────────────────────────────────────────────────────
    # def write_json(obj, fname):
    #     path = out_dir / fname
    #     with open(path, "w") as f:
    #         json.dump(obj, f, indent=2)
    #     log.info(f"Wrote {path}  ({len(obj):,} entries)")

    # def clean_pairs(pairs):
    #     """Strip internal-only fields before writing annotation files."""
    #     return [{k: v for k, v in p.items() if k != "member_idxs"} for p in pairs]

    # def subset_manifest(subs):
    #     return [
    #         {
    #             "subset_id": s["subset_id"],
    #             "members": s["member_ids"],
    #             "countries": s["countries"],
    #             "n_countries": s["n_countries"],
    #             "complete": s["complete"],
    #         }
    #         for s in subs
    #     ]

    # # def image_ids_from(subs):
    # #     ids = set()
    # #     for s in subs:
    # #         ids.update(s["member_ids"])
    # #     return sorted(ids)

    # # Annotation files (one pair = one future caption)
    # write_json(clean_pairs(train_pairs), "cap.v1.train.json")
    # write_json(clean_pairs(val_pairs), "cap.v1.val.json")
    # write_json(clean_pairs(test_pairs), "cap.v1.test.json")

    # # Subset manifests (for RecallSubset evaluation)
    # write_json(subset_manifest(train_subs), "subsets.v1.train.json")
    # write_json(subset_manifest(val_subs), "subsets.v1.val.json")
    # write_json(subset_manifest(test_subs), "subsets.v1.test.json")

    # # # Image split lists (which images belong to each split's search pool)
    # # write_json(image_ids_from(train_subs), "split.v1.train.json")
    # # write_json(image_ids_from(val_subs), "split.v1.val.json")
    # # write_json(image_ids_from(test_subs), "split.v1.test.json")

    # # log.info(
    # #     f"\n{'=' * 60}\n"
    # #     f"  DONE\n"
    # #     f"  Subsets : {len(all_subsets):,}\n"
    # #     f"  Pairs   : {len(all_pairs):,}\n"
    # #     f"  Output  : {out_dir.resolve()}\n"
    # #     f"{'=' * 60}"
    # # )


if __name__ == "__main__":
    main()
