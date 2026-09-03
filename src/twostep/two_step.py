import json
from argparse import ArgumentParser

import faiss
import numpy as np
from tqdm import tqdm

from src.model.backbones import MODEL_NAMES, load_backbone
from src.utils import get_device, read_index
from src.metrics import evaluate


def _retrieve(
    faiss_index,
    query_embeds_np: np.ndarray,
    topk: int,
    indices_filter: np.ndarray = None,
):
    results = []
    for ix, embed in enumerate(tqdm(query_embeds_np, desc="retrieve", leave=False)):
        if indices_filter is not None:
            cand = np.ascontiguousarray(
                indices_filter[ix][indices_filter[ix] >= 0], dtype="int64"
            )
            selector = faiss.IDSelectorArray(cand)
            params = faiss.SearchParameters(sel=selector)
            _, indices = faiss_index.search(embed.reshape(1, -1), topk, params=params)
        else:
            _, indices = faiss_index.search(embed.reshape(1, -1), topk)
        indices = indices[0]
        results.append(indices)
    return np.stack(results, axis=0)


def load_query(path: str, min_relevant: int = 5):
    with open(path, "r") as f:
        query = json.loads(f.read())["data"]
        query = [q for q in query if len(q["relevant_ids"]) >= 5]
    return query


def main(args):
    index, meta = read_index(args.index_dir)
    gps_index, _ = read_index(args.index_dir, prefix="gps")
    meta = meta["metadata"]

    backbone = load_backbone(args.model, get_device())

    query = load_query(args.test_query, args.min_relevant)
    all_gt = [q["relevant_ids"] for q in query]

    query_texts = [q["text"] for q in query]
    query_emb = backbone.encode_text(query_texts, args.batch_size)

    # first pass
    first_pass = _retrieve(gps_index, query_emb, args.topk_first)

    # second pass
    second_pass = _retrieve(index, query_emb, args.topk_second, first_pass)
    all_preds = [[meta[i]["id"] for i in row] for row in second_pass]

    # metrics
    result = evaluate(all_preds, all_gt)
    print(result)

    with open(args.output, "w") as f:
        f.write(json.dumps(result, indent=4))

    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--test-query", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--topk-first", type=int, default=10_000)
    parser.add_argument("--topk-second", type=int, default=100)
    parser.add_argument("--min-relevant", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()
    main(args)
