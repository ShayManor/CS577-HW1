"""Provided experiment utilities. Do not modify for submission."""
from __future__ import annotations
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def fingerprint(records):
    payload = [{"id": r["id"], "tokens": r["tokens"]} for r in records]
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def ensure_disjoint(splits):
    ids = set()
    for name, records in splits.items():
        new = {r["id"] for r in records}
        if new & ids:
            raise ValueError(f"Sentence IDs overlap across splits: {name}")
        ids |= new


def load_bert_cache(path, records, expected_dim):
    """No pickle: a cache contains flat vectors, offsets, ids, fingerprint."""
    with np.load(path, allow_pickle=False) as cache:
        if str(cache["fingerprint"].item()) != fingerprint(records):
            raise ValueError(f"Cache/input fingerprint mismatch: {path}")
        if cache["ids"].tolist() != [r["id"] for r in records]:
            raise ValueError("Cache sentence IDs/order mismatch")
        offsets = cache["offsets"].astype(np.int64)
        expected = np.cumsum([0] + [len(r["tokens"]) for r in records])
        if not np.array_equal(offsets, expected):
            raise ValueError("Cache word offsets mismatch")
        vectors = cache["vectors"].astype(np.float32)
        if vectors.shape != (int(expected[-1]), expected_dim):
            raise ValueError("Cache vector shape mismatch")
        if not np.isfinite(vectors).all():
            raise ValueError("Cache vectors contain non-finite values")
    return vectors


def load_qwen_resources(asset_dir):
    asset_dir = Path(asset_dir)
    meta = json.loads((asset_dir / "manifest.json").read_text())
    table = np.load(asset_dir / "qwen_embeddings.npy", mmap_mode="r",
                    allow_pickle=False)
    if list(table.shape) != meta["qwen_shape"] or table.ndim != 2:
        raise ValueError("Qwen embedding table/manifest shape mismatch")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(asset_dir / "qwen_tokenizer.json"))
    if tokenizer.get_vocab_size(with_added_tokens=True) > table.shape[0]:
        raise ValueError("Tokenizer contains IDs outside the embedding table")
    return table, tokenizer, meta


def prepare_qwen_table(records_by_split, table, tokenizer, hw, chunk_size=128):
    """Pool unique original word strings ONCE, without using their labels.

    A zero row at index 0 is reserved for sentence boundaries. Reusing frozen
    lookup vectors across splits is allowed; no data-fitted statistics are
    estimated here. All pooling is performed by the student's function.
    """
    words = sorted({w for records in records_by_split.values()
                    for r in records for w in r["tokens"]})
    mapping = {word: i + 1 for i, word in enumerate(words)}
    result = np.zeros((len(words) + 1, table.shape[1]), dtype=np.float32)
    counts = {}
    for start in range(0, len(words), chunk_size):
        part = words[start:start + chunk_size]
        pieces = hw.tokenize_word_pieces(part, tokenizer)
        if len(pieces) != len(part) or any(not ids for ids in pieces):
            raise ValueError("Expected one nonempty subword ID list per word")
        width = max(map(len, pieces))
        emb = np.zeros((len(part), width, table.shape[1]), dtype=np.float32)
        mask = np.zeros((len(part), width), dtype=bool)
        for i, ids in enumerate(pieces):
            if any(t < 0 or t >= table.shape[0] for t in ids):
                raise ValueError("Subword ID outside embedding table")
            emb[i, :len(ids)] = table[ids]
            mask[i, :len(ids)] = True
            counts[part[i]] = len(ids)
        with torch.no_grad():
            pooled = hw.mean_pool_subwords(torch.from_numpy(emb),
                                           torch.from_numpy(mask))
        if pooled.shape != (len(part), table.shape[1]):
            raise ValueError("mean_pool_subwords returned an incorrect shape")
        result[start + 1:start + len(part) + 1] = pooled.cpu().numpy()
    return result, mapping, counts


class TaggingDataset(Dataset):
    """One supervised example per original word; window vectors are lazy."""
    def __init__(self, records, mode, hw, vocab=None, featurizer=None,
                 qwen_table=None, qwen_mapping=None, bert_vectors=None):
        self.mode = mode
        self.vector_table = qwen_table
        self.bert_vectors = bert_vectors
        self.features = None
        labels, indices, feats = [], [], []
        for record in records:
            words = record["tokens"]
            labels.extend(hw.TAG_TO_ID[t] for t in record["tags"])
            if mode in "123":
                ids = torch.tensor(hw.encode_words(words, vocab), dtype=torch.long)
                indices.append(hw.make_windows(ids, radius=0 if mode == "1" else 2))
                if mode == "3":
                    feats.append(featurizer.transform(words))
            elif mode == "4":
                ids = torch.tensor([qwen_mapping[w] for w in words], dtype=torch.long)
                indices.append(hw.make_windows(ids, radius=2))
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.indices = torch.cat(indices) if indices else None
        if feats:
            self.features = torch.cat(feats)
        if mode == "5" and (bert_vectors is None or len(bert_vectors) != len(labels)):
            raise ValueError("BERT cache is missing or has an incorrect word count")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.mode in "123":
            inputs = {"windows": self.indices[i]}
            if self.mode == "3":
                inputs["features"] = self.features[i]
        elif self.mode == "4":
            # Advanced indexing copies only this window, not the full corpus.
            values = self.vector_table[self.indices[i].numpy()]
            inputs = {"windows": torch.from_numpy(values)}
        elif self.mode == "5":
            inputs = {"vectors": torch.from_numpy(self.bert_vectors[i])}
        else:
            raise ValueError(f"Unknown configuration {self.mode}")
        return inputs, self.labels[i]


def slice_metrics(train_records, records, predictions, hw, piece_counts=None):
    train_counts = Counter(w.lower() for r in train_records for w in r["tokens"])
    observed = defaultdict(set)
    for r in train_records:
        for w, tag in zip(r["tokens"], r["tags"]):
            observed[w.lower()].add(tag)
    gold = [hw.TAG_TO_ID[t] for r in records for t in r["tags"]]
    words = [w for r in records for w in r["tokens"]]
    if len(predictions) != len(gold):
        raise ValueError("Expected one prediction per original word")
    if any(not isinstance(p, (int, np.integer)) or p < 0 or p >= len(hw.TAGS)
           for p in predictions):
        raise ValueError("Prediction contains an invalid tag ID")
    groups = {"overall": [], "non_punctuation": [], "unseen": [], "ambiguous": []}
    if piece_counts is not None:
        groups.update({"pieces_1": [], "pieces_2": [], "pieces_3plus": []})
    for i, (w, g) in enumerate(zip(words, gold)):
        groups["overall"].append(i)
        if hw.TAGS[g] != "PUNCT":
            groups["non_punctuation"].append(i)
        if train_counts[w.lower()] == 0:
            groups["unseen"].append(i)
        if len(observed[w.lower()]) > 1:
            groups["ambiguous"].append(i)
        if piece_counts is not None:
            count = piece_counts[w]
            if count < 1:
                raise ValueError("Every word must have at least one subword")
            groups["pieces_1" if count == 1 else "pieces_2" if count == 2
                   else "pieces_3plus"].append(i)
    return {name: {"n": len(ids), "accuracy":
                   sum(predictions[i] == gold[i] for i in ids) / len(ids) if ids else None}
            for name, ids in groups.items()}


def select_errors(records, predictions, hw, limit=5):
    """Deterministically choose first errors sorted by (sentence ID, word index)."""
    errors, cursor = [], 0
    for record in records:
        for i, (word, tag) in enumerate(zip(record["tokens"], record["tags"])):
            pred = hw.TAGS[predictions[cursor]]
            if pred != tag:
                errors.append({"id": record["id"], "word_index": i, "word": word,
                               "sentence": " ".join(record["tokens"]),
                               "gold": tag, "predicted": pred})
            cursor += 1
    return sorted(errors, key=lambda x: (x["id"], x["word_index"]))[:limit]
