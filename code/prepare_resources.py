"""Download pinned homework resources and prepare the inputs used by the runner."""
import argparse
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, BertModel

import hw1
import support

ROOT = Path(__file__).resolve().parent
GUM_REVISION = 'b58e74bc22d17220c9198864c253a50a897bf27f'
QWEN_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
BERT_REVISION = 'cd5ef92a9fb2f889e972770a36d4ed042daf221e'


def fetch(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    partial = target.with_suffix(target.suffix + ".partial")
    try:
        print(f"Downloading {target.name}", flush=True)
        with urlopen(url, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, 8 * 1024 * 1024)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def convert_data(text):
    records, words, tags, sid = [], [], [], None
    for line in text.splitlines() + [""]:
        if not line.strip():
            if words:
                if not sid:
                    raise ValueError("Missing sentence identifier")
                records.append({"id": sid, "tokens": words, "tags": tags})
            words, tags, sid = [], [], None
        elif line.startswith("# sent_id = "):
            sid = line.split(" = ", 1)[1]
        elif not line.startswith("#"):
            columns = line.split("\t")
            if columns[0].isdigit():
                words.append(columns[1])
                tags.append(columns[3])
    return records


def write_bert_cache(records, tokenizer, encoder, target):
    vectors = []
    for i, record in enumerate(records):
        vector = hw1.extract_bert_words(record["tokens"], tokenizer, encoder)
        if vector.shape != (len(record["tokens"]), encoder.config.hidden_size):
            raise ValueError("extract_bert_words returned an incorrect shape")
        array = vector.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError("extract_bert_words returned nonfinite values")
        vectors.append(array)
        if (i + 1) % 250 == 0:
            print(f"{target.name}: {i + 1}/{len(records)} sentences", flush=True)
    temporary = target.with_suffix(".partial.npz")
    try:
        np.savez(temporary, vectors=np.concatenate(vectors),
                 offsets=np.cumsum([0] + [len(r["tokens"]) for r in records]),
                 ids=np.array([r["id"] for r in records]),
                 fingerprint=np.array(support.fingerprint(records)))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    data, assets = ROOT / "data", ROOT / "assets"
    scratch = assets / ".downloads"
    scratch.mkdir(parents=True, exist_ok=True)
    datasets = {}
    try:
        gum_url = f"https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/{GUM_REVISION}"
        for split, source in [("train", "train"), ("valid", "dev")]:
            name = f"en_gum-ud-{source}.conllu"
            raw = fetch(f"{gum_url}/{name}", scratch / name)
            records = convert_data(raw.read_text(encoding="utf-8"))
            data.mkdir(exist_ok=True)
            (data / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
            datasets[split] = records
        fetch(f"{gum_url}/LICENSE.txt", data / "LICENSE.txt")
        qwen_url = f"https://huggingface.co/Qwen/Qwen3.8-27B/resolve/{QWEN_REVISION}"
        fetch(f"{qwen_url}/tokenizer.json", assets / "qwen_tokenizer.json")
        fetch(f"{qwen_url}/LICENSE", assets / "QWEN_LICENSE")
        table = assets / "qwen_embeddings.npy"
        if not table.exists():
            name = "model-00003-of-00018.safetensors"
            shard = fetch(f"{qwen_url}/{name}", scratch / name)
            partial = assets / "qwen_embeddings.partial.npy"
            with safe_open(shard, framework="pt", device="cpu") as weights:
                tensor = weights.get_slice("model.language_model.embed_tokens.weight")
                shape = tensor.get_shape()
                array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=tuple(shape))
                for start in range(0, shape[0], 1024):
                    array[start:start + 1024] = tensor[start:start + 1024].float().numpy().astype(np.float16)
                array.flush()
                del array
            partial.replace(table)
            shard.unlink()
        shape = np.load(table, mmap_mode="r", allow_pickle=False).shape
        if shape != (248320, 5120):
            raise ValueError("Incorrect Qwen embedding shape; remove qwen_embeddings.npy and rerun")
        bert_url = f"https://huggingface.co/google-bert/bert-base-cased/resolve/{BERT_REVISION}"
        bert_dir = assets / "bert-base-cased"
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors", "README.md"):
            fetch(f"{bert_url}/{name}", bert_dir / name)
        tokenizer = AutoTokenizer.from_pretrained(bert_dir, use_fast=True, local_files_only=True)
        encoder = BertModel.from_pretrained(bert_dir, local_files_only=True).float().eval().to(args.device)
        encoder.requires_grad_(False)
        for split, records in datasets.items():
            write_bert_cache(records, tokenizer, encoder, assets / f"bert_{split}.npz")
        manifest = {"resource_id": "cs577-hw1-gum-r2.18", "qwen_model": "Qwen/Qwen3.8-27B",
                    "qwen_revision": QWEN_REVISION, "qwen_shape": list(shape),
                    "bert_model": "google-bert/bert-base-cased", "bert_revision": BERT_REVISION,
                    "bert_dim": encoder.config.hidden_size}
        (assets / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("Resources ready. Run python run_experiments.py --config config.json")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        (assets / "qwen_embeddings.partial.npy").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
