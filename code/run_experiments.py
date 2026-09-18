"""Provided runner: fixed experiments, development selection, and artifacts."""
from __future__ import annotations
import argparse
import copy
import importlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch.utils.data import DataLoader
import support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--models", nargs="+", choices=list("12345"), default=list("12345"))
    parser.add_argument("--module", default="hw1", help="Submission module to import")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    base = cfg_path.parent
    resolve = lambda value: (base / value).resolve()
    out = resolve(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    hw = importlib.import_module(args.module)
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    records = {name: hw.get_data(resolve(cfg[name])) for name in ("train", "valid")}
    support.ensure_disjoint(records)
    vocab = hw.build_vocab(records["train"], cfg["min_word_frequency"])
    features = hw.LinguisticFeatures().fit(records["train"]) if "3" in args.models else None
    assets = resolve(cfg["asset_dir"])
    manifest = None
    qwen_table = qwen_mapping = piece_counts = None
    if "4" in args.models:
        raw, tok, manifest = support.load_qwen_resources(assets)
        qwen_table, qwen_mapping, piece_counts = support.prepare_qwen_table(
            records, raw, tok, hw)
    if "5" in args.models and manifest is None:
        manifest = json.loads((assets / "manifest.json").read_text())
    bert = {}
    if "5" in args.models:
        bert = {split: support.load_bert_cache(assets / f"bert_{split}.npz", recs,
                                             manifest["bert_dim"])
                for split, recs in records.items()}
    for mode in args.models:
        datasets = {split: support.TaggingDataset(
            recs, mode, hw, vocab=vocab, featurizer=features,
            qwen_table=qwen_table, qwen_mapping=qwen_mapping,
            bert_vectors=bert.get(split)) for split, recs in records.items()}
        for seed in cfg["seeds"]:
            support.set_seed(seed)
            common = dict(hidden_dim=cfg["hidden_dim"], num_tags=len(hw.TAGS),
                          dropout=cfg["dropout"])
            if mode in "123":
                model = hw.WindowTagger(len(vocab), radius=0 if mode == "1" else 2,
                    feature_dim=features.dim if mode == "3" else 0,
                    embedding_dim=cfg["embedding_dim"], **common)
            elif mode == "4":
                model = hw.QwenTagger(qwen_table.shape[1],
                    projection_dim=cfg.get("projection_dim"), **common)
            else:
                model = hw.BertTagger(manifest["bert_dim"], **common)
            model.to(device)
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                         lr=cfg["learning_rate"], weight_decay=0.0)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"],
                shuffle=True, generator=generator, num_workers=0)
            valid_loader = DataLoader(datasets["valid"], batch_size=cfg["batch_size"],
                shuffle=False, num_workers=0)
            history, best_accuracy, best_state, best_epoch = [], -1.0, None, None
            for epoch in range(1, cfg["epochs"] + 1):
                train_stats = hw.train_epoch(model, train_loader, optimizer, device)
                valid_stats = hw.evaluate(model, valid_loader, device)
                history.append({"epoch": epoch, "train_loss": train_stats["loss"],
                    "train_accuracy": train_stats["accuracy"],
                    "valid_loss": valid_stats["loss"],
                    "valid_accuracy": valid_stats["accuracy"]})
                if valid_stats["accuracy"] > best_accuracy:
                    best_accuracy, best_epoch = valid_stats["accuracy"], epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"{mode} seed={seed} epoch={epoch}: "
                      f"valid_accuracy={valid_stats['accuracy']:.4f}", flush=True)
            model.load_state_dict(best_state)
            final = hw.evaluate(model, valid_loader, device)
            prefix = out / f"{mode}_seed{seed}"
            torch.save({"state_dict": best_state, "model": mode, "seed": seed,
                "epoch": best_epoch, "vocab": vocab, "config": cfg,
                "suffix_vocabs": features.suffix_vocabs if mode == "3" else None,
                "source_dim": qwen_table.shape[1] if mode == "4" else
                    manifest["bert_dim"] if mode == "5" else None,
                "train_fingerprint": support.fingerprint(records["train"])},
                prefix.with_suffix(".pt"))
            result = {"model": mode, "seed": seed, "best_epoch": best_epoch,
                "projection_dim": cfg.get("projection_dim") if mode == "4" else None,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "history": history,
                "metrics": support.slice_metrics(records["train"], records["valid"],
                    final["predictions"], hw, piece_counts if mode == "4" else None),
                "errors": support.select_errors(records["valid"], final["predictions"], hw),
                "predictions": final["predictions"],
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "numpy": np.__version__}, "device": str(device)}
            prefix.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
