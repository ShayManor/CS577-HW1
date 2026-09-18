# Setup

Use Python 3.12. Run commands from `hw1/code/`. Complete `hw1.py`; keep the other Python files unchanged.

On `data.cs.purdue.edu`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install test/cs577_hw1_checks-1.3.0-cp310-abi3-linux_x86_64.whl
```

For local training:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `py -3.12 -m venv .venv` and activate with `.venv\Scripts\Activate.ps1`. Reactivate the environment in each new shell.

# Resources and training

Sources: [GUM data](https://github.com/UniversalDependencies/UD_English-GUM/tree/b58e74bc22d17220c9198864c253a50a897bf27f), [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), and [BERT](https://huggingface.co/google-bert/bert-base-cased/tree/cd5ef92a9fb2f889e972770a36d4ed042daf221e).

After implementing `extract_bert_words`, download and prepare the resources:

```bash
python prepare_resources.py
```

This downloads the specified data and model files, extracts Qwen's input embeddings, and calls `extract_bert_words` to cache BERT contextual word representations. BERT stays frozen, so these cached representations are reused across epochs and seeds. It downloads only the required Qwen shard, not the full model. Rerun after changing `extract_bert_words`.

After completing the remaining functions, train all five models and generate report outputs:

```bash
python run_experiments.py --config config.json
python summarize_results.py --results results
```

Outputs are saved in `results/`. CPU is the default; add `--device cuda` or `--device mps` to preparation or training when supported.

# Check and submit

Run on `data.cs.purdue.edu`:

```bash
hw1 check --submission hw1.py
hw1 submit --submission hw1.py
turnin -c cs577 -p hw1 -v
```

`hw1 check` prints each check’s earned points and weight, task subtotals, and the coding score out of 45; it also writes `check_report.json`. Optional and diagnostic checks have weight 0.

Only `hw1.py` is submitted. Submitting replaces your previous submission.
