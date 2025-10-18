# Instructions for Real-World Experiments

This directory provides everything you need to reproduce the **real-world experiments** from our work — specifically, extracting features from LLM activations using the **Linear IDOL** framework.

---


## Running the Code

Create a virtual environment and install dependencies with:

```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

To train a **Linear IDOL** model with:

* `tau = 20`
* `z_dim = 768`
* 8th-layer activations of a language model
* 50M tokens (in batches of 0.1M tokens each)

Run the following command:

```bash
python main.py \
  --tau 20 \
  --buffer-size 0.1m \
  --total-tokens 50m \
  --topk 25 \
  --noise-mode lap \
  --out-batch-ratio 0.1 \
  --lr 0.01 \
  --seed 456 \
  --layer 8 \
  --z-dim 768 \
  --results-dir path/to/your/savingdir
```

### Output Directory Structure

Results will be saved under the directory specified by `--results-dir`, following a structure like:

```
results/
└── EleutherAI-pythia-160m-deduped/
    └── unspecified/
        └── monology-pile-uncopyrighted/
            └── [PARAMS]/
                ├── ckps/
                │   ├── ckpsToken_25.01M.ckp
                │   └── ckpsToken_50.00M.ckp
                ├── [PARAMS]_log.output
                ├── [PARAMS]_loss.png
                └── [PARAMS]_loss.json
```

`[PARAMS]` is a placeholder for the full parameter configuration used in the experiment.

---

## Special Notes on Submodule: `examples/submodule_dl`

We use the [**dictionary_learning**](https://github.com/saprmarks/dictionary_learning/tree/main)
 repo as a submodule for efficient LLM activation generation. However, the default buffer implementation **shuffles** activation data during refresh, which **breaks temporal dependencies** — a crucial issue for our setting.

The code we uploaded already handle this, if you are directly cloning from github then you need to do the following:

To resolve this:

### Initialize the Submodule

```bash
git submodule update --init --recursive
```

### Keep Submodule Updated

```bash
git pull --recurse-submodules
git submodule update --remote --merge
```

### Disable Buffer Shuffling (Important!)

Edit the following line in the submodule:

**File:** `examples/submodule_dl/dictionary_learning/buffer.py`
**Line:** 74 (as of May 15, 2025)

**Replace:**

```python
idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
```

**With:**

```python
idxs = unreads[:self.out_batch_size]
```

This ensures temporal ordering is preserved while still benefiting from efficient batching.
