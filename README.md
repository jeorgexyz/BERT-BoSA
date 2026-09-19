# BERT BoSA
### Bidirectional Encoder Representations from Transformers using Boltzmann Simulated Annealing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeorgexyz/BERT-BoSA/blob/main/BERT_BoSA_Colab.ipynb)

BERT BoSA trains a BERT-style transformer encoder from scratch for text classification, using Simulated Annealing to search over training hyperparameters. The SA search perturbs the **learning rate**, **dropout rate**, and **number of encoder layers**, scoring each candidate by training a fresh model for a fixed number of batches and measuring its accuracy on a held-out validation split. The best configuration found is then used for the full training run.

Note that this is the BERT *architecture* trained directly on the classification task — there is no masked-language-model pretraining — so expect from-scratch accuracy rather than pretrained-BERT numbers. For context, simple bag-of-embeddings baselines reach about 92% on AG_NEWS; this repo exists to demonstrate SA-driven hyperparameter search, not to beat them.

## Requirements

- Python 3.8 or later
- PyTorch
- HuggingFace `datasets` (for AG_NEWS)
- matplotlib (only for `plot_sa.py`)

Install with:

```
pip install -r requirements.txt
```

## Dataset

The AG_NEWS dataset (4-class news topic classification) is downloaded automatically via the HuggingFace `datasets` library (`fancyzhx/ag_news`) on first run and cached locally. 5% of the training set is held out as a validation split, used by the SA search and for per-epoch monitoring; the vocabulary is built from the remaining training texts.

## Folder Structure

```
BERT-BoSA/
├── models/
│   ├── bert.py                  # embeddings + transformer encoder (with padding mask)
│   └── classification_head.py   # dropout + linear over the [CLS] token
├── utils/
│   ├── data_utils.py            # dataset loading, vocab, dataloaders
│   └── train_utils.py           # shared accuracy computation
├── optimization/
│   └── simulated_annealing.py   # SA search + candidate evaluation
├── config.py                    # initial + fixed hyperparameters
├── train.py                     # SA search, full training, checkpointing
├── eval.py                      # test-set evaluation of a checkpoint
├── plot_sa.py                   # plots the SA trace from sa_log.csv
├── BERT_BoSA_Colab.ipynb        # end-to-end run on a Colab GPU
├── requirements.txt
├── LICENSE
└── README.md
```

## Usage

1. **Configuration**: `config.py` holds the initial values of the SA-optimizable parameters (`initial_config`) and the fixed hyperparameters (`FIXED_CONFIG`).

2. **Training**:

   ```
   python train.py
   ```

   This runs the SA search, retrains with the best configuration found, and saves the model weights, configuration, and vocabulary to `model.pt`. Useful flags:

   ```
   --sa-iterations N     SA iterations (default 15)
   --sa-train-batches N  training batches per SA candidate (default 100)
   --skip-sa             skip the search and train with the initial config
   --epochs N            epochs for the final training run
   --seed N              random seed (default 42)
   --output PATH         checkpoint path (default model.pt)
   --sa-log PATH         per-iteration SA trace CSV (default sa_log.csv)
   ```

   Each SA iteration trains a candidate model for `--sa-train-batches` batches, so the search cost scales with both flags. Candidate evaluations use a fixed seed so configs are compared under identical initialisation and batch order. The cooling rate is derived from `--sa-iterations` so the temperature always decays from 0.1 to 0.001 over the full run.

   **Runtime:** the default model (8 layers, 512-dim) takes roughly 2 s per batch on a modern CPU, so a full run is many hours without a GPU. Use `--skip-sa`, fewer `--sa-iterations`/`--sa-train-batches`, or shrink `embedding_dim`/`n_layers` in `config.py` to experiment on CPU.

3. **Evaluation**:

   ```
   python eval.py --checkpoint model.pt
   ```

   This restores the model and its training vocabulary from the checkpoint and reports accuracy on the AG_NEWS test set.

4. **Plotting the search**:

   ```
   python plot_sa.py --log sa_log.csv --output sa_trace.png
   ```

   Plots validation accuracy of each candidate (accepted vs. rejected), the current and best-so-far configs, and the temperature schedule.

**No GPU?** Use the Colab badge at the top — the notebook runs all of the above on a free Colab GPU and lets you download the plot and logs.

## Customization

- **Dataset**: To use a different text classification dataset, modify `_load_ag_news` in `utils/data_utils.py` and update `num_classes` in `config.py`.
- **Model**: `models/bert.py` and `models/classification_head.py` define the architecture.
- **Search space**: `make_random_perturbation` in `optimization/simulated_annealing.py` defines which hyperparameters SA explores and how they are perturbed.

## How the search works

Each SA step perturbs one hyperparameter (learning rate ×[0.5, 2] clamped to [1e-5, 1e-3]; dropout ±0.1 clamped to [0, 0.5]; layers ±2 clamped to [1, 16]) and scores the candidate as `cost = 1 − val_accuracy`. A better candidate is always accepted; a worse one is accepted with the Boltzmann/Metropolis probability `exp(−Δcost / T)` — the "Boltzmann" in the name. Because cost is on the accuracy scale, a starting temperature of 0.1 accepts a 5-point accuracy drop with probability ≈ 0.6, letting the search escape local optima early and becoming greedy as `T` cools.

## Known limitations

- **Short-horizon bias**: candidates are scored after only `--sa-train-batches` steps, which favours configs that learn fast (shallower models, higher learning rates) over those that would win with full training.
- **Segment embeddings** are included for fidelity to BERT but are always zero, since classification inputs are single sentences.
- **Head dropout** is fixed at 0.1 and is not part of the SA search space.
- **No pretraining, warmup, or LR schedule** — deliberate, to keep the code small and focused on the search.

## License

MIT — see [LICENSE](LICENSE).
