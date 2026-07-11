# Attention Placement in CNN-LSTM Architectures — A Controlled Study

Code for the Advanced AI course final report:
"Attention Is All You Need... But Where? A Controlled Study of Attention
Placement in CNN-LSTM Architectures"

## What this does

Trains the SAME CNN-LSTM architecture with attention inserted at 3
different positions (before the CNN, between CNN and LSTM, after the
LSTM), across 3 datasets of increasing sequence length (AG News = short,
IMDB = medium, ArXiv = long), across multiple random seeds — to test
whether attention placement affects performance, and whether that effect
depends on sequence length.

## Files

- `model_base.py`      — the shared CNN-LSTM-Attention model. One class,
                          `attention_position` argument controls where
                          attention sits.
- `data_loader.py`      — loads and tokenizes AG News / IMDB / ArXiv.
- `run_experiments.py`  — trains all (position x seed) combinations for
                          one dataset, saves results to results/*.json.
- `requirements.txt`    — Python dependencies.

## Setup

```bash
pip install -r requirements.txt
```

## Running

Run each dataset separately (each command finishes before you run the
next):

```bash
python run_experiments.py --dataset agnews --seeds 1 2 3 4 5
python run_experiments.py --dataset imdb   --seeds 1 2 3 4 5
python run_experiments.py --dataset arxiv  --seeds 1 2 3 4 5
```

Each command runs 3 attention positions x 5 seeds = 15 training runs,
and saves all metrics (accuracy, timing, parameter count, predictions)
to `results/<dataset>_results.json`.

Total across all three commands: 45 training runs.

## Notes

- Requires internet access to download datasets from Hugging Face on
  first run (cached locally afterward).
- CPU training is fine for AG News/IMDB; ArXiv (longest sequences) will
  be slower — reduce `--epochs` if needed for a quick test run.
- Reduce `--seeds` to `1 2` first to sanity-check everything works end
  to end before committing to the full 5-seed run.
