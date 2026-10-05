# When Does Verification Pay Off? A Closer Look at LLMs as Solution Verifiers

### [Paper](https://arxiv.org/abs/2512.02304) | [Project Page](https://agenticlearning.ai/llm-verification/) | [Data](https://huggingface.co/datasets/Jacklu0831/llm-verification-raw)

## Overview

This codebase provides a framework for studying how the end performance of a solver-verifier system depends on factors like model family, model size, post-training, task type, and more. The frameworks supports:

- **Both Real-World and Synthetic Tasks**: Real-world tasks include GSM8K, AIME, MMLU, CommonsenseQA, and GPQA. Synthetic tasks include SAT, Sudoku, Matrix Multiplication (with generation scripts in `src/generate_problems`). See [Supported Datasets](#supported-datasets).
- **Large Suite of Open-Source Models**: Post-trained models from Llama3, Qwen2.5, Qwen3, and DeepSeek-R1 families. Base models from Llama3, Qwen2.5, Qwen3 families. See [Supported Models](#supported-models).
- **Solver and Verifier Evaluation Metrics**: Solver accuracy. Verifier accuracy, TPR, FPR, FNR, Precision, Recall, F1, Gain, etc. See `src/inference.py`.
- **Rejection sampling**: Solver can iteratively re-solve problems that fail test-time verification. See `src/rejection_sampling.py`.
- **Embedding analysis**: Compute pairwise similarity between model outputs across different LLMs. See `src/compute_embedding.py`.
- **Automatic caching**: Efficiently reuse solver outputs when experimenting with different verifier configurations. See `src/solver_cache.py`.

We also release all raw data behind the paper on [Hugging Face](https://huggingface.co/datasets/Jacklu0831/llm-verification-raw): 271,025 solver generations and 70,855 verifier responses, from which every paper figure can be rebuilt (see [Reproducing the Paper Figures](#reproducing-the-paper-figures)).

## Installation

```bash
# Create Python 3.11 environment
uv venv --python 3.11
source .venv/bin/activate

# Install PyTorch with CUDA 11.8 support (adjust for your own CUDA version)
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
uv pip install -r requirements.txt
```

## Reproducing the Paper Figures

`aggregate_data/` holds the CSVs behind the paper's figures. Regenerate all 12 figures in `figures/`:

```bash
python analysis/plot_figures.py
```

To rebuild `aggregate_data/` from the raw solver and verifier outputs on [Hugging Face](https://huggingface.co/datasets/Jacklu0831/llm-verification-raw):

```bash
hf download Jacklu0831/llm-verification-raw --repo-type dataset --local-dir raw_data
python analysis/rebuild_aggregate_data.py --raw-data raw_data --overwrite
```

---

## Scripts and Example Commands

### 1. Main Experiment Pipeline (`src/inference.py`)

The primary script for running solver-verifier evaluations. Generates solutions with a solver model then verify with a verifier model, computing solver and verifier metrics along the way. All sampling automatically supports Multi-GPU (thanks [VLLM](https://github.com/vllm-project/vllm)!).

```bash
python src/inference.py \
    --solver_model_name Qwen/Qwen3-0.6B \
    --verifier_model_name Qwen/Qwen3-0.6B \
    --dataset_name gsm \
    --output_dir result
```

#### Notable arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--solver_model_name` | str | required | HuggingFace model ID for solving |
| `--verifier_model_name` | str | required | HuggingFace model ID for verification |
| `--dataset_name` | str | required | Dataset to use (see [Supported Datasets](#supported-datasets)) |
| `--dataset_subset_ratio` | float | `1.0` | Fraction of dataset to use |
| `--data_generation_kwargs` | str | `""` | Parameters for procedurally-generated datasets (comma-separated key=value pairs) |

---

### 2. Rejection Sampling (`src/rejection_sampling.py`)

Implements iterative rejection sampling: problems that fail verification are re-solved in subsequent rounds. The script tracks metrics per iteration, including solver accuracy at each round, number of newly accepted problems per round, and cumulative compute (GFLOPs).

```bash
python src/rejection_sampling.py \
    --solver_model_name Qwen/Qwen3-0.6B \
    --verifier_model_name Qwen/Qwen3-0.6B \
    --dataset_name gsm \
    --max_attempts 5 \
    --output_dir result
```

#### Notable Arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_attempts` | int | `5` | Maximum solver attempts per problem |
| `--oracle_verifier` | flag | - | Use ground truth for verification instead of LLM |

---

### 3. Embedding Similarity Analysis (`src/compute_embedding.py`)

Computes pairwise semantic similarity between solver outputs from different models using solution embeddings.

```bash
python src/compute_embedding.py \
    --model_names Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B \
    --dataset_name gsm \
    --output_dir utils/similarity_maps
```

**Note:** This script requires cached solver outputs for all specified models. Run `inference.py` (optionally with `--no_verify`) first to populate the cache. See [Solver Cache System](#solver-cache-system) for details.

---

## Supported Models

See `Table 1` in our [paper](https://arxiv.org/abs/2512.02304) for all supported models. To use a model not in this list, add it to `MODEL_SIZES` in `src/inference.py`.

---

## Supported Datasets

### Real-World Datasets

| Dataset Name | Description | Source |
|--------------|-------------|--------|
| `gsm` | GSM8K math word problems | `openai/gsm8k` |
| `aime` | AIME competition math (1983-2025) | `TianHongZXY/aime-1983-2025` |
| `mmlu_[SPLIT]` | All MMLU subjects (57 categories); `SPLIT=all,stem,social_sciences,humanities,other` | `cais/mmlu` |
| `csqa` | CommonsenseQA | `tau/commonsense_qa` |
| `gpqa` | GPQA (graduate-level science) | `Idavidrein/gpqa` |

### Synthetic Datasets

These datasets are generated on-the-fly in `inference.py` with configurable parameters via `--data_generation_kwargs`. In the [paper](https://arxiv.org/abs/2512.02304), we use the following generation parameters, but feel free to play around with them to see how they affect difficulty!

#### 2SAT/3SAT

```bash
--dataset_name sat \
--data_generation_kwargs "sat_type=2,num_samples=100,min_vars=2,max_vars=8,min_clauses=2,max_clauses=8,seed=42"
```

#### Sudoku

```bash
--dataset_name sudoku \
--data_generation_kwargs "size=4,num_samples=100,min_empty=4,max_empty=8,seed=42"
```

#### Matrix Multiplication

```bash
--dataset_name matmul \
--data_generation_kwargs "size=5,num_samples=100,min_val=-5,max_val=5,seed=42"
```

---

## Solver Cache System

Solver outputs are automatically cached based on a SHA256 hash of all parameters affecting the output. This enables fast iteration when testing different verifier configurations for the same solver. The caching system is automatic, deterministic (up to same hardware configuration), and robust. You can use `--no_load_solver_cache` for force re-running solver, or `--no_save_solver_cache` to not save outputs to cache.

Workflow for efficient experimentation:
```bash
# Step 1: Populate cache with solver outputs (skip verification with --no_verify)
python src/inference.py \
    --no_verify \
    --solver_model_name Qwen/Qwen3-0.6B \
    --dataset_name gsm \
    --output_dir result

# Step 2: Test different verifiers quickly (loads from solver cache automatically)
python src/inference.py \
    --solver_model_name Qwen/Qwen3-0.6B \
    --verifier_model_name Qwen/Qwen3-1.7B \
    --dataset_name gsm \
    --output_dir result
```

---

## Citations

If you have any questions or find any bugs, please feel free to contact Jack Lu (yl11330@nyu.edu). If you found our work helpful, please consider giving us a ⭐ and citing us!
```bibtex
@inproceedings{lu2026llmverification,
    title={When Does Verification Pay Off? A Closer Look at LLMs as Solution Verifiers},
    author={Jack Lu and Ryan Teehan and Jinran Jin and Mengye Ren},
    booktitle={ICLR 2026 Workshop on AI with Recursive Self-Improvement},
    year={2026}
}
```
