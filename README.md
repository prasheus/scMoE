The `scMoE/` folder contains inference code. The `model/` folder contains the trained weights and normalization artifacts.

## Install

From the repository root:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

If the environment already exists:

```bash
source venv/bin/activate
pip install -e .
```

## Input Format

Input can be:

- `.h5ad`
- `.txt` / `.tsv`
- `.csv`
- an `AnnData` object

For text files, rows should be genes and columns should be cells. For `.h5ad`, genes should be in `adata.var_names` and cells in `adata.obs_names`.

## Normalization

To Use the same expression scale expected by the model:

- Already normalized: `norm_type=False`
- Raw UMI counts: `norm_type="cpm_log1p"`
- TPM values: `norm_type="tpm_log1p"`

Do not recompute z-score on new data. model internally applies the train set `train_mean.npy` and `train_std.npy`.

## Inference

For raw UMI count data:

```python
from scMoE import Profiler

profiler = Profiler(
    test_input="query.h5ad",
    pretrain_dir="model",
    norm_type="cpm_log1p",
    batch_size=8192, #default (lower if inference runs OOM)
)

adata = profiler.load().profile()
```

For TPM data:

```python
from scMoE import Profiler

adata = (
    Profiler(
        test_input="query.h5ad",
        pretrain_dir="model",
        norm_type="tpm_log1p",
    )
    .load()
    .profile()
)
```

For data already normalized:

```python
from scMoE import Profiler

adata = (
    Profiler(
        test_input="query.h5ad",
        pretrain_dir="model",
        norm_type=False,
    )
    .load()
    .profile()
)
```

## Output

Predictions are added to `adata.obs`:

```text
malignancy_call      Normal or Malignant
malignancy_score     malignancy probability-like score
primary_expert       expert with the highest gate weight
gate_entropy         uncertainty/spread of gate weights
expert_weight_*      per-expert gate weights
```

View results:

```python
prediction_cols = [
    "malignancy_call",
    "malignancy_score",
    "primary_expert",
    "gate_entropy",
]

expert_cols = [c for c in adata.obs.columns if c.startswith("expert_weight_")]

adata.obs[prediction_cols + expert_cols].head()
```

## Save

Save the full AnnData object:

```python
adata.write_h5ad("predictions.h5ad")
```

Save only the prediction table:

```python
adata.obs[prediction_cols + expert_cols].to_csv("predictions.csv")
```

## Example

```python
from scMoE import Profiler

input_path = "query.h5ad"

adata = (
    Profiler(
        test_input=input_path,
        pretrain_dir="model",
        norm_type="cpm_log1p",
    )
    .load()
    .profile()
)

prediction_cols = [
    "malignancy_call",
    "malignancy_score",
    "primary_expert",
    "gate_entropy",
]
expert_cols = [c for c in adata.obs.columns if c.startswith("expert_weight_")]

print(adata.obs[prediction_cols + expert_cols].head())

adata.write_h5ad("predictions.h5ad")
adata.obs[prediction_cols + expert_cols].to_csv("predictions.csv")
```

## Note

- `geneorder.tsv` controls the model gene order. Missing genes are filled with zero after alignment.