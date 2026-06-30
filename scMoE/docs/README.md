## Preprocessing

gene expression at input format: 

- Raw UMI counts: CPM normalization with target sum `1e6`, then `log1p`.
- TPM values: `log1p` only.
- leave data if it already matches one of the two formats above.

Do not recompute z-score mean/std on validation, test, or
external data. (done with the model internally)


```python
from scMoE import Profiler

profiler = Profiler(
    "query.h5ad",
    pretrain_dir="scMoE/model",
    norm_type="cpm_log1p",  # raw UMI counts
)

adata = profiler.load().profile()
```

For TPM input:

```python
profiler = Profiler("query.h5ad", pretrain_dir="scMoE/model", norm_type="tpm_log1p")
adata = profiler.load().profile()
```

For data that is already normalized exactly like training:

```python
profiler = Profiler("query.h5ad", pretrain_dir="scMoE/model", norm_type=False)
adata = profiler.load().profile()
```

The output is added to `adata.obs`.
