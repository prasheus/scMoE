import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .network import MoE, infer_config_from_state_dict
from .preprocessing import (
    dense_batch_aligned,
    feature_indexer,
    load_features,
    load_input,
    zscore_batch,
)


class Profiler:

    def __init__(
        self,
        test_input,
        pretrain_dir=None,
        norm_type=False,
        use_raw=False,
        batch_size=8192,
        device=None,
        model_file=None,
        feature_file=None,
        config_file="config.json",
    ):
        self.test_input = test_input
        self.pretrain_dir = Path(pretrain_dir) if pretrain_dir is not None else self._default_pretrain_dir()
        self.norm_type = norm_type
        self.use_raw = use_raw
        self.batch_size = int(batch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_file = model_file
        self.feature_file = feature_file
        self.config_file = config_file

        self.adata = None
        self.model = None
        self.features = None
        self.train_mean = None
        self.train_std = None
        self.temperature = 1.0
        self.threshold = 0.5
        self.config = {}
        self.missing_features = []
        self.fitted = False

    def load(self):
        self.adata = load_input(self.test_input, use_raw=self.use_raw, norm_type=self.norm_type)
        self._load_artifacts()
        self.fitted = True
        return self

    @staticmethod
    def _default_pretrain_dir():
        package_dir = Path(__file__).resolve().parent
        candidates = (
            package_dir / "model",
            package_dir.parent / "model",
            Path.cwd() / "model",
        )
        for path in candidates:
            if path.exists():
                return path
        return candidates[1]

    def profile(self):
        if not self.fitted:
            raise RuntimeError("Call load() before profile().")

        target_idx, source_idx, missing = feature_indexer(self.adata.var_names, self.features)
        self.missing_features = missing
        self._report_feature_overlap(len(missing))

        probs = []
        gates = []

        self.model.eval()
        n_cells = self.adata.n_obs

        with torch.no_grad():
            for start in range(0, n_cells, self.batch_size):
                end = min(start + self.batch_size, n_cells)
                batch_indices = np.arange(start, end)
                x = dense_batch_aligned(
                    self.adata,
                    batch_indices,
                    target_idx,
                    source_idx,
                    len(self.features),
                )
                x = zscore_batch(x, self.train_mean, self.train_std)
                x_tensor = torch.from_numpy(x).to(self.device)

                logits, gate_weights, _ = self.model(x_tensor)
                logits = logits / float(self.temperature)
                probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
                gates.append(gate_weights.cpu().numpy())

        probs = np.concatenate(probs)
        gates = np.concatenate(gates, axis=0)
        pred = probs >= float(self.threshold)

        self.adata.obs["malignancy_call"] = pd.Categorical(
            np.where(pred, "Malignant", "Normal"),
            categories=["Normal", "Malignant"],
        )
        self.adata.obs["malignancy_score"] = probs
        self.adata.obs["primary_expert"] = gates.argmax(axis=1).astype(int)
        self.adata.obs["gate_entropy"] = -np.sum(
            gates * np.log(np.clip(gates, 1e-8, 1.0)),
            axis=1,
        )

        for i in range(gates.shape[1]):
            self.adata.obs[f"expert_weight_{i}"] = gates[:, i]

        return self.adata

    def predict(self):
        return self.profile()

    def _load_artifacts(self):
        model_path = self._resolve_file(
            self.model_file,
            candidates=("moe.pt",),
            required=True,
        )
        feature_path = self._resolve_file(
            self.feature_file,
            candidates=("geneorder.tsv",),
            required=True,
        )
        config_path = self._resolve_file(self.config_file, candidates=(self.config_file,), required=False)

        self.features = load_features(feature_path)

        checkpoint = self._torch_load(model_path)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        self.config = infer_config_from_state_dict(state_dict)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
            checkpoint_config = checkpoint["config"]
            self.config.update({k: v for k, v in checkpoint_config.items() if k in self.config})
            self.temperature = float(checkpoint_config.get("temperature", self.temperature))
            self.threshold = float(checkpoint_config.get("threshold", self.threshold))

        if config_path is not None:
            with open(config_path) as f:
                file_config = json.load(f)
            self.config.update({k: v for k, v in file_config.items() if k in self.config})
            self.temperature = float(file_config.get("temperature", self.temperature))
            self.threshold = float(file_config.get("threshold", self.threshold))

        if isinstance(checkpoint, dict):
            self.temperature = float(checkpoint.get("temperature", self.temperature))
            self.threshold = float(checkpoint.get("threshold", self.threshold))

        self._load_normalization(checkpoint)

        self.model = MoE(**self.config).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if len(self.features) != int(self.config["input_dim"]):
            raise ValueError(
                f"Feature count ({len(self.features)}) does not match model input_dim "
                f"({self.config['input_dim']})."
            )

    def _load_normalization(self, checkpoint):
        mean_path = self.pretrain_dir / "train_mean.npy"
        std_path = self.pretrain_dir / "train_std.npy"

        if mean_path.exists() and std_path.exists():
            self.train_mean = np.load(mean_path).astype(np.float32)
            self.train_std = np.load(std_path).astype(np.float32)
        elif isinstance(checkpoint, dict) and "train_mean" in checkpoint and "train_std" in checkpoint:
            self.train_mean = np.asarray(checkpoint["train_mean"], dtype=np.float32)
            self.train_std = np.asarray(checkpoint["train_std"], dtype=np.float32)
        else:
            raise FileNotFoundError(
                "Missing train_mean/train_std. Provide train_mean.npy and train_std.npy "
            )

        self.train_std = self.train_std.copy()
        self.train_std[self.train_std < 1e-6] = 1.0

    def _resolve_file(self, explicit, candidates, required):
        if explicit is not None:
            path = self.pretrain_dir / explicit
            if not path.exists() and Path(explicit).exists():
                path = Path(explicit)
            if path.exists():
                return path
            if required:
                raise FileNotFoundError(f"Required artifact not found: {explicit}")
            return None

        for name in candidates:
            path = self.pretrain_dir / name
            if path.exists():
                return path

        if required:
            raise FileNotFoundError(
                f"None of these required artifacts were found in {self.pretrain_dir}: {candidates}"
            )
        return None

    def _torch_load(self, model_path):
        try:
            return torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(model_path, map_location=self.device)

    def _report_feature_overlap(self, missing_count):
        total = len(self.features)
        missing_pct = 100.0 * missing_count / max(total, 1)
        print(f"Model features: {total}")
        print(f"Missing features: {missing_count} ({missing_pct:.2f}%)")
        if missing_pct > 20:
            print("Warning: more than 20% of model features are missing in the input data.")
