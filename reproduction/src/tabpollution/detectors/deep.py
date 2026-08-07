"""Paper-aligned character Transformer detectors for schema-variable tables.

The formal modes implement the published Flat Text and Datum-wise design
principles: explicit CLS pooling, character-level encoding, local positional
encoding inside each datum, no row-level positional encoding for Datum-wise,
and adversarial table-identity regularization for Datum-wise + TA.  Small
dimensions are retained only when a smoke configuration explicitly requests
them and are marked as non-formal in provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import Detector, feature_frame, normalize_scalar, record_feature_items, serialize_record


CHARS = ["<PAD>", "<UNK>"] + list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:|<>_-. ?/+=,%") + ["<CLS>"]
CHAR_TO_ID = {c: i for i, c in enumerate(CHARS)}
CLS_ID = CHAR_TO_ID["<CLS>"]


def _encode(text: str, length: int, *, add_cls: bool = True) -> tuple[list[int], bool]:
    ids = [CHAR_TO_ID.get(c, 1) for c in text]
    content_length = length - int(add_cls)
    truncated = len(ids) > content_length
    encoded = ([CLS_ID] if add_cls else []) + ids[:content_length]
    return (encoded + [0] * length)[:length], truncated


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.weight * grad, None


class FlatNet(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, max_len: int):
        super().__init__()
        self.emb = nn.Embedding(len(CHARS), dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 2, .1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(dim, 1)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.emb(ids) + self.pos[:, :ids.shape[1]]
        mask = ids.eq(0)
        h = self.encoder(x, src_key_padding_mask=mask)
        pooled = h[:, 0]
        return self.head(pooled).squeeze(1), pooled


class DatumNet(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, max_datum: int,
                 table_adaptation: bool = False, positional_columns: bool = False,
                 max_columns: int = 24, table_classes: int = 8):
        super().__init__()
        self.table_adaptation = table_adaptation
        self.emb = nn.Embedding(len(CHARS), dim, padding_idx=0)
        self.local_pos = nn.Parameter(torch.zeros(1, 1, max_datum, dim))
        self.cls_datum = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_target = nn.Parameter(torch.zeros(1, 1, dim))
        self.column_pos = nn.Parameter(torch.zeros(1, max_columns + 1, dim)) if positional_columns else None
        datum_layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, .1, batch_first=True)
        row_layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, .1, batch_first=True)
        self.datum_encoder = nn.TransformerEncoder(datum_layer, 1)
        self.row_encoder = nn.TransformerEncoder(row_layer, layers)
        self.detect_head = nn.Linear(dim, 1)
        self.table_head = nn.Linear(dim, table_classes)

    def forward(self, ids: torch.Tensor, grl_weight: float = 0.) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, l = ids.shape
        present = ids.ne(0).any(dim=2)
        safe_ids = ids.clone()
        missing_batch, missing_column = (~present).nonzero(as_tuple=True)
        safe_ids[missing_batch, missing_column, 0] = CLS_ID
        x = self.emb(safe_ids) + self.local_pos[:, :, :l]
        x = x.reshape(b * c, l, -1)
        mask = safe_ids.reshape(b * c, l).eq(0)
        h = self.datum_encoder(x, src_key_padding_mask=mask)
        datum = h[:, 0]
        datum = datum.reshape(b, c, -1)
        target = self.cls_target.expand(b, -1, -1)
        row = torch.cat([target, datum], dim=1)
        if self.column_pos is not None:
            row = row + self.column_pos[:, :row.shape[1]]
        row_mask = torch.cat([
            torch.zeros((b, 1), dtype=torch.bool, device=ids.device),
            ~present,
        ], dim=1)
        out = self.row_encoder(row, src_key_padding_mask=row_mask)
        pooled = out[:, 0]
        detection = self.detect_head(pooled).squeeze(1)
        table = self.table_head(GradReverse.apply(pooled, grl_weight))
        return detection, table, pooled


class DeepTextDetector(Detector):
    """Modes: flat, table, datum, datum_ta."""

    def __init__(self, mode: str = "flat", seed: int = 2026, dim: int = 24,
                 heads: int = 4, layers: int = 1, max_len: int = 192,
                 max_datum: int = 32, max_columns: int = 24, epochs: int = 2,
                 batch_size: int = 32, device: str = "auto", table_classes: int = 8):
        if mode not in {"flat", "table", "datum", "datum_ta"}:
            raise ValueError(mode)
        self.mode, self.seed = mode, seed
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = torch.device("cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu")
        self.config = dict(dim=dim, heads=heads, layers=layers, max_len=max_len,
                           max_datum=max_datum, max_columns=max_columns, epochs=epochs,
                           batch_size=batch_size, device=device, table_classes=table_classes)
        torch.manual_seed(seed); np.random.seed(seed)
        self.model = (FlatNet(dim, heads, layers, max_len) if mode == "flat" else
                      DatumNet(dim, heads, layers, max_datum,
                               table_adaptation=mode == "datum_ta",
                               positional_columns=mode == "table", max_columns=max_columns,
                               table_classes=table_classes)).to(self.device)
        self.loss_history: list[dict[str, float]] = []
        self.truncation_rate = 0.
        self.columns_seen: list[int] = []

    def _tensorize(self, records: pd.DataFrame) -> torch.Tensor:
        if self.mode == "flat":
            encoded, trunc = zip(*[_encode(serialize_record(row), self.config["max_len"])
                                   for _, row in records.iterrows()])
            self.truncation_rate = float(np.mean(trunc))
            return torch.tensor(encoded, dtype=torch.long)
        rows, truncs = [], []
        for _, row in records.iterrows():
            cells = [f"{column}:{normalize_scalar(value)}" for column, value in record_feature_items(row)]
            cells = cells[:self.config["max_columns"]]
            self.columns_seen.append(len(cells))
            enc = [_encode(cell, self.config["max_datum"]) for cell in cells]
            truncs.extend(v[1] for v in enc)
            row_ids = [v[0] for v in enc]
            pad = [[0] * self.config["max_datum"]] * (self.config["max_columns"] - len(row_ids))
            rows.append(row_ids + pad)
        self.truncation_rate = float(np.mean(truncs)) if truncs else 0.
        return torch.tensor(rows, dtype=torch.long)

    def fit(self, train_records: pd.DataFrame, train_labels: np.ndarray,
            val_records: pd.DataFrame | None = None, val_labels: np.ndarray | None = None,
            **context: Any) -> "DeepTextDetector":
        x = self._tensorize(train_records)
        y = torch.tensor(train_labels, dtype=torch.float32)
        table = torch.tensor(np.asarray(context.get("table_labels", np.zeros(len(y)))), dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y, table), batch_size=self.config["batch_size"], shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-3)
        bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
        self.model.train()
        total_steps = max(1, self.config["epochs"] * len(loader)); step = 0
        for epoch in range(self.config["epochs"]):
            det_sum = tab_sum = 0.
            for xb, yb, tb in loader:
                xb, yb, tb = xb.to(self.device), yb.to(self.device), tb.to(self.device)
                weight = .5 * (1 - np.cos(np.pi * step / total_steps)) if self.mode == "datum_ta" else 0.
                optimizer.zero_grad()
                if self.mode == "flat":
                    logits, _ = self.model(xb); table_logits = None
                else:
                    logits, table_logits, _ = self.model(xb, float(weight))
                det_loss = bce(logits, yb)
                table_loss = ce(table_logits, tb) if self.mode == "datum_ta" else torch.tensor(0., device=self.device)
                loss = det_loss + (table_loss if self.mode == "datum_ta" else 0.)
                loss.backward(); optimizer.step(); step += 1
                det_sum += float(det_loss.detach()); tab_sum += float(table_loss.detach())
            self.loss_history.append({"epoch": epoch + 1, "detection_loss": det_sum / len(loader),
                                      "table_loss": tab_sum / len(loader),
                                      "adaptation_weight": float(weight)})
        return self

    def predict_score(self, records: pd.DataFrame, **context: Any) -> np.ndarray:
        x = self._tensorize(records); self.model.eval(); out = []
        with torch.no_grad():
            for (xb,) in DataLoader(TensorDataset(x), batch_size=self.config["batch_size"]):
                xb = xb.to(self.device)
                logits = self.model(xb)[0]
                out.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        return np.asarray(out)

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mode": self.mode, "seed": self.seed, "config": self.config,
                    "state_dict": self.model.state_dict(), "loss_history": self.loss_history,
                    "truncation_rate": self.truncation_rate}, path)

    @classmethod
    def load(cls, path: str | Path) -> "DeepTextDetector":
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["config"].get("device") == "cuda" and not torch.cuda.is_available():
            state["config"]["device"] = "cpu"
        obj = cls(state["mode"], state["seed"], **state["config"])
        obj.model.load_state_dict(state["state_dict"]); obj.loss_history = state["loss_history"]
        obj.truncation_rate = state["truncation_rate"]
        return obj

    def get_provenance(self) -> dict[str, Any]:
        formal_architecture = (
            self.config["dim"] == 192 and self.config["heads"] == 6
            and self.config["layers"] == 6 and self.config["epochs"] >= 20
        )
        return {"implementation": "paper_aligned_self_implementation", "mode": self.mode,
                "formal_architecture": formal_architecture,
                "paper_performance_reproduction": False,
                "device": str(self.device),
                "cpu_tiny_runthrough": self.config["dim"] <= 24 and self.config["epochs"] <= 2,
                "config": self.config, "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "truncation_rate": self.truncation_rate, "loss_history": self.loss_history}

    def permutation_max_delta(self, records: pd.DataFrame, repeats: int = 4) -> float:
        if self.mode not in {"datum", "datum_ta"}: return float("nan")
        base = self.predict_score(records)
        deltas = []
        frame = feature_frame(records)
        for i in range(repeats):
            cols = frame.columns.to_list(); np.random.default_rng(self.seed + i).shuffle(cols)
            deltas.append(np.max(np.abs(base - self.predict_score(frame[cols]))))
        return float(max(deltas))
