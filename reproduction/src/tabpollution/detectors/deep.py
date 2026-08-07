"""Tiny, paper-aligned character Transformer runthrough implementations.

These are deliberately CPU-sized engineering runthroughs, not reported-paper
hyperparameter reproductions.  The datum-wise mode preserves column-permutation
invariance by omitting row-level positional encodings.
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

from .base import Detector, feature_frame, normalize_scalar, serialize_record


CHARS = ["<PAD>", "<UNK>"] + list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:|<>_-. ?/+=,%")
CHAR_TO_ID = {c: i for i, c in enumerate(CHARS)}


def _encode(text: str, length: int) -> tuple[list[int], bool]:
    ids = [CHAR_TO_ID.get(c, 1) for c in text]
    truncated = len(ids) > length
    return (ids[:length] + [0] * length)[:length], truncated


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
        denom = (~mask).sum(1).clamp_min(1).unsqueeze(1)
        pooled = (h * (~mask).unsqueeze(2)).sum(1) / denom
        return self.head(pooled).squeeze(1), pooled


class DatumNet(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, max_datum: int,
                 table_adaptation: bool = False, positional_columns: bool = False,
                 max_columns: int = 24):
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
        self.table_head = nn.Linear(dim, 8)

    def forward(self, ids: torch.Tensor, grl_weight: float = 0.) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, l = ids.shape
        x = self.emb(ids) + self.local_pos[:, :, :l]
        x = x.reshape(b * c, l, -1)
        mask = ids.reshape(b * c, l).eq(0)
        h = self.datum_encoder(x, src_key_padding_mask=mask)
        denom = (~mask).sum(1).clamp_min(1).unsqueeze(1)
        datum = (h * (~mask).unsqueeze(2)).sum(1) / denom
        datum = datum.reshape(b, c, -1)
        target = self.cls_target.expand(b, -1, -1)
        row = torch.cat([target, datum], dim=1)
        if self.column_pos is not None:
            row = row + self.column_pos[:, :row.shape[1]]
        out = self.row_encoder(row)
        pooled = out[:, 0]
        detection = self.detect_head(pooled).squeeze(1)
        table = self.table_head(GradReverse.apply(pooled, grl_weight))
        return detection, table, pooled


class DeepTextDetector(Detector):
    """Modes: flat, table, datum, datum_ta."""

    def __init__(self, mode: str = "flat", seed: int = 2026, dim: int = 24,
                 heads: int = 4, layers: int = 1, max_len: int = 192,
                 max_datum: int = 32, max_columns: int = 24, epochs: int = 2,
                 batch_size: int = 32):
        if mode not in {"flat", "table", "datum", "datum_ta"}:
            raise ValueError(mode)
        self.mode, self.seed = mode, seed
        self.config = dict(dim=dim, heads=heads, layers=layers, max_len=max_len,
                           max_datum=max_datum, max_columns=max_columns, epochs=epochs,
                           batch_size=batch_size)
        torch.manual_seed(seed); np.random.seed(seed)
        self.model = (FlatNet(dim, heads, layers, max_len) if mode == "flat" else
                      DatumNet(dim, heads, layers, max_datum,
                               table_adaptation=mode == "datum_ta",
                               positional_columns=mode == "table", max_columns=max_columns))
        self.loss_history: list[dict[str, float]] = []
        self.truncation_rate = 0.
        self.columns_seen: list[int] = []

    def _tensorize(self, records: pd.DataFrame) -> torch.Tensor:
        frame = feature_frame(records)
        if self.mode == "flat":
            encoded, trunc = zip(*[_encode(serialize_record(row), self.config["max_len"])
                                   for _, row in frame.iterrows()])
            self.truncation_rate = float(np.mean(trunc))
            return torch.tensor(encoded, dtype=torch.long)
        rows, truncs = [], []
        for _, row in frame.iterrows():
            cells = [f"{c}:{normalize_scalar(row[c])}" for c in row.index][:self.config["max_columns"]]
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
                weight = .5 * (1 - np.cos(np.pi * step / total_steps)) if self.mode == "datum_ta" else 0.
                optimizer.zero_grad()
                if self.mode == "flat":
                    logits, _ = self.model(xb); table_logits = None
                else:
                    logits, table_logits, _ = self.model(xb, float(weight))
                det_loss = bce(logits, yb)
                table_loss = ce(table_logits, tb) if self.mode == "datum_ta" else torch.tensor(0.)
                loss = det_loss + (float(weight) * table_loss if self.mode == "datum_ta" else 0.)
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
        obj = cls(state["mode"], state["seed"], **state["config"])
        obj.model.load_state_dict(state["state_dict"]); obj.loss_history = state["loss_history"]
        obj.truncation_rate = state["truncation_rate"]
        return obj

    def get_provenance(self) -> dict[str, Any]:
        return {"implementation": "paper_aligned_self_implementation", "mode": self.mode,
                "paper_performance_reproduction": False, "cpu_tiny_runthrough": True,
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
