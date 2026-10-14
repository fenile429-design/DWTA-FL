"""
Checkpoint container for DWTA-FL (bestmodel.pt).

Holds, in a single file:
  - the best DWTA-FL model weights (+ metadata of the run that produced them)
  - every measured manuscript / supplementary table
  - every figure specification (series of x/y values)

The file is written with pickle (the .pt extension is kept as requested), so
test.py can render everything without any value being stored in code.
"""

import os
import pickle
import tempfile
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Checkpoint:
    def __init__(self, path: str = config.CHECKPOINT_PATH):
        self.path = path
        self.data: Dict[str, Any] = {
            "model": None,
            "tables": {},
            "figures": {},
            "meta": {},
        }

    # ---------------- model -----------------
    def update_best_model(
        self,
        weights: List[Any],
        score: float,
        info: Dict[str, Any],
    ) -> bool:
        """Keep the weights with the highest score; returns True if replaced."""
        current = self.data["model"]
        if current is None or score > current["score"]:
            self.data["model"] = {"weights": weights, "score": score, "info": info}
            self.save()
            return True
        return False

    # ---------------- tables / figures -----------------
    def add_table(self, key: str, title: str, columns: List[str], rows: List[List[Any]]):
        self.data["tables"][key] = {"title": title, "columns": columns, "rows": rows}

    def add_figure(self, key: str, spec: Dict[str, Any]):
        self.data["figures"][key] = spec

    def set_meta(self, **kwargs):
        self.data["meta"].update(kwargs)

    # ---------------- io -----------------
    def save(self):
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)

    @staticmethod
    def load(path: str = config.CHECKPOINT_PATH) -> "Checkpoint":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run `python train.py` first to create it."
            )
        ck = Checkpoint(path)
        with open(path, "rb") as f:
            ck.data = pickle.load(f)
        return ck
