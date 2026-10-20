"""Read the capstone's local CSV data without coercing identifiers."""

from __future__ import annotations

import csv
import os
from pathlib import Path


_TABLES = {
    "customers",
    "holidays",
    "order_items",
    "orders",
    "pincodes",
    "products",
    "support_logs",
}


def find_dataset_root(path: Path | str | None = None) -> Path:
    """Resolve an explicit path, the environment setting, or the repo default."""
    if path is not None:
        root = Path(path).expanduser()
    elif os.environ.get("ZENTROMART_DATASET_DIR"):
        root = Path(os.environ["ZENTROMART_DATASET_DIR"]).expanduser()
    else:
        root = Path(__file__).resolve().parents[2] / "Capstone_Project_Dataset_FTTT_2026"

    root = root.resolve()
    if not (root / "DATA_DICTIONARY.md").is_file():
        raise FileNotFoundError(
            f"Dataset not found at {root}; set ZENTROMART_DATASET_DIR "
            "or place the extracted capstone folder at the repository root."
        )
    return root


class Dataset:
    """Lazy CSV access that preserves IDs, pincodes, blanks, and list fields."""

    def __init__(self, root: Path | str | None = None):
        self.root = find_dataset_root(root)
        self._cache: dict[str, list[dict[str, str]]] = {}

    def table(self, name: str) -> list[dict[str, str]]:
        if name not in _TABLES:
            raise ValueError(f"Unknown capstone table: {name}")
        if name not in self._cache:
            path = self.root / "data" / f"{name}.csv"
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self._cache[name] = [
                    {key: value or "" for key, value in row.items() if key}
                    for row in reader
                ]
        # Return fresh mappings so callers cannot corrupt the shared cache.
        return [row.copy() for row in self._cache[name]]

    def product_by_sku(self, sku: str) -> dict[str, str] | None:
        normalized = sku.strip().casefold()
        return next(
            (row for row in self.table("products")
             if row["sku"].strip().casefold() == normalized),
            None,
        )

    def holiday_dates(self) -> set[str]:
        return {row["date"] for row in self.table("holidays") if row["date"]}
