from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping


def ensure_parent_dir(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def setup_logger(log_path: Path) -> logging.Logger:
	ensure_parent_dir(log_path)

	logger = logging.getLogger("breast_cancer_training")
	logger.setLevel(logging.INFO)
	logger.propagate = False

	if not logger.handlers:
		formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

		file_handler = logging.FileHandler(log_path, encoding="utf-8")
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(formatter)
		logger.addHandler(stream_handler)

	return logger


def append_csv_row(csv_path: Path, fieldnames: list[str], row: Mapping[str, Any]) -> None:
	ensure_parent_dir(csv_path)
	file_exists = csv_path.exists()

	with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		if not file_exists:
			writer.writeheader()
		writer.writerow(row)


def save_json(path: Path, payload: Any) -> None:
	ensure_parent_dir(path)
	with path.open("w", encoding="utf-8") as json_file:
		json.dump(payload, json_file, indent=2)
