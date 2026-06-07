"""Data preparation helpers for the Pascucci model calibration pipeline."""

from __future__ import annotations

import csv
import re
import zipfile
from pathlib import Path
from typing import List
from xml.etree import ElementTree as ET

import numpy as np


PRICE_COLUMN = "€/MWh"
CONSUMPTION_COLUMN = "Consumo (W)"
PRODUCTION_COLUMN = "Produzione (W)"


def _validate_block_size(n: int) -> int:
    n_int = int(n)
    if n_int <= 0:
        raise ValueError("n must be a positive integer")
    return n_int


def _parse_float(value: object, *, column: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{column} must contain numeric values")
    text = text.replace(",", ".")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"{column} must contain numeric values, found {value!r}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{column} must contain finite numeric values")
    return parsed


def _hourly_mean(values: np.ndarray, *, n: int, mul_factor: float) -> np.ndarray:
    """Average complete blocks and intentionally discard an incomplete tail."""

    n_hours = int(values.shape[0]) // int(n)
    trimmed = values[: n_hours * int(n)]
    if n_hours == 0:
        return np.asarray([], dtype=np.float64)
    return trimmed.reshape(n_hours, int(n)).mean(axis=1).astype(np.float64) * float(mul_factor)


def prepare_H(filepath: str, n: int = 1, mul_factor: float = 1.0) -> np.ndarray:
    """Return hourly mean net power ``Consumo (W) - Produzione (W)``."""

    n_int = _validate_block_size(n)
    path = Path(filepath)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [name for name in (CONSUMPTION_COLUMN, PRODUCTION_COLUMN) if name not in fieldnames]
        if missing:
            raise ValueError(f"prepare_H missing required columns: {', '.join(missing)}")

        values = [
            _parse_float(row[CONSUMPTION_COLUMN], column=CONSUMPTION_COLUMN)
            - _parse_float(row[PRODUCTION_COLUMN], column=PRODUCTION_COLUMN)
            for row in reader
        ]
    return _hourly_mean(np.asarray(values, dtype=np.float64), n=n_int, mul_factor=mul_factor)


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _cell_column_index(ref: str) -> int:
    match = re.match(r"([A-Z]+)", str(ref))
    if match is None:
        raise ValueError(f"Invalid XLSX cell reference {ref!r}")
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _element_text(element: ET.Element) -> str:
    return "".join(element.itertext())


def _read_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    try:
        data = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(data)
    strings = []
    for item in root.iter():
        if _strip_namespace(item.tag) == "si":
            strings.append(_element_text(item))
    return strings


def _first_worksheet_name(archive: zipfile.ZipFile) -> str:
    names = archive.namelist()
    if "xl/worksheets/sheet1.xml" in names:
        return "xl/worksheets/sheet1.xml"
    candidates = sorted(name for name in names if name.startswith("xl/worksheets/") and name.endswith(".xml"))
    if not candidates:
        raise ValueError("XLSX file does not contain worksheets")
    return candidates[0]


def _cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        for child in cell:
            if _strip_namespace(child.tag) == "is":
                return _element_text(child)
        return ""

    value_node = None
    for child in cell:
        if _strip_namespace(child.tag) == "v":
            value_node = child
            break
    if value_node is None or value_node.text is None:
        return ""

    raw_value = value_node.text
    if cell_type == "s":
        index = int(raw_value)
        if index < 0 or index >= len(shared_strings):
            raise ValueError(f"Invalid shared string index {index}")
        return shared_strings[index]
    return raw_value


def _read_xlsx_rows(filepath: str) -> List[List[str]]:
    with zipfile.ZipFile(filepath) as archive:
        shared_strings = _read_shared_strings(archive)
        worksheet_name = _first_worksheet_name(archive)
        root = ET.fromstring(archive.read(worksheet_name))

    rows: List[List[str]] = []
    for row in root.iter():
        if _strip_namespace(row.tag) != "row":
            continue
        values: List[str] = []
        for cell in row:
            if _strip_namespace(cell.tag) != "c":
                continue
            ref = cell.attrib.get("r", "")
            col_idx = _cell_column_index(ref) if ref else len(values)
            while len(values) <= col_idx:
                values.append("")
            values[col_idx] = _cell_value(cell, shared_strings)
        rows.append(values)
    return rows


def prepare_S(filepath: str, n: int = 1, mul_factor: float = 1.0) -> np.ndarray:
    """Return hourly mean linear prices from the ``€/MWh`` XLSX column."""

    n_int = _validate_block_size(n)
    rows = _read_xlsx_rows(filepath)
    if not rows:
        raise ValueError("prepare_S requires a header row")
    headers = [str(value).strip() for value in rows[0]]
    if PRICE_COLUMN not in headers:
        raise ValueError(f"prepare_S missing required column: {PRICE_COLUMN}")
    price_idx = headers.index(PRICE_COLUMN)
    prices = []
    for row in rows[1:]:
        value = row[price_idx] if price_idx < len(row) else ""
        prices.append(_parse_float(value, column=PRICE_COLUMN))
    return _hourly_mean(np.asarray(prices, dtype=np.float64), n=n_int, mul_factor=mul_factor)