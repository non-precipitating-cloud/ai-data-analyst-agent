"""文件工具测试。"""

from __future__ import annotations

import pandas as pd

from tests.conftest import SALES_CSV
from src.tools.file_tools import (
    _inspect_schema,
    _profile_dataset,
    _read_dataset,
    load_dataframe,
)


def test_load_dataframe() -> None:
    df = load_dataframe(SALES_CSV)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_read_dataset() -> None:
    r = _read_dataset(SALES_CSV)
    assert r["num_rows"] == 2160
    assert r["num_columns"] == 12
    assert "sales" in r["columns"]
    assert len(r["head"]) == 5


def test_inspect_schema() -> None:
    r = _inspect_schema(SALES_CSV)
    assert set(r["columns"]) >= {"sales", "region", "product", "category"}
    assert "dtypes" in r


def test_profile_dataset() -> None:
    p = _profile_dataset(SALES_CSV)
    assert p["num_rows"] == 2160
    assert "sales" in p["numeric_columns"]
    assert p["numeric_stats"]["sales"]["mean"] > 0
    assert "missing_values" in p


def test_missing_file() -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        load_dataframe("datasets/does_not_exist.csv")


def test_load_dataframe_utf8_bom(tmp_path) -> None:
    """UTF-8 BOM CSV 中的中文应被正确读取（不出现乱码）。"""
    csv_path = tmp_path / "bom.csv"
    pd.DataFrame({"地区": ["华东", "华南"], "销售额": [100, 200]}).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    # 确认文件头带 BOM
    assert csv_path.read_bytes()[:3] == b"\xef\xbb\xbf"

    loaded = load_dataframe(csv_path)
    assert list(loaded["地区"]) == ["华东", "华南"]


def test_sales_csv_chinese_readable() -> None:
    """真实的 sales.csv（UTF-8 BOM）中文字段应正确读取。"""
    df = load_dataframe(SALES_CSV)
    assert "华东" in set(df["region"].unique())
    assert "电子产品" in set(df["category"].unique())
