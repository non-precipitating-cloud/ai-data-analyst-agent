"""文件读取与数据集概览工具测试（src.tools.file_tools）。

覆盖：
- load_dataframe：CSV 加载为 pandas DataFrame，缺失文件抛 FileNotFoundError；
- _read_dataset：行数 / 列数 / 列名 / 前 5 行预览；
- _inspect_schema：列清单与 dtype 类型；
- _profile_dataset：数值列、统计量与缺失值画像；
- UTF-8 BOM 编码及中文内容的正确读取（含真实 sales.csv）。
"""

from __future__ import annotations

import pandas as pd

from src.config.settings import get_settings
from tests.conftest import SALES_CSV
from src.tools.file_tools import (
    _inspect_schema,
    _profile_dataset,
    _read_dataset,
    load_dataframe,
)


def test_load_dataframe() -> None:
    """样例 CSV 应被加载为非空的 pandas DataFrame。"""
    df = load_dataframe(SALES_CSV)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_read_dataset() -> None:
    """数据集摘要应准确返回 2160 行、12 列，包含 sales 列且预览恰好 5 行。"""
    r = _read_dataset(SALES_CSV)
    assert r["num_rows"] == 2160
    assert r["num_columns"] == 12
    assert "sales" in r["columns"]
    assert len(r["head"]) == 5


def test_inspect_schema() -> None:
    """结构检查结果应包含关键业务列，并提供 dtypes 类型信息。"""
    r = _inspect_schema(SALES_CSV)
    assert set(r["columns"]) >= {"sales", "region", "product", "category"}
    assert "dtypes" in r


def test_profile_dataset() -> None:
    """数据画像应给出数值列清单、sales 均值等统计量以及缺失值情况。"""
    p = _profile_dataset(SALES_CSV)
    assert p["num_rows"] == 2160
    assert "sales" in p["numeric_columns"]
    assert p["numeric_stats"]["sales"]["mean"] > 0
    assert "missing_values" in p


def test_missing_file() -> None:
    """读取不存在的文件应抛 FileNotFoundError，而不是静默返回空数据。"""
    import pytest

    with pytest.raises(FileNotFoundError):
        load_dataframe("datasets/does_not_exist.csv")


def test_load_dataframe_utf8_bom(tmp_path, monkeypatch) -> None:
    """UTF-8 BOM CSV 中的中文应被正确读取（不出现乱码）。"""
    # 把临时目录声明为额外数据根（模拟 Docker 挂载外部数据卷的合法场景）
    monkeypatch.setattr(get_settings(), "extra_data_dirs", [str(tmp_path)])
    csv_path = tmp_path / "bom.csv"
    # 用 utf-8-sig 编码写出带 BOM 的中文测试数据
    pd.DataFrame({"地区": ["华东", "华南"], "销售额": [100, 200]}).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    # 确认文件头带 BOM
    assert csv_path.read_bytes()[:3] == b"\xef\xbb\xbf"

    loaded = load_dataframe(csv_path)
    assert list(loaded["地区"]) == ["华东", "华南"]


def test_resolve_dataset_path_blocks_traversal() -> None:
    """路径穿越/任意文件读取必须被拦截：.env、系统文件不允许作为数据集读取。"""
    import pytest

    s = get_settings()
    # 直接指向上层目录中的 .env（内含密钥），应抛 PermissionError 而非放行
    with pytest.raises(PermissionError):
        s.resolve_dataset_path(".env")
    # 相对路径穿越尝试同样要被规范化后拦在数据目录之外
    with pytest.raises(PermissionError):
        s.resolve_dataset_path("datasets/../.env")
    # 数据目录内的合法文件不受影响
    assert s.resolve_dataset_path("datasets/sales.csv").name == "sales.csv"


def test_sales_csv_chinese_readable() -> None:
    """真实的 sales.csv（UTF-8 BOM）中文字段应正确读取。"""
    df = load_dataframe(SALES_CSV)
    assert "华东" in set(df["region"].unique())
    assert "电子产品" in set(df["category"].unique())
