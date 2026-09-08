"""Python 沙箱工具测试。"""

from __future__ import annotations

import pytest

from src.tools.python_tool import (
    SandboxSecurityError,
    execute_python,
    run_python,
    validate_code,
)


def test_validate_allows_pandas() -> None:
    # 不应抛异常
    validate_code("import pandas as pd\nimport numpy as np\nprint(df['sales'].mean())")


def test_validate_rejects_os_import() -> None:
    with pytest.raises(SandboxSecurityError):
        validate_code("import os")


def test_validate_rejects_subprocess() -> None:
    with pytest.raises(SandboxSecurityError):
        validate_code("import subprocess")


def test_validate_rejects_system_call() -> None:
    with pytest.raises(SandboxSecurityError):
        validate_code("import os\nos.system('dir')")


def test_validate_rejects_open() -> None:
    with pytest.raises(SandboxSecurityError):
        validate_code("open('C:/secret.txt')")


def test_validate_rejects_eval() -> None:
    with pytest.raises(SandboxSecurityError):
        validate_code("eval('1+1')")


def test_validate_rejects_dunder() -> None:
    with pytest.raises(SandboxSecurityError):
        validate_code("x = ''.__class__")


def test_validate_rejects_pandas_file_io() -> None:
    """pandas 文件读写应被拦截，防止访问 workspace 之外的文件。"""
    for code in (
        "pd.read_csv('C:/secret.csv')",
        "pd.read_excel('C:/x.xlsx')",
        "df.to_csv('C:/out.csv')",
        "pd.read_json('C:/x.json')",
    ):
        with pytest.raises(SandboxSecurityError):
            validate_code(code)


def test_validate_allows_pandas_transform() -> None:
    """pandas 数据分析（非文件 IO）仍应放行。"""
    validate_code("df.groupby('region')['sales'].sum()")
    validate_code("print(df['sales'].mean())")


def test_validate_rejects_empty() -> None:
    with pytest.raises(SandboxSecurityError):
        validate_code("")


def test_run_python_executes() -> None:
    out = run_python("print(df['sales'].mean())", "datasets/sales.csv")
    assert out.strip()
    assert "拒绝" not in out


def test_run_python_blocks_dangerous() -> None:
    # 底层函数对危险代码直接抛 SandboxSecurityError
    with pytest.raises(SandboxSecurityError):
        run_python("import os; os.system('dir')", "datasets/sales.csv")


def test_execute_python_tool_returns_rejection() -> None:
    # Tool 层将拒绝转为友好提示，保证不中断 Agent 链路
    out = execute_python.invoke({"code": "import os", "dataset_path": "datasets/sales.csv"})
    assert "拒绝" in out


def test_run_python_reads_utf8_bom(tmp_path) -> None:
    """沙箱 execute_python 读取 UTF-8 BOM CSV，中文应正确（不出现乱码）。"""
    import pandas as pd

    csv_path = tmp_path / "bom.csv"
    pd.DataFrame({"地区": ["华东"], "销售额": [100]}).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    out = run_python("print(df['地区'].tolist())", str(csv_path))
    assert "华东" in out
