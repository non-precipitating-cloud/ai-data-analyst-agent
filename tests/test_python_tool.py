"""Python 沙箱工具测试（src.tools.python_tool）。

分两层验证：
- validate_code：静态安全校验，放行 pandas/numpy 数据分析，拦截 import os/subprocess、
  文件读写、eval、dunder 越权、空代码等危险写法（抛 SandboxSecurityError）；
- run_python / execute_python：真实执行与工具层封装，危险代码在 Tool 层转为友好中文提示，
  并验证沙箱内能正确读取 UTF-8 BOM 的中文 CSV。
"""

from __future__ import annotations

import pytest

from src.tools.python_tool import (
    SandboxSecurityError,
    execute_python,
    run_python,
    validate_code,
)


def test_validate_allows_pandas() -> None:
    """常规 pandas/numpy 导入与数据分析代码应通过校验。"""
    # 不应抛异常
    validate_code("import pandas as pd\nimport numpy as np\nprint(df['sales'].mean())")


def test_validate_rejects_os_import() -> None:
    """import os 可触碰系统能力，必须被拦截。"""
    with pytest.raises(SandboxSecurityError):
        validate_code("import os")


def test_validate_rejects_subprocess() -> None:
    """import subprocess 可拉起任意进程，必须被拦截。"""
    with pytest.raises(SandboxSecurityError):
        validate_code("import subprocess")


def test_validate_rejects_system_call() -> None:
    """os.system 执行系统命令，必须被拦截。"""
    with pytest.raises(SandboxSecurityError):
        validate_code("import os\nos.system('dir')")


def test_validate_rejects_open() -> None:
    """open 任意路径可越权读盘（如 C:/secret.txt），必须被拦截。"""
    with pytest.raises(SandboxSecurityError):
        validate_code("open('C:/secret.txt')")


def test_validate_rejects_eval() -> None:
    """eval 可动态执行任意表达式，必须被拦截。"""
    with pytest.raises(SandboxSecurityError):
        validate_code("eval('1+1')")


def test_validate_rejects_dunder() -> None:
    """借助 __class__ 等双下属性可逃逸沙箱，必须被拦截。"""
    with pytest.raises(SandboxSecurityError):
        validate_code("x = ''.__class__")


def test_validate_rejects_pandas_file_io() -> None:
    """pandas 文件读写应被拦截，防止访问 workspace 之外的文件。"""
    # 覆盖 read_csv/read_excel/to_csv/read_json 四类越权文件 IO
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
    """空代码没有任何分析意义，应被拒绝。"""
    with pytest.raises(SandboxSecurityError):
        validate_code("")


def test_run_python_executes() -> None:
    """合法代码在沙箱内应真实执行并输出 sales 均值，且不出现拒绝提示。"""
    out = run_python("print(df['sales'].mean())", "datasets/sales.csv")
    assert out.strip()
    assert "拒绝" not in out


def test_run_python_blocks_dangerous() -> None:
    """底层 run_python 遇到危险代码应直接抛 SandboxSecurityError。"""
    # 底层函数对危险代码直接抛 SandboxSecurityError
    with pytest.raises(SandboxSecurityError):
        run_python("import os; os.system('dir')", "datasets/sales.csv")


def test_execute_python_tool_returns_rejection() -> None:
    """Tool 层把安全拒绝转成中文友好提示，保证 Agent 链路不被异常打断。"""
    # Tool 层将拒绝转为友好提示，保证不中断 Agent 链路
    out = execute_python.invoke({"code": "import os", "dataset_path": "datasets/sales.csv"})
    assert "拒绝" in out


def test_run_python_reads_utf8_bom(tmp_path, monkeypatch) -> None:
    """沙箱 execute_python 读取 UTF-8 BOM CSV，中文应正确（不出现乱码）。"""
    import pandas as pd

    # 把临时目录声明为额外数据根，路径边界校验才会放行该测试文件
    from src.config.settings import get_settings

    monkeypatch.setattr(get_settings(), "extra_data_dirs", [str(tmp_path)])
    # 构造带 BOM 的中文 CSV
    csv_path = tmp_path / "bom.csv"
    pd.DataFrame({"地区": ["华东"], "销售额": [100]}).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    # 在沙箱内打印中文列内容，输出应包含“华东”
    out = run_python("print(df['地区'].tolist())", str(csv_path))
    assert "华东" in out
