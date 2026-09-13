"""AI Data Analyst Agent 应用入口（薄启动层）。

本文件不含业务逻辑，只负责启动 src.cli.runner 中的命令行交互循环，
让入口与业务解耦，既方便直接执行，也便于被测试或其他模块导入复用。

用法：
    python main.py
"""

# 导入 CLI 运行器：交互式分析会话的真正入口函数
from src.cli.runner import main

# 仅在被直接执行（python main.py）时启动；被 import 时不自动运行
if __name__ == "__main__":
    main()
