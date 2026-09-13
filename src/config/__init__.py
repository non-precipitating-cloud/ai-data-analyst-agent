"""配置包：对外暴露统一的配置入口。

业务代码只需 ``from src.config import Settings, get_settings``，
无需关心配置具体定义在哪个子模块，便于以后调整内部实现结构。
"""

# 从 settings 子模块再导出：配置类（类型注解用）与单例工厂函数
from src.config.settings import Settings, get_settings

# 显式声明包的公开 API，同时约束 from src.config import * 的导出范围
__all__ = ["Settings", "get_settings"]
