"""LLM 子包：对外暴露大模型实例的统一工厂入口。

业务代码通过 ``from src.llm import get_llm`` 获取 ChatModel，
不直接依赖具体后端实现（当前为 OpenAI 兼容接口），
便于以后扩展其他模型供应商而不影响调用方。
"""

# 工厂函数：根据全局配置构造并返回 LangChain ChatModel
from src.llm.factory import get_llm

# 公开 API 白名单
__all__ = ["get_llm"]
