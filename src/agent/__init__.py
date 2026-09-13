"""LangGraph Agent 包：AI 数据分析师智能体的核心装配层。

本包负责把「任务理解 → Skill 选择 → 数据画像 → 规划 → 工具调用循环 →
洞察提炼 → 报告生成」整条工作流组装成一张可编译执行的 LangGraph 图。

对外只暴露两个入口：
- build_graph：构建并编译 Agent 工作流图（模块导入时即生成单例 graph）；
- make_initial_state：根据用户数据文件路径与分析需求构造图的初始状态。
"""

# 从 graph 模块引入图构建器与初始状态工厂，作为包的统一对外 API
from src.agent.graph import build_graph, make_initial_state

# 声明包的公开符号，支持 `from src.agent import *`，也起到接口文档作用
__all__ = ["build_graph", "make_initial_state"]
