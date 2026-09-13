"""LangGraph 节点包子模块：汇聚工作流中每个节点的实现。

每个节点都是一个「接收 AgentState、返回状态增量字典」的纯函数式组件，
执行顺序由 src/agent/graph.py 中的边决定：
task_understanding → skill_selection → profiler → planner
→ agent ⇄ tools → insight → report。

本文件只负责统一导出，graph.py 通过 `from src.agent.nodes import ...`
即可一次性拿到全部节点函数。
"""

# 以下按工作流阶段导入各节点函数（agent_node 与 tools_node 同属工具调用循环）
from src.agent.nodes.insight import insight_node
from src.agent.nodes.planner import planner_node
from src.agent.nodes.profiler import profiler_node
from src.agent.nodes.report import report_node
from src.agent.nodes.skill_selection import skill_selection_node
from src.agent.nodes.task_understanding import task_understanding_node
from src.agent.nodes.tool_calling import agent_node, tools_node

# 公开节点清单：顺序与图中执行顺序大致对应，兼作节点目录
__all__ = [
    "task_understanding_node",
    "skill_selection_node",
    "profiler_node",
    "planner_node",
    "agent_node",
    "tools_node",
    "insight_node",
    "report_node",
]
