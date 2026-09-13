"""LangGraph 图组装模块：定义节点、条件边与循环，是 Agent 的「总装车间」。

在整体架构中的位置：
- 上接调用入口（CLI / API）：入口通过 make_initial_state 准备输入，
  再执行 build_graph() 编译出的图；
- 下接 src/agent/nodes 下的各个具体节点函数（理解、选 Skill、画像、
  规划、LLM 决策、工具执行、洞察、报告）。

图的主干流程为：
START → task_understanding → skill_selection → profiler → planner
      → agent ⇄ tools（条件循环）→ insight → report → END
"""

from __future__ import annotations

# LangChain 消息类型：SystemMessage 设定智能体人设，HumanMessage 承载用户输入
from langchain_core.messages import HumanMessage, SystemMessage
# LangGraph 图构件：StateGraph 定义状态机，START/END 为虚拟起止节点
from langgraph.graph import END, START, StateGraph

# Agent 的系统级提示词（定义数据分析智能体的工作方式与约束）
from src.agent.prompts import AGENT_SYSTEM
# 逐个引入图中要挂载的节点函数
from src.agent.nodes import (
    agent_node,
    insight_node,
    planner_node,
    profiler_node,
    report_node,
    skill_selection_node,
    task_understanding_node,
    tools_node,
)
# 图的共享状态结构（TypedDict），所有节点都围绕它读写数据
from src.agent.state import AgentState
# 全局配置访问（步数上限等运行参数来自环境/配置文件）
from src.config.settings import get_settings


def make_initial_state(
    dataset_path: str,
    user_request: str,
    conversation_context: str = "",
) -> AgentState:
    """构造图运行所需的初始状态字典。

    在图执行开始前调用一次，把系统提示词、用户的数据文件路径与分析需求
    封装成 messages，并为各阶段产物（画像、计划、工具记录、洞察等）
    初始化空容器，供后续节点通过「累加 reducer」追加内容。

    :param dataset_path: 用户上传的数据集文件路径（如 CSV/Excel）
    :param user_request: 用户用自然语言描述的分析需求
    :param conversation_context: 前几轮对话的摘要上下文（连续追问时注入）；
        为空串表示这是一次全新的、无历史的任务
    :return: 符合 AgentState 结构的初始状态字典
    """
    settings = get_settings()
    # 首轮消息：数据路径 + 分析需求；有历史时把历史摘要一并附上，
    # 使「为什么？」这类追问能直接复用上一次的分析结论
    first_turn = f"数据文件：{dataset_path}\n分析需求：{user_request}"
    if conversation_context:
        first_turn = f"{conversation_context}\n\n{first_turn}"

    return {
        # 对话历史：首条为系统人设，次条为本次任务（含可选的历史上下文）
        "messages": [
            SystemMessage(content=AGENT_SYSTEM),
            HumanMessage(content=first_turn),
        ],
        "user_request": user_request,
        "dataset_path": dataset_path,
        "conversation_context": conversation_context,
        # 数据画像结果（由 profiler 节点填充）
        "dataset_metadata": {},
        # 分析计划（由 planner 节点产出的步骤列表）
        "analysis_plan": [],
        # 命中的 Skill 标识及其格式化后的方法论正文
        "selected_skills": [],
        "skills_context": "",
        # 循环控制：当前已执行的 LLM 决策步数 / 允许的最大步数（防死循环）
        "step_count": 0,
        "max_steps": settings.max_steps,
        # running 表示分析进行中，report 节点完成后置为 done
        "status": "running",
        # 以下均为「累加字段」：节点返回新片段，LangGraph 自动追加
        "observations": [],
        "insights": [],
        "errors": [],
        "tool_calls": [],
        "tool_results": [],
        "llm_calls": [],
        "generated_charts": [],
        "degraded": [],
    }


def route_after_agent(state: AgentState) -> str:
    """条件路由：决定 agent 节点（LLM 决策）之后的走向。

    三条终止路径，任一满足即结束工具循环：
    1. LLM 不再请求工具（正常收敛）；
    2. 达到 max_steps 步数上限（硬性防死循环）；
    3. 本轮请求的工具调用**全部**是已成功执行过的重复调用——再执行一遍
       不会产生任何新信息，继续循环只会白白烧掉步数与 token。

    :param state: 图当前共享状态
    :return: "tools" 表示继续执行工具调用；"insight" 表示结束循环、进入洞察
    """
    # 取对话中最后一条消息（即 LLM 刚做出的决策）；无消息时置空
    last = state.get("messages", [None])[-1] if state.get("messages") else None
    # tool_calls 非空表示 LLM 本轮想调用工具而非输出最终结论
    tool_calls = getattr(last, "tool_calls", None) or []
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 15)

    if not tool_calls:
        return "insight"
    if step_count >= max_steps:
        return "insight"
    if _no_progress(state, tool_calls):
        # 无进展：提前收敛，由 insight 节点说明「未能取得新数据」
        return "insight"
    return "tools"


def _no_progress(state: AgentState, tool_calls: list) -> bool:
    """判断本轮的工具调用是否全部为「已成功执行过的重复调用」。

    :param state: 图当前共享状态（读其中的 tool_calls 台账）
    :param tool_calls: LLM 本轮请求的工具调用列表
    :return: 全部重复（无新信息可获取）返回 True
    """
    # 延迟导入：避免 graph 与 nodes 之间在模块加载期形成循环依赖
    from src.agent.nodes.tool_calling import call_signature, executed_signatures

    done = executed_signatures(state)
    if not done:
        return False
    return all(
        call_signature(tc.get("name", ""), tc.get("args") or {}) in done
        for tc in tool_calls
    )


def build_graph():
    """构建并编译 Agent 的 LangGraph 工作流图。

    依次完成三件事：注册全部节点 → 连接固定边与条件边 → compile 编译。
    编译后的图可像普通函数一样被 invoke/stream 调用。

    :return: 编译完成、可直接执行的 CompiledGraph 对象
    """
    g = StateGraph(AgentState)

    # ===== 注册节点：名字 → 节点函数映射，边中通过名字引用 =====
    g.add_node("task_understanding", task_understanding_node)
    g.add_node("skill_selection", skill_selection_node)
    g.add_node("profiler", profiler_node)
    g.add_node("planner", planner_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("insight", insight_node)
    g.add_node("report", report_node)

    # ===== 固定顺序的前置流水线：理解需求 → 选方法论 → 画像 → 规划 → 开始推理 =====
    g.add_edge(START, "task_understanding")
    g.add_edge("task_understanding", "skill_selection")
    g.add_edge("skill_selection", "profiler")
    g.add_edge("profiler", "planner")
    g.add_edge("planner", "agent")

    # 循环：agent ⇄ tools，直到 LLM 不再请求工具 / 达到最大步数
    # 条件边根据 route_after_agent 的返回值，把 agent 导向 tools 或 insight
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "insight": "insight"},
    )
    # 工具执行完毕后无条件回到 agent，由 LLM 观察结果再做下一步决策
    g.add_edge("tools", "agent")

    # ===== 收尾：洞察提炼 → 生成报告 → 图结束 =====
    g.add_edge("insight", "report")
    g.add_edge("report", END)

    return g.compile()


# 模块级单例：import 本模块即完成一次图编译，调用方直接 invoke(graph) 即可
graph = build_graph()
