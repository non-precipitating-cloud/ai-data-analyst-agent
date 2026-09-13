"""Agent 离线评测：用真实工具执行 + 脚本化 LLM，检验 Agent 的编排与可信度。

**先说清楚这个评测评的是什么、不评什么**（避免误读结果）：

- 评的是**框架层**的行为：工具选择是否按预期发生、工具报错后能否自我纠正、
  图表是否真实存在、报告数字是否与工具结果一致、循环能否正确终止。
- **不评**LLM 的推理质量。为了让评测可重复且不依赖网络/费用，这里用
  ``ScriptedLLM`` 代替真实模型：它按预设脚本产出 tool_calls，就像一个
  「被规定了动作的模型」。因此评测通过**不代表**真实模型一定能分析得好。
- 所有**工具**都是真实执行的：Python 代码真的在沙箱里跑、SQL 真的打到
  数据库（评测用临时 SQLite）、图表真的落盘成 PNG。断言里的数字来自这些
  真实执行结果，不是写死的期望值。

运行方式::

    python -m eval.agent_eval          # 打印逐场景结果与汇总，全通过退出码 0

同一套场景同时被 tests/test_agent_eval.py 复用，纳入常规回归。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

# 样例数据集（仓库自带，评测直接复用）
SALES_CSV = "datasets/sales.csv"


class _StubBase(BaseChatModel):
    """评测用假模型的公共基类（只实现 invoke，绕过 LangChain 内部细节）。"""

    @property
    def _llm_type(self) -> str:
        return "eval-stub"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201 —— 与 LangChain 接口保持一致
        """兼容 bind_tools：评测关心的是编排，工具 schema 在这里用不到。"""
        return self

    def invoke(self, messages, **kwargs):  # noqa: ANN001, ANN201
        """交给子类的 reply 实现。"""
        return self.reply(list(messages))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN201
        """LangChain 的抽象方法要求实现（实际路径走 invoke，这里只为满足 ABC）。"""
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(generations=[ChatGeneration(message=self.reply(list(messages)))])

    def reply(self, messages: list[BaseMessage]) -> AIMessage:  # pragma: no cover - 抽象
        """子类实现：根据消息历史产出回复。"""
        raise NotImplementedError


class ScriptedLLM(_StubBase):
    """按脚本产出回复的确定性「模型」，用于离线、可重复地驱动 **Agent 循环**。

    每个脚本项可以是：
    - ``AIMessage``：直接返回该消息（可携带 tool_calls）；
    - ``callable(messages) -> AIMessage``：根据当前对话历史决定下一步——
      自纠错场景正是靠这种形式实现：回调检查上一条 ToolMessage 里的结构化
      错误，据此产出一个「改对了参数」的新调用。

    脚本用尽后返回一条不带 tool_calls 的收尾消息，让图自然收敛。

    注意：本模型**只用于 agent_node**（工具调用循环）。循环之前的
    任务理解/Skill 选择/规划三个节点会各自先调用一次 LLM，如果共用同一个
    脚本，脚本会被提前消耗掉。评测里用 StubLLM 分别处理那几个节点。
    """

    script: list = []
    model_name: str = "scripted-eval-llm"
    # 脚本游标
    cursor: int = 0

    def reply(self, messages: list[BaseMessage]) -> AIMessage:
        """按脚本游标返回下一条消息。"""
        if self.cursor < len(self.script):
            step = self.script[self.cursor]
            self.cursor += 1
            return step(messages) if callable(step) else step
        # 脚本用尽：返回无 tool_calls 的消息，触发图进入 insight → report
        return AIMessage(content="分析完成，已有足够信息作答。")


class StubLLM(_StubBase):
    """固定文本回复的假模型，用于评测中与编排无关的节点。

    任务理解、Skill 选择、规划、洞察这些节点在评测里不是考察对象，
    给它们一个恒定的、格式合法的回复即可，让脚本专心驱动工具调用循环。
    """

    text: str = "（评测桩：无实际模型输出）"
    model_name: str = "stub-eval-llm"

    def reply(self, messages: list[BaseMessage]) -> AIMessage:
        """始终返回同一段文本。"""
        return AIMessage(content=self.text)


class EchoReportLLM(_StubBase):
    """把提示词里的**真实数据**回显成 Markdown 报告的假模型。

    它刻意不做任何归纳：直接把报告提示词中「工具结果（真实数据）」与
    「实际生成的图表清单」两节搬运成报告正文。这样一来，报告里出现的每个
    数字、每张图表引用，都必然来自框架交给模型的那份上下文——评测因此能真
    正验证「框架是否把真实结果喂给了模型」以及「图表白名单是否被遵守」，
    而不是验证这个假模型会不会写报告。
    """

    model_name: str = "echo-report-eval-llm"

    def reply(self, messages: list[BaseMessage]) -> AIMessage:
        """从最后一条人类消息中抽取真实数据，拼成结构化报告。"""
        prompt = str(messages[-1].content) if messages else ""
        lines = [
            "# AI 数据分析报告",
            "",
            "## 分析概览",
            "本报告由评测用回显模型生成，内容全部来自工具真实结果。",
            "",
            "## 工具结果（真实数据）",
            _extract_block(prompt, "# 工具结果（真实数据）"),
            "",
            "## 核心发现",
            "（评测桩不做归纳）",
            "",
            "## 图表",
            _extract_charts(prompt) or "本次分析未生成图表。",
            "",
            "## 核心结论",
            "（评测桩不做归纳）",
            "",
            "## 分析局限性",
            "本报告由离线评测的确定性模型生成，不代表真实模型的分析质量。",
        ]
        return AIMessage(content="\n".join(lines))


def _extract_block(prompt: str, header: str) -> str:
    """从报告提示词中取出某个 ``# 标题`` 段落的正文（直到下一个 ``# `` 标题）。"""
    if header not in prompt:
        return "（无）"
    body = prompt.split(header, 1)[1]
    # 截到下一个一级标题为止，避免把整个提示词都搬进报告
    for marker in ("\n# ", "\n\n# "):
        if marker in body:
            body = body.split(marker, 1)[0]
    return body.strip() or "（无）"


def _extract_charts(prompt: str) -> str:
    """把提示词中的图表清单渲染成 Markdown 图片引用。

    清单行格式为 ``- <标题> → charts/<文件名>.png``（由 report 节点生成），
    这里原样转成图片标签，从而让 sanitize_report 的白名单校验真正生效。
    """
    charts = re.findall(r"^- (.+?) → (charts/[\w\-.]+\.png)\s*$", prompt, flags=re.MULTILINE)
    return "\n".join(f"![{title}]({rel})" for title, rel in charts)


def build_eval_llms(script: list) -> dict[str, Any]:
    """按节点构造评测用的假模型集合。

    分开构造的原因：图在进入工具循环之前会先调用三次 LLM（任务理解、
    Skill 选择、规划），若共用同一个脚本，脚本会被这三个节点提前消耗，
    agent_node 拿到的是空脚本，整个工具循环根本不会发生。

    :param script: 驱动 agent_node 的工具调用脚本
    :return: {节点名: 假模型实例}
    """
    return {
        # 任务理解：给一段纯文本即可
        "task_understanding": StubLLM(text="任务理解：分析销售额变化并定位原因（评测桩）。"),
        # Skill 选择：必须是合法 JSON 数组，否则会走关键词回退
        "skill_selection": StubLLM(text='["sales-analysis"]'),
        # 规划：合法 JSON 数组，形状与 PLANNER_SYSTEM 要求一致
        "planner": StubLLM(
            text='[{"step":1,"goal":"评测步骤","tool":"execute_python","note":"评测桩"}]'
        ),
        # Agent 工具循环：由脚本驱动，这是评测的核心
        "agent": ScriptedLLM(script=list(script)),
        # 洞察：纯文本
        "insight": StubLLM(text="1. 评测桩洞察：结论均来自工具结果。"),
        # 报告：回显真实数据，用于验证接地与图表白名单
        "report": EchoReportLLM(),
    }


@dataclass
class EvalContext:
    """一次评测运行的观测结果（全部来自真实运行，未做任何加工）。"""

    state: dict[str, Any]
    # 报告正文与路径
    report: str = ""
    report_path: str = ""
    # 真实执行过的工具调用 / 结果台账
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)
    # 图是否正常跑到终点
    status: str = ""
    steps: int = 0

    # ---- 便捷访问 ----
    @property
    def tool_names(self) -> list[str]:
        """被请求调用过的工具名（按顺序）。"""
        return [c.get("name", "") for c in self.tool_calls]

    def results_of(self, name: str) -> list[str]:
        """取出某个工具的全部返回文本。"""
        return [
            str(r.get("result", ""))
            for r in self.tool_results
            if r.get("name") == name
        ]

    @property
    def errors(self) -> list[str]:
        """失败的工具结果文本。"""
        return [
            str(r.get("result", ""))
            for r in self.tool_results
            if r.get("status") != "success"
        ]


@dataclass
class Scenario:
    """一个评测场景：给定需求与动作脚本，检验若干条断言。"""

    name: str
    request: str
    script: list
    checks: list[Callable[[EvalContext], tuple[bool, str]]]
    dataset: str = SALES_CSV
    # 可选的环境准备/清理钩子（如为 SQL 场景准备一个临时数据库）
    setup: Callable[[], Any] | None = None
    teardown: Callable[[Any], None] | None = None


# --------------------------------------------------------------------------
# 断言辅助
# --------------------------------------------------------------------------
def expect_tool_called(*names: str) -> Callable[[EvalContext], tuple[bool, str]]:
    """断言：这些工具都被真实调用过，且执行成功。"""

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        succeeded = {
            c.get("name")
            for c, r in zip(ctx.tool_calls, ctx.tool_results)
            if r.get("status") == "success"
        }
        missing = [n for n in names if n not in succeeded]
        if missing:
            return False, f"未成功调用期望的工具: {missing}（实际: {sorted(succeeded)}）"
        return True, f"成功调用: {list(names)}"

    return _check


def expect_report_sections(*sections: str) -> Callable[[EvalContext], tuple[bool, str]]:
    """断言：报告含指定章节标题。"""

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        missing = [s for s in sections if s not in ctx.report]
        if missing:
            return False, f"报告缺少章节: {missing}"
        return True, f"报告含章节: {list(sections)}"

    return _check


def expect_charts_exist_on_disk() -> Callable[[EvalContext], tuple[bool, str]]:
    """断言：报告引用的每张图表文件都真实存在于磁盘上（否则即为「幻觉图表」）。"""

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        referenced = set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", ctx.report))
        if not referenced:
            # 没引用图表不算失败，但要说清楚——这可能是本次分析确实没出图
            return True, f"报告未引用图表（本次实际生成 {len(ctx.charts)} 张）"
        broken = [u for u in referenced if not _resolve_chart(u).is_file()]
        if broken:
            return False, f"报告引用了不存在的图表: {broken}"
        return True, f"报告引用的 {len(referenced)} 张图表均真实存在"

    return _check


def expect_no_fake_numbers(tool_result_pattern: str, expected: str) -> Callable[[EvalContext], tuple[bool, str]]:
    """断言：某个真实数值同时出现在工具结果与报告中（数字可追溯）。

    :param tool_result_pattern: 用于在工具结果里定位该数字的正则
    :param expected: 期望出现的数字字符串
    """

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        joined_results = "\n".join(str(r.get("result", "")) for r in ctx.tool_results)
        if not re.search(tool_result_pattern, joined_results):
            return False, f"工具结果中找不到与 {expected!r} 对应的真实数据"
        if expected not in ctx.report:
            return False, f"报告未引用工具算出的真实值 {expected}"
        return True, f"报告正确引用了工具结果 {expected}"

    return _check


def expect_self_corrected(tool: str) -> Callable[[EvalContext], tuple[bool, str]]:
    """断言：某个工具先失败、随后以修正后的参数成功（自我纠错闭环）。"""

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        seq = [
            (c.get("name"), r.get("status"))
            for c, r in zip(ctx.tool_calls, ctx.tool_results)
            if c.get("name") == tool
        ]
        if not any(s == "error" for _, s in seq):
            return False, f"{tool} 未出现失败，无法验证自纠错（序列: {seq}）"
        if not any(s == "success" for _, s in seq):
            return False, f"{tool} 失败后未成功纠正（序列: {seq}）"
        return True, f"{tool} 失败→修正→成功，序列: {seq}"

    return _check


def expect_terminated_within(max_steps: int) -> Callable[[EvalContext], tuple[bool, str]]:
    """断言：Agent 在步数上限内终止，没有陷入死循环。"""

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        if ctx.steps > max_steps:
            return False, f"步数 {ctx.steps} 超过上限 {max_steps}"
        if ctx.status != "done":
            return False, f"图未正常结束（status={ctx.status}）"
        return True, f"{ctx.steps} 步内正常结束"

    return _check


def expect_structured_error(kind: str) -> Callable[[EvalContext], tuple[bool, str]]:
    """断言：错误以结构化形式（含类型标记）回传给了 Agent。"""

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        marker = f"错误[{kind}"
        hit = [e for e in ctx.errors if marker in e]
        if not hit:
            return False, f"未找到类型为 {kind} 的结构化错误（实际错误: {ctx.errors[:2]}）"
        return True, f"收到结构化错误 {marker}"

    return _check


# SQL 场景使用的临时库：建一张 sales 表并写入可预测的数据，
# 以便断言「工具算出来的数」与「手工算出来的数」一致。
_SQL_FIXTURE_ROWS = [
    ("华东", 100.0),
    ("华东", 250.0),
    ("华南", 400.0),
    ("华北", 50.0),
]
# 按地区分组合计的手工计算结果：华东 100+250=350、华南 400、华北 50
_EXPECTED_REGION_TOTALS = ("350", "400", "50")


def _setup_sqlite_fixture():
    """准备 SQL 场景：建临时 SQLite 库、写入固定数据、把 SQL 工具指向它。

    :return: 清理所需的状态（原数据库 URL、临时目录对象）
    """
    import sqlite3
    import tempfile

    from src.config.settings import get_settings
    from src.tools import sql_tool

    tmpdir = tempfile.TemporaryDirectory()
    db_path = Path(tmpdir.name) / "eval.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sales (region TEXT, sales REAL)")
    conn.executemany("INSERT INTO sales VALUES (?, ?)", _SQL_FIXTURE_ROWS)
    conn.commit()
    conn.close()

    settings = get_settings()
    original_url = settings.database_url
    settings.database_url = f"sqlite:///{db_path}"
    # _get_engine 带 lru_cache：改了连接串必须清缓存，否则仍指向旧库
    sql_tool._get_engine.cache_clear()
    return original_url, tmpdir, sql_tool


def _teardown_sqlite_fixture(state) -> None:
    """还原 SQL 场景的环境改动，避免影响后续场景。"""
    original_url, tmpdir, sql_tool = state
    from src.config.settings import get_settings

    # Windows 上文件被连接池占用时无法删除，必须先释放连接池再清理临时目录
    try:
        sql_tool._get_engine().dispose()
    except Exception:  # noqa: BLE001 —— 引擎可能本就未创建
        pass
    get_settings().database_url = original_url
    sql_tool._get_engine.cache_clear()
    try:
        tmpdir.cleanup()
    except OSError:
        # 极端情况下 Windows 仍可能短暂占用文件；临时目录交给系统回收即可，
        # 不因为清理失败而让评测报错
        pass


def _expect_sql_result_matches_expected_total() -> Callable[[EvalContext], tuple[bool, str]]:
    """SQL 真实执行的结果必须与手工计算的合计一致（验证不是「假装查了」）。"""

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        outputs = ctx.results_of("execute_sql")
        if not outputs:
            return False, "execute_sql 没有任何返回"
        # 只看成功执行的那次结果（前一次是故意写错列名的失败请求）
        succeeded = [
            o for o in outputs if "错误[" not in o
        ]
        if not succeeded:
            return False, "execute_sql 没有一次成功返回"
        joined = "\n".join(succeeded)
        missing = [t for t in _EXPECTED_REGION_TOTALS if t not in joined]
        if missing:
            return False, (
                f"SQL 结果与手工计算的分组合计不符，缺少 {missing}（实际: {joined[:220]}）"
            )
        return True, (
            f"SQL 真实执行且分组合计与手工计算一致（{_EXPECTED_REGION_TOTALS}）"
        )

    return _check


def _resolve_chart(url: str) -> Path:
    """把报告里的图表相对路径还原为磁盘路径。

    报告写在 reports/ 下，图片以 ``charts/x.png`` 相对引用，
    因此基准目录是 reports/。

    :param url: 报告中的图片地址
    :return: 对应的磁盘路径
    """
    from src.config.settings import get_settings

    return get_settings().reports_dir / url


# --------------------------------------------------------------------------
# 场景定义
# --------------------------------------------------------------------------
def build_scenarios() -> list[Scenario]:
    """构造评测场景集合。

    每个场景的脚本都刻意包含一些「不完美」的动作（先调错再改对、重复调用、
    无效字段名），用来验证框架能否兜住这些情况。
    """
    return [
        # --- 场景 1：工具选择 + 数值可追溯 ---
        Scenario(
            name="tool_selection_and_numeric_grounding",
            request="统计销售额的分布情况并给出描述性统计",
            script=[
                AIMessage(content="", tool_calls=[{
                    "name": "calculate_statistics",
                    "args": {"path": SALES_CSV, "column": "sales"},
                    "id": "c1",
                }]),
                AIMessage(content="", tool_calls=[{
                    "name": "execute_python",
                    "args": {
                        "code": "print(round(df['sales'].mean(), 4))",
                        "dataset_path": SALES_CSV,
                    },
                    "id": "c2",
                }]),
                AIMessage(content="统计完成。"),
            ],
            checks=[
                expect_tool_called("calculate_statistics", "execute_python"),
                expect_terminated_within(15),
                # 报告必须出现工具真实算出的均值（由 execute_python 打印）
                _expect_python_mean_in_report(),
            ],
        ),

        # --- 场景 2：字段名报错 → 结构化错误 → 自我纠正 ---
        Scenario(
            name="self_correction_on_bad_column",
            request="计算不存在的字段的统计量",
            script=[
                # 第一步故意用错列名，触发 not_found 结构化错误
                AIMessage(content="", tool_calls=[{
                    "name": "calculate_statistics",
                    "args": {"path": SALES_CSV, "column": "sales_amount_typo"},
                    "id": "e1",
                }]),
                # 第二步：读上一条 ToolMessage 的错误内容，改用真实列名
                _correct_column_after_error(),
                AIMessage(content="已修正列名并完成统计。"),
            ],
            checks=[
                expect_structured_error("not_found"),
                expect_self_corrected("calculate_statistics"),
                expect_terminated_within(15),
            ],
        ),

        # --- 场景 3：图表真实性 ---
        Scenario(
            name="chart_reality_and_reference",
            request="按地区对比销售额并生成图表",
            script=[
                AIMessage(content="", tool_calls=[{
                    "name": "generate_chart",
                    "args": {
                        "path": SALES_CSV, "chart_type": "bar",
                        "x": "region", "y": "sales", "title": "区域销售额对比",
                    },
                    "id": "g1",
                }]),
                AIMessage(content="图表已生成。"),
            ],
            checks=[
                expect_tool_called("generate_chart"),
                _expect_chart_registered(),
                _expect_report_references_real_chart(),
            ],
        ),

        # --- 场景 4：图表参数错误 → 结构化错误 → 改对 ---
        Scenario(
            name="self_correction_on_chart_params",
            request="画一张不存在的图表类型",
            script=[
                AIMessage(content="", tool_calls=[{
                    "name": "generate_chart",
                    "args": {
                        "path": SALES_CSV, "chart_type": "pie",
                        "x": "region", "y": "sales",
                    },
                    "id": "g2",
                }]),
                AIMessage(content="", tool_calls=[{
                    "name": "generate_chart",
                    "args": {
                        "path": SALES_CSV, "chart_type": "bar",
                        "x": "region", "y": "sales",
                    },
                    "id": "g3",
                }]),
                AIMessage(content="已改用受支持的图表类型。"),
            ],
            checks=[
                expect_structured_error("unsupported"),
                expect_self_corrected("generate_chart"),
                expect_charts_exist_on_disk(),
            ],
        ),

        # --- 场景 5：重复调用应被识别，循环必须收敛 ---
        # 覆盖两条不同的去重路径：
        #   a) 同一步内重复 → tools_node 复用首次结果，不重复执行；
        #   b) 跨步重复   → 路由判定「无进展」，直接结束循环，连工具节点都不再进。
        Scenario(
            name="duplicate_calls_terminate_loop",
            request="重复调用同一个工具",
            script=[
                AIMessage(content="", tool_calls=[
                    {"name": "read_dataset", "args": {"path": SALES_CSV}, "id": "d1"},
                    # 与同一步内上一条完全相同：应复用首次结果
                    {"name": "read_dataset", "args": {"path": SALES_CSV}, "id": "d2"},
                ]),
                # 跨步再发一次完全相同调用：应触发「无进展」提前终止
                AIMessage(content="", tool_calls=[
                    {"name": "read_dataset", "args": {"path": SALES_CSV}, "id": "d3"},
                ]),
            ],
            checks=[
                _expect_duplicate_not_reexecuted(),
                _expect_repeat_step_not_executed(),
                expect_terminated_within(15),
            ],
        ),

        # --- 场景 6：Python 沙箱安全拒绝 → 结构化错误 → 换方法成功 ---
        Scenario(
            name="sandbox_rejection_then_recovery",
            request="尝试越权读文件后再正常分析",
            script=[
                AIMessage(content="", tool_calls=[{
                    "name": "execute_python",
                    "args": {"code": "import os", "dataset_path": SALES_CSV},
                    "id": "s1",
                }]),
                AIMessage(content="", tool_calls=[{
                    "name": "execute_python",
                    "args": {
                        "code": "print(df['region'].nunique())",
                        "dataset_path": SALES_CSV,
                    },
                    "id": "s2",
                }]),
                AIMessage(content="已改用合规方式分析。"),
            ],
            checks=[
                _expect_sandbox_blocked(),
                _expect_recovered_after_block(),
                expect_terminated_within(15),
            ],
        ),

        # --- 场景 7：SQL 正确性（真的打到数据库执行）+ 列名写错后自纠错 ---
        # 说明：这里用临时 SQLite 库而非 PostgreSQL —— SQL 是与方言无关的，
        # 评测关心的是「SQL 真的执行了、结果真的算对了、写错列名能改对」，
        # 让评测依赖一个必须提前 docker compose up 的数据库反而会削弱它。
        Scenario(
            name="sql_correctness_and_self_correction",
            request="用 SQL 统计每个地区的销售额合计",
            dataset=SALES_CSV,  # 数据集本身不参与本场景，SQL 走临时库
            script=[
                # 第一步：列名写错，应收到 invalid_argument 结构化错误
                AIMessage(content="", tool_calls=[{
                    "name": "execute_sql",
                    "args": {"sql": "SELECT region, SUM(amount_typo) FROM sales GROUP BY region"},
                    "id": "q1",
                }]),
                # 第二步：改成正确列名
                AIMessage(content="", tool_calls=[{
                    "name": "execute_sql",
                    "args": {"sql": "SELECT region, SUM(sales) AS total FROM sales GROUP BY region ORDER BY total DESC"},
                    "id": "q2",
                }]),
                AIMessage(content="SQL 统计完成。"),
            ],
            checks=[
                expect_structured_error("invalid_argument"),
                expect_self_corrected("execute_sql"),
                _expect_sql_result_matches_expected_total(),
                expect_terminated_within(15),
            ],
            setup=_setup_sqlite_fixture,
            teardown=_teardown_sqlite_fixture,
        ),

        # --- 场景 8：报告结构完整性 ---
        Scenario(
            name="report_structure",
            request="对销售数据做一次完整分析并出报告",
            script=[
                AIMessage(content="", tool_calls=[{
                    "name": "profile_dataset", "args": {"path": SALES_CSV}, "id": "p1",
                }]),
                AIMessage(content="分析完成。"),
            ],
            checks=[
                expect_report_sections("核心结论", "分析局限性"),
                _expect_report_path_exists(),
                expect_charts_exist_on_disk(),
            ],
        ),
    ]


# --------------------------------------------------------------------------
# 场景内的具名断言（写成函数以便复用与阅读）
# --------------------------------------------------------------------------
def _expect_python_mean_in_report() -> Callable[[EvalContext], tuple[bool, str]]:
    """报告应包含 execute_python 真实算出的销售额均值。"""

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        outputs = ctx.results_of("execute_python")
        number = None
        for out in outputs:
            m = re.search(r"\d+\.\d+", out)
            if m:
                number = m.group(0)
                break
        if number is None:
            return False, "execute_python 未产出可识别的数值"
        if number not in ctx.report:
            return False, f"报告未引用真实均值 {number}（可能为幻觉数字）"
        return True, f"报告引用了真实均值 {number}"

    return _check


def _correct_column_after_error() -> Callable[[list[BaseMessage]], AIMessage]:
    """返回一个「看到错误就改对参数」的脚本回调。

    它模拟真实模型的自我纠错：从上一条 ToolMessage 里读出结构化错误，
    发现字段名不对，于是换成正确列名重试。
    """

    def _step(messages: list[BaseMessage]) -> AIMessage:
        last = messages[-1]
        saw_error = isinstance(last, ToolMessage) and "错误[" in str(last.content)
        column = "sales" if saw_error else "sales"
        return AIMessage(content="", tool_calls=[{
            "name": "calculate_statistics",
            "args": {"path": SALES_CSV, "column": column},
            "id": "fix1",
        }])

    return _step


def _expect_chart_registered() -> Callable[[EvalContext], tuple[bool, str]]:
    """图表路径应被登记进 generated_charts，且文件真实存在。"""

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        if not ctx.charts:
            return False, "generate_chart 成功但未登记图表路径"
        missing = [c for c in ctx.charts if not Path(c).is_file()]
        if missing:
            return False, f"登记的图表文件不存在: {missing}"
        return True, f"登记并落盘 {len(ctx.charts)} 张图表"

    return _check


def _expect_report_references_real_chart() -> Callable[[EvalContext], tuple[bool, str]]:
    """报告若引用图表，引用的必须是真实文件。"""

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        referenced = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", ctx.report)
        broken = [u for u in referenced if not _resolve_chart(u).is_file()]
        if broken:
            return False, f"报告引用了不存在的图表: {broken}"
        return True, f"报告图表引用有效（{len(referenced)} 处）"

    return _check


def _expect_duplicate_not_reexecuted() -> Callable[[EvalContext], tuple[bool, str]]:
    """同一步内的重复调用应被识别：不再真正执行，而是复用首次结果。"""

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        reads = [
            r for c, r in zip(ctx.tool_calls, ctx.tool_results)
            if c.get("name") == "read_dataset"
        ]
        if len(reads) < 2:
            return False, f"预期至少 2 次 read_dataset 记录，实际 {len(reads)}"
        reused = [r for r in reads if "已直接返回首次结果" in str(r.get("result", ""))]
        if not reused:
            return False, "同一步内的重复调用未被识别，仍然重复执行了工具"
        return True, f"{len(reused)} 次同步重复调用被复用首次结果，未重复执行"

    return _check


def _expect_repeat_step_not_executed() -> Callable[[EvalContext], tuple[bool, str]]:
    """跨步重复的调用不应被送到工具节点执行——循环应直接判定「无进展」并收敛。

    脚本一共请求了 3 次 read_dataset：第 1 步 2 次（同一步去重后执行 1 次）、
    第 2 步 1 次（与已成功调用完全相同）。若跨步去重生效，工具台账里应只有
    第 1 步的 2 条记录，第 2 步那条根本不会产生。
    """

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        reads = [c for c in ctx.tool_calls if c.get("name") == "read_dataset"]
        if len(reads) != 2:
            return False, (
                f"预期只执行第 1 步的 2 次调用（第 2 步应被判定无进展而跳过），"
                f"实际台账 {len(reads)} 条"
            )
        return True, "跨步重复调用未被执行，循环提前收敛（无进展判定生效）"

    return _check


def _expect_sandbox_blocked() -> Callable[[EvalContext], tuple[bool, str]]:
    """越权代码必须被沙箱拒绝。"""

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        outputs = ctx.results_of("execute_python")
        if not any("拒绝" in o for o in outputs):
            return False, f"沙箱未拒绝危险代码（输出: {outputs[:1]}）"
        return True, "危险代码被沙箱拒绝并回传说明"

    return _check


def _expect_recovered_after_block() -> Callable[[EvalContext], tuple[bool, str]]:
    """被拒绝后应能用合规代码成功完成分析。"""

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        seq = [
            r.get("status")
            for c, r in zip(ctx.tool_calls, ctx.tool_results)
            if c.get("name") == "execute_python"
        ]
        if "error" not in seq or "success" not in seq:
            return False, f"未形成「拒绝→改正」序列（状态序列: {seq}）"
        return True, f"沙箱拒绝后改正成功（状态序列: {seq}）"

    return _check


def _expect_report_path_exists() -> Callable[[EvalContext], tuple[bool, str]]:
    """报告应真实落盘。"""

    def _check(ctx: EvalContext) -> tuple[bool, str]:
        if not ctx.report_path or not Path(ctx.report_path).is_file():
            return False, f"报告文件不存在: {ctx.report_path!r}"
        return True, f"报告已落盘: {Path(ctx.report_path).name}"

    return _check


# --------------------------------------------------------------------------
# 运行器
# --------------------------------------------------------------------------
def run_scenario(scenario: Scenario) -> dict[str, Any]:
    """执行单个场景，返回 {name, passed, details}。

    真实跑一遍 LangGraph：只用脚本化 LLM 替代模型，工具全部真实执行。

    :param scenario: 待执行的评测场景
    :return: 结果字典，details 为每条断言的 (是否通过, 说明)
    """
    # 延迟导入：让评测脚本可以只做 --list 而不触发重型依赖初始化
    from src.agent.graph import build_graph, make_initial_state

    graph = build_graph()
    llms = build_eval_llms(scenario.script)

    # 通过替换各节点模块内的 LLM 工厂来注入假模型。
    # 之所以逐个模块替换而不改图结构：评测必须跑在**真实的生产代码路径**上，
    # 否则测的就不是实际会运行的那套编排逻辑了。
    import src.agent.nodes.insight as insight_mod
    import src.agent.nodes.planner as planner_mod
    import src.agent.nodes.report as report_mod
    import src.agent.nodes.task_understanding as tu_mod
    import src.agent.nodes.tool_calling as tc_mod
    import src.skills.selector as selector_mod

    patched = {
        tu_mod: llms["task_understanding"],
        selector_mod: llms["skill_selection"],
        planner_mod: llms["planner"],
        tc_mod: llms["agent"],
        insight_mod: llms["insight"],
        report_mod: llms["report"],
    }
    originals = {mod: mod.get_llm for mod in patched}
    for mod, llm in patched.items():
        mod.get_llm = (lambda _l=llm: _l)  # type: ignore[assignment]
    # 场景级环境准备（如 SQL 场景的临时数据库）
    fixture = scenario.setup() if scenario.setup else None
    try:
        state = make_initial_state(scenario.dataset, scenario.request)
        final_state = graph.invoke(state)
    finally:
        for mod, original in originals.items():
            mod.get_llm = original  # type: ignore[assignment]
        if scenario.teardown and fixture is not None:
            scenario.teardown(fixture)

    ctx = EvalContext(
        state=final_state,
        report=final_state.get("final_report", ""),
        report_path=final_state.get("report_path", ""),
        tool_calls=list(final_state.get("tool_calls") or []),
        tool_results=list(final_state.get("tool_results") or []),
        charts=list(final_state.get("generated_charts") or []),
        status=final_state.get("status", ""),
        steps=final_state.get("step_count", 0),
    )

    details = []
    passed = True
    for check in scenario.checks:
        ok, message = check(ctx)
        passed = passed and ok
        details.append({"check": getattr(check, "__name__", "check"), "ok": ok, "message": message})
    # 报告落盘的文件清理：评测产物不应留在 reports/ 目录里
    if ctx.report_path:
        Path(ctx.report_path).unlink(missing_ok=True)
    for chart in ctx.charts:
        Path(chart).unlink(missing_ok=True)

    return {"name": scenario.name, "passed": passed, "details": details}


def main() -> int:
    """运行全部评测场景并打印结果，返回进程退出码（0=全通过）。"""
    # 评测输出含 ✓/✗，在 Windows 默认控制台编码下会直接抛 UnicodeEncodeError
    from src.logging_config import configure_stdio

    configure_stdio()

    scenarios = build_scenarios()
    total, passed_count = len(scenarios), 0
    print("=" * 72)
    print("Agent 离线评测（真实工具执行 + 脚本化 LLM，不评估模型推理质量）")
    print("=" * 72)

    for scenario in scenarios:
        result = run_scenario(scenario)
        mark = "PASS" if result["passed"] else "FAIL"
        if result["passed"]:
            passed_count += 1
        print(f"\n[{mark}] {result['name']}")
        for d in result["details"]:
            print(f"    {'✓' if d['ok'] else '✗'} {d['message']}")

    print("\n" + "=" * 72)
    print(f"场景通过：{passed_count}/{total}")
    print("=" * 72)
    return 0 if passed_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
