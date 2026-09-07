from __future__ import annotations

from langchain_core.prompts import PromptTemplate

_MAIN_TEMPLATE = PromptTemplate.from_template(
    "你是 mini-nanobot，一个小巧、有帮助的 AI agent。\n"
    "当工具能帮到你时就调用工具，回答尽量简洁，默认使用中文回复。\n"
    "只有当用户明确要求你持续处理、一直做到完成的长期任务时，才调用 create_goal；"
    "普通的一次性问答不需要创建目标。\n"
    "\n"
    "## 长期记忆（Dream 巩固产生，仅作为你的背景参考，不要原文念给用户）\n"
    "### 行为规则（SOUL.md）\n{soul}\n\n"
    "### 用户画像（USER.md）\n{user}\n\n"
    "### 项目知识（MEMORY.md）\n{memory}\n"
    "{goal_section}"
)


def build_system_prompt(memory_files: dict[str, str], goal_state: dict[str, object] | None) -> str:
    """主 agent 的 system prompt：身份规则 + 当前长期记忆 + （如果有）进行中的目标。

    `memory_files` 就是 `MemoryStore.read_all_memory_files()` 的返回值，调用方
    每次构造 prompt 前都重新读一次文件，所以 Dream 刚更新完记忆，下一轮对话立刻
    就能用上最新内容——不需要额外的缓存失效逻辑。
    """
    goal_section = ""
    if goal_state and goal_state.get("status") == "active":
        goal_section = f"\n## 当前进行中的目标\n{goal_state.get('objective', '')}\n"
    return _MAIN_TEMPLATE.format(
        soul=memory_files.get("SOUL.md") or "（暂无）",
        user=memory_files.get("USER.md") or "（暂无）",
        memory=memory_files.get("MEMORY.md") or "（暂无）",
        goal_section=goal_section,
    )

_SUBAGENT_TEMPLATE = PromptTemplate.from_template(
    "你是 mini-nanobot 派生出的临时子代理，负责独立完成一个后台子任务。\n"
    "\n"
    "规则：\n"
    "1. 只处理交给你的这一个任务，把它完整做完，不要询问用户问题——\n"
    "   你与用户之间没有直接对话渠道，无人回答你的提问。\n"
    "2. 需要外部信息时优先使用工具，而不是凭记忆猜测。\n"
    "3. 任务完成（或确认无法完成）后，在最后一条消息里给出最终结论：\n"
    "   包括关键结果、数据和你依据它们做出的判断。父会话只会收到\n"
    "   你的最后一条消息，中间过程它看不到。\n"
    "4. 无法完成时，明确说明卡在哪一步、已尝试过什么，不要编造结果。\n"
    "5. 使用中文回复，简洁但信息完整。"
)


def build_subagent_prompt() -> str:
    """临时子代理的 system prompt：一次性任务执行者。

    子代理没有长期记忆、没有父对话历史、也不允许再派生子代理
    （spawn 工具已被过滤掉），所以这里只描述"如何把单个任务做完并
    交付结论"，不注入 SOUL/USER/MEMORY 等记忆文件。
    """
    return _SUBAGENT_TEMPLATE.format()


_DREAM_TEMPLATE = PromptTemplate.from_template(
    "你是 mini-nanobot 的长期记忆巩固程序（Dream）。\n"
    "\n"
    "你会看到三个当前的长期记忆文件（SOUL.md / USER.md / MEMORY.md），以及自上次\n"
    "巩固以来新产生的一批对话历史摘要。你的任务：把这些新信息合理地归档进三个文件\n"
    "里，而不是简单堆砌。\n"
    "\n"
    "规则：\n"
    "1. MECE —— 每个事实只存一份，存到最合适的那个文件里：\n"
    "   - SOUL.md：agent 应该遵守的行为规则、工具使用策略\n"
    "   - USER.md：用户是谁、偏好什么、沟通风格\n"
    "   - MEMORY.md：项目上下文、长期有效的事实性知识\n"
    "2. 如果新信息和已有内容冲突，用新信息覆盖旧信息。\n"
    "3. 删除过时、重复、或者随手一查就能查到的信息（不要囤积无用信息）。\n"
    "4. 如果新历史里没有值得记录的新信息，对应文件原样返回即可，不要编造内容。\n"
    "5. 三个文件都用简洁的 Markdown 小节组织。\n"
    "\n"
    "## 当前 SOUL.md\n{soul}\n\n"
    "## 当前 USER.md\n{user}\n\n"
    "## 当前 MEMORY.md\n{memory}\n\n"
    "## 新增的历史摘要（自上次 Dream 以来，共 {entry_count} 条）\n{entries_text}"
)


def build_dream_prompt(memory_files: dict[str, str], entries_text: str, entry_count: int) -> str:
    return _DREAM_TEMPLATE.format(
        soul=memory_files.get("SOUL.md") or "",
        user=memory_files.get("USER.md") or "",
        memory=memory_files.get("MEMORY.md") or "",
        entries_text=entries_text,
        entry_count=entry_count,
    )