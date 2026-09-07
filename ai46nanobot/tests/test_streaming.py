"""流式改造（非流式→流式）的行为测试。

覆盖链路：AgentService._run_once 分支 -> MessageBus 出站事件 ->
ChannelManager 路由 -> BaseChannel 默认降级缓冲 / ConsoleChannel 实时打印。

不依赖真实 API key / 网络（FakeGraph 模拟模型事件流）。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from mini_nanobot.bus import InboundMessage, MessageBus, OutboundMessage
from mini_nanobot.channels.base import BaseChannel
from mini_nanobot.channels.console import ConsoleChannel
from mini_nanobot.channels.manager import ChannelManager
from mini_nanobot.service import AgentService


# --------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------

class FakeChunk:
    def __init__(self, text: str) -> None:
        self.content = text


class FakeGraph:
    """模拟真实图：非流式走 ainvoke，流式走 astream_events 逐字吐 chunk。"""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.ainvoke_calls = 0
        self.stream_calls = 0

    async def ainvoke(self, input, config=None, **kw):
        self.ainvoke_calls += 1
        return {"messages": [AIMessage(content="".join(self.chunks))]}

    async def astream_events(self, input, config=None, **kw):
        self.stream_calls += 1
        for t in self.chunks:
            yield {"event": "on_chat_model_stream", "data": {"chunk": FakeChunk(t)}}


class FakeRuntime:
    def __init__(self, graph: FakeGraph) -> None:
        self.graph = graph
        self.memory = None
        self.sessions = None
        self.dream_auto_threshold = 0
        self.llm = None


class SpyChannel(BaseChannel):
    """不覆写 send_delta —— 走 BaseChannel 默认「攒缓冲降级」逻辑。"""

    name = "spy"

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(bus)
        self.sent: list[tuple[str, str]] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send(self, session_id: str, content: str, metadata=None) -> None:
        self.sent.append((session_id, content))


def _msg(session_id: str, *, supports_stream: bool) -> InboundMessage:
    return InboundMessage(
        channel="console",
        session_id=session_id,
        content="hi",
        metadata={"supports_stream": supports_stream},
    )


def _drain(bus: MessageBus) -> list[OutboundMessage]:
    out = []
    while not bus.outbound.empty():
        out.append(bus.outbound.get_nowait())
    return out


# --------------------------------------------------------------------------
# Part A: service._run_once 分支
# --------------------------------------------------------------------------

async def test_streaming_branch_sends_deltas_then_stream_end() -> None:
    bus = MessageBus()
    svc = AgentService(bus, FakeRuntime(FakeGraph(["你", "好", "！"])))
    await svc._run_once(_msg("s1", supports_stream=True), "hi")

    kinds = [e.event for e in _drain(bus)]
    assert kinds == ["delta", "delta", "delta", "stream_end"]


async def test_streaming_delta_content_concatenates_in_order() -> None:
    bus = MessageBus()
    svc = AgentService(bus, FakeRuntime(FakeGraph(["你", "好", "！"])))
    await svc._run_once(_msg("s1", supports_stream=True), "hi")

    text = "".join(e.content for e in _drain(bus) if e.event == "delta")
    assert text == "你好！"


async def test_non_streaming_branch_sends_single_final() -> None:
    bus = MessageBus()
    graph = FakeGraph(["完整回复"])
    svc = AgentService(bus, FakeRuntime(graph))
    await svc._run_once(_msg("s2", supports_stream=False), "hi")

    evs = _drain(bus)
    assert [e.event for e in evs] == ["final"]
    assert evs[0].content == "完整回复"
    assert graph.ainvoke_calls == 1 and graph.stream_calls == 0


async def test_empty_stream_falls_back_to_final() -> None:
    bus = MessageBus()
    svc = AgentService(bus, FakeRuntime(FakeGraph([])))
    await svc._run_once(_msg("s3", supports_stream=True), "hi")

    evs = _drain(bus)
    assert [e.event for e in evs] == ["final"]
    assert "空响应" in evs[0].content


async def test_missing_metadata_treated_as_non_streaming() -> None:
    bus = MessageBus()
    svc = AgentService(bus, FakeRuntime(FakeGraph(["缺省回复"])))
    await svc._run_once(InboundMessage(channel="c", session_id="s4", content="hi"), "hi")

    evs = _drain(bus)
    assert [e.event for e in evs] == ["final"]


# --------------------------------------------------------------------------
# Part B: manager 路由 + channel 输出
# --------------------------------------------------------------------------

async def test_default_channel_buffers_deltas_until_end() -> None:
    bus = MessageBus()
    spy = SpyChannel(bus)
    mgr = ChannelManager(bus, [spy])
    task = None
    try:
        import asyncio

        task = asyncio.create_task(mgr._dispatch_outbound())
        for d in ["你", "好", "！"]:
            await bus.publish_outbound(OutboundMessage(
                channel="spy", session_id="s1", content=d, event="delta"))
        await bus.publish_outbound(OutboundMessage(
            channel="spy", session_id="s1", content="", event="stream_end"))
        await asyncio.sleep(0.05)
    finally:
        if task is not None:
            task.cancel()
    assert spy.sent == [("s1", "你好！")]


async def test_console_send_delta_prints_inline(capsys) -> None:
    console = ConsoleChannel(MessageBus(), sessions=None)
    await console.send_delta("s1", "你")
    await console.send_delta("s1", "好")
    assert capsys.readouterr().out == "你好"


async def test_console_send_delta_end_prints_newline(capsys) -> None:
    console = ConsoleChannel(MessageBus(), sessions=None)
    await console.send_delta("s1", "a")
    await console.send_delta_end("s1")
    assert capsys.readouterr().out == "a\n"
