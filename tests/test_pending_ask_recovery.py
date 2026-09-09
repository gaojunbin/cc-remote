"""Zero-model tests for resident question recovery after a client cold start.

All questions, client identities and sessions below are synthetic. No relay,
native engine, user transcript, model API or notification service is contacted.
"""
from __future__ import annotations

import asyncio

import pytest

from cc_remote.protocol import (
    AnswerQuestion, AskUser, AskUserClosed, Delta, Error, Hello,
    TurnEnd, TurnResult, UserMsg, deserialize, serialize,
)
from cc_remote.wrapper.claude_questions import AskCancelled, AskSuperseded, AskTimeout
from cc_remote.wrapper.machine import _BtwSpawnFailure
from cc_remote.wrapper.ringbuffer import RingBuffer
from tests.test_multisession import _mk_ctx, _mk_machine


OPTIONS = [{"label": "A", "ds": "First option"}, {"label": "B"}]


async def _start_ask(machine, ctx, **kwargs):
    task = asyncio.create_task(machine._on_ask(
        ctx, "Synthetic question", OPTIONS, **kwargs,
    ))

    async def emitted():
        while ctx.active_ask is None:
            if task.done():
                await task
            await asyncio.sleep(0)

    await asyncio.wait_for(emitted(), timeout=1)
    return task, ctx.active_ask.ask_id


def _hello(machine, ctx, *, client="client-a", mode="fresh"):
    return Hello(
        role="client", client_id=client, route_id="connection-new",
        cursors=None if mode == "fresh" else {
            ctx.key: ctx.seq if mode == "tail" else 0,
        },
        generations={ctx.key: machine.instance_id} if mode != "rebuild" else {},
    )


async def _answer(machine, ctx, ask_id, *, client="client-a", sid=None):
    return await machine._handle_answer_question(AnswerQuestion(
        sid=sid or ctx.key, ask_id=ask_id, answer="A", client_id=client,
        cmd_id="answer-command",
    ))


@pytest.mark.parametrize("state", ["idle", "running"])
@pytest.mark.parametrize("mode", ["fresh", "tail", "rebuild", "evicted"])
def test_hello_restores_full_active_question_without_consuming_sequence(state, mode):
    async def run():
        machine, transport = _mk_machine()
        ctx = _mk_ctx("session-a", "session-a")
        ctx.state = state
        if mode == "evicted":
            ctx.buffer = RingBuffer(1, 10_000)
        machine.sessions[ctx.key] = ctx
        await machine._emit(ctx, UserMsg(msg_id="turn-a", prompt="Synthetic turn"))
        task, ask_id = await _start_ask(
            machine, ctx, header="Synthetic header", allow_text=True,
            multi_select=True, secret=True,
        )
        original = next(event for event in transport.sent if isinstance(event, AskUser))
        await machine._emit(ctx, Delta(message_id="output-a", text="Synthetic output"))
        if mode == "evicted":
            assert all(not isinstance(frame, AskUser) for _, frame in ctx.buffer._buf)
        sequence = ctx.seq
        tail = ctx.buffer.tail_seq
        transport.sent.clear()

        await machine._handle_client_hello(_hello(machine, ctx, mode=mode))
        restored = [event for event in transport.sent if isinstance(event, AskUser)]
        assert len(restored) == 1
        event = restored[0]
        assert event.ask_id == ask_id and event.question == original.question
        assert event.options == OPTIONS and event.header == "Synthetic header"
        assert event.allow_text and event.multi_select and event.secret
        assert event.sid == "session-a" and event.to == "client-a"
        assert event.route_id == "connection-new" and event.seq is None
        assert deserialize(serialize(event)) == event  # unchanged v35 schema
        assert ctx.seq == sequence and ctx.buffer.tail_seq == tail
        assert original.to is None and original.route_id is None
        assert ctx.active_ask.to is None and ctx.active_ask.route_id is None
        assert await _answer(machine, ctx, ask_id) is None
        assert await task == ["A"]
        assert ctx.active_ask is None
        assert ctx.pending_asks == {} and ctx.pending_ask_specs == {}

    asyncio.run(run())


@pytest.mark.parametrize("btw", [False, True])
@pytest.mark.parametrize("mode", ["fresh", "tail", "rebuild"])
def test_private_question_recovery_and_answer_keep_original_client_scope(btw, mode):
    async def run():
        machine, transport = _mk_machine()
        ctx = _mk_ctx("session-private", "session-private")
        ctx.btw = btw
        ctx.owner_client_id = "client-a" if btw else None
        ctx.state = "running"
        machine.sessions[ctx.key] = ctx
        await machine._emit(ctx, UserMsg(msg_id="turn-a", prompt="Synthetic turn"))
        task, ask_id = await _start_ask(machine, ctx, to=None if btw else "client-a")
        # A targeted non-question frame must not leak through general replay.
        await machine._emit(ctx, Error(
            code="internal", message="Synthetic private error", to="client-a"))
        transport.sent.clear()
        await machine._handle_client_hello(_hello(machine, ctx, client="client-b", mode=mode))
        assert not any(isinstance(event, (AskUser, Error)) for event in transport.sent)
        if btw:
            assert transport.sent == []
        result = await _answer(machine, ctx, ask_id, client="client-b")
        assert result.type == "error" and result.code == "auth"
        assert not ctx.pending_asks[ask_id].done()
        transport.sent.clear()
        await machine._handle_client_hello(_hello(machine, ctx, mode=mode))
        seeds = [event for event in transport.sent if isinstance(event, AskUser)]
        assert len(seeds) == 1 and seeds[0].to == "client-a"
        assert ctx.active_ask.to == "client-a"
        assert await _answer(machine, ctx, ask_id) is None
        assert await task == "A"
        transport.sent.clear()
        await machine._handle_client_hello(_hello(machine, ctx, client="client-b", mode="rebuild"))
        assert not any(isinstance(event, (AskUser, AskUserClosed))
                       for event in transport.sent)
        assert not any(isinstance(event, Error)
                       and event.message == "Synthetic private error"
                       for event in transport.sent)

    asyncio.run(run())


def test_recovery_requires_client_identity_and_remains_machine_session_local():
    async def run():
        machine, transport = _mk_machine()
        other_machine, other_transport = _mk_machine()
        ctx = _mk_ctx("session-a", "session-a")
        unrelated = _mk_ctx("session-b", "session-b")
        machine.sessions = {ctx.key: ctx, unrelated.key: unrelated}
        other_machine.sessions[ctx.key] = _mk_ctx(ctx.key, ctx.session_id)
        task, ask_id = await _start_ask(machine, ctx)
        transport.sent.clear()
        await machine._handle_client_hello(Hello(role="client"))
        assert transport.sent == []
        await other_machine._handle_client_hello(_hello(other_machine, ctx))
        assert not any(isinstance(event, AskUser) for event in other_transport.sent)
        await machine._handle_client_hello(_hello(machine, ctx))
        assert all(event.sid == ctx.key for event in transport.sent if isinstance(event, AskUser))
        result = await _answer(machine, ctx, ask_id, sid=unrelated.key)
        assert result.type == "error" and not ctx.pending_asks[ask_id].done()
        await _answer(machine, ctx, ask_id)
        await task

    asyncio.run(run())


@pytest.mark.parametrize("boundary", ["answer", "interrupt", "close", "terminal", "error_terminal", "cancel", "timeout"])
def test_finished_questions_never_reappear_from_old_ring_frames(boundary):
    async def run():
        machine, transport = _mk_machine()
        ctx = _mk_ctx("session-a", "session-a")
        machine.sessions[ctx.key] = ctx
        task, ask_id = await _start_ask(machine, ctx, timeout=0.01 if boundary == "timeout" else 30)
        if boundary == "answer":
            # Test before the waiting coroutine consumes the answer/emits close.
            ctx.pending_asks[ask_id].set_result("A")
        elif boundary == "interrupt":
            machine._cancel_pending_asks(ctx)
            assert ctx.active_ask is None
        elif boundary == "close":
            await machine._emit(ctx, AskUserClosed(ask_id=ask_id, reason="cancelled"))
            assert ctx.active_ask is None
        elif boundary in {"terminal", "error_terminal"}:
            await machine._emit(ctx, TurnEnd(result=TurnResult(
                subtype="success" if boundary == "terminal" else "error_during_execution",
                duration_ms=0, is_error=boundary == "error_terminal")))
            assert ctx.active_ask is None
        elif boundary == "cancel":
            task.cancel()
        else:
            with pytest.raises(AskTimeout):
                await task
        if boundary not in {"cancel", "timeout"}:
            transport.sent.clear()
            await machine._handle_client_hello(_hello(machine, ctx, mode="rebuild"))
            assert not any(isinstance(event, AskUser) for event in transport.sent)
        if boundary == "answer":
            assert await task == "A"
        elif boundary == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await task
        elif boundary != "timeout":
            with pytest.raises(AskCancelled):
                await task
        assert ctx.active_ask is None
        assert ctx.pending_asks == {} and ctx.pending_ask_specs == {}
        transport.sent.clear()
        await machine._handle_client_hello(_hello(machine, ctx, mode="rebuild"))
        assert not any(isinstance(event, AskUser) for event in transport.sent)

    asyncio.run(run())


def test_rekey_recovers_question_under_current_session_without_focus_steal():
    async def run():
        machine, transport = _mk_machine()
        ctx = _mk_ctx("tmp-synthetic")
        machine.sessions[ctx.key] = ctx
        machine.focused_sid = "unrelated-session"
        task, ask_id = await _start_ask(machine, ctx)
        await machine._capture_session_id(ctx, "real-synthetic")
        transport.sent.clear()
        await machine._handle_client_hello(_hello(machine, ctx))
        event = next(event for event in transport.sent if isinstance(event, AskUser))
        assert event.sid == "real-synthetic" and event.ask_id == ask_id
        assert machine.focused_sid == "unrelated-session"
        await _answer(machine, ctx, ask_id)
        await task

    asyncio.run(run())


def test_emission_failure_does_not_leave_recoverable_question():
    async def run():
        machine, transport = _mk_machine()
        ctx = _mk_ctx("session-a", "session-a")
        machine.sessions[ctx.key] = ctx
        original_send = transport.send

        async def fail_question(event):
            if isinstance(event, AskUser):
                raise RuntimeError("Synthetic transport failure")
            await original_send(event)

        transport.send = fail_question
        with pytest.raises(RuntimeError, match="Synthetic transport failure"):
            await machine._on_ask(ctx, "Synthetic question", OPTIONS)
        assert ctx.active_ask is None and ctx.pending_asks == {}
        assert ctx.pending_ask_specs == {}
        transport.send = original_send
        transport.sent.clear()
        await machine._handle_client_hello(_hello(machine, ctx, mode="rebuild"))
        assert not any(isinstance(event, AskUser) for event in transport.sent)

    asyncio.run(run())


def test_answer_during_hello_seed_is_followed_by_close_on_same_session():
    async def run():
        machine, transport = _mk_machine()
        ctx = _mk_ctx("session-a", "session-a")
        machine.sessions[ctx.key] = ctx
        task, ask_id = await _start_ask(machine, ctx)
        started = asyncio.Event()
        release = asyncio.Event()
        original_send = transport.send

        async def pause_seed(event):
            if isinstance(event, AskUser) and event.seq is None:
                started.set()
                await release.wait()
            await original_send(event)

        transport.send = pause_seed
        transport.sent.clear()
        hello = asyncio.create_task(machine._handle_client_hello(_hello(machine, ctx)))
        await asyncio.wait_for(started.wait(), timeout=1)
        await _answer(machine, ctx, ask_id)
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(hello, timeout=1)
        assert await asyncio.wait_for(task, timeout=1) == "A"
        events = [event for event in transport.sent if isinstance(event, (AskUser, AskUserClosed))]
        assert [event.type for event in events] == ["ask_user", "ask_user_closed"]
        assert all(event.sid == ctx.key and event.ask_id == ask_id for event in events)
        assert ctx.active_ask is None

    asyncio.run(run())


def test_interrupt_before_first_emit_cannot_publish_cancelled_question():
    async def run():
        machine, transport = _mk_machine()
        ctx = _mk_ctx("session-a", "session-a")
        async with ctx.emit_lock:
            task = asyncio.create_task(machine._on_ask(ctx, "Synthetic question", OPTIONS))
            while not ctx.pending_asks:
                await asyncio.sleep(0)
            machine._cancel_pending_asks(ctx)
        with pytest.raises(AskCancelled):
            await task
        assert not any(isinstance(event, AskUser) for event in transport.sent)
        assert ctx.active_ask is None and ctx.pending_asks == {}

    asyncio.run(run())


def test_private_answer_is_rejected_even_before_first_question_emit():
    async def run():
        machine, transport = _mk_machine()
        ctx = _mk_ctx("session-a", "session-a")
        machine.sessions[ctx.key] = ctx
        async with ctx.emit_lock:
            task = asyncio.create_task(machine._on_ask(
                ctx, "Synthetic question", OPTIONS, to="client-a"))
            while not ctx.pending_asks:
                await asyncio.sleep(0)
            ask_id = next(iter(ctx.pending_asks))
            # The rejected reply itself waits on emit_lock, but must never
            # resolve the private Future while AskUser has not been sent yet.
            reply = asyncio.create_task(_answer(machine, ctx, ask_id, client="client-b"))
            await asyncio.sleep(0)
            assert ctx.active_ask is None
            assert not ctx.pending_asks[ask_id].done()
        result = await asyncio.wait_for(reply, timeout=1)
        assert result.code == "auth" and result.to == "client-b"
        assert not ctx.pending_asks[ask_id].done()
        await _answer(machine, ctx, ask_id)
        assert await task == "A"
        assert ctx.active_ask is None

    asyncio.run(run())


def test_superseded_question_recovery_contains_only_current_single_slot():
    async def run():
        machine, transport = _mk_machine()
        ctx = _mk_ctx("session-a", "session-a")
        machine.sessions[ctx.key] = ctx
        first, first_id = await _start_ask(machine, ctx)
        second = asyncio.create_task(machine._on_ask(
            ctx, "Synthetic replacement", OPTIONS, ask_id="ask-replacement"))
        await asyncio.sleep(0)
        assert list(ctx.pending_asks) == [first_id]
        assert ctx.active_ask.ask_id == first_id
        ctx.pending_asks[first_id].set_exception(AskSuperseded())
        with pytest.raises(AskSuperseded):
            await first
        while ctx.active_ask is None:
            await asyncio.sleep(0)
        assert list(ctx.pending_asks) == ["ask-replacement"]
        transport.sent.clear()
        await machine._handle_client_hello(_hello(machine, ctx, mode="rebuild"))
        assert [event.ask_id for event in transport.sent if isinstance(event, AskUser)] == ["ask-replacement"]
        await _answer(machine, ctx, "ask-replacement")
        assert await second == "A"
        assert ctx.active_ask is None

    asyncio.run(run())


@pytest.mark.parametrize("btw", [False, True])
def test_idle_question_is_busy_for_deletion_and_cannot_be_evicted(btw):
    async def run():
        machine, transport = _mk_machine()
        machine.cfg.max_concurrent_sessions = 1
        ctx = _mk_ctx("session-a", "session-a")
        ctx.engine = "codex"
        machine.sessions[ctx.key] = ctx
        task, ask_id = await _start_ask(machine, ctx, to="client-a")
        assert ctx.state == "idle" and machine._session_delete_busy(ctx)
        assert machine.focused_sid != ctx.key  # otherwise cap protects focus
        if btw:
            with pytest.raises(_BtwSpawnFailure) as rejected:
                await machine._spawn_btw(ctx, owner_client_id="client-a")
            assert rejected.value.code == "busy"
        else:
            assert await machine._spawn(resume_id=None, engine="codex") is None
            assert any(isinstance(event, Error) and event.code == "busy"
                       for event in transport.sent)
        assert machine.sessions == {ctx.key: ctx}
        assert ctx.active_ask.ask_id == ask_id
        await _answer(machine, ctx, ask_id)
        assert await task == "A"
        assert not machine._session_delete_busy(ctx)

    asyncio.run(run())
