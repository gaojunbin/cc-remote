"""Bounded attention-close routing with synthetic events and local temp SQLite."""
from __future__ import annotations

import asyncio
import sqlite3
import time

import pytest

from cc_remote.protocol import AskUser, AskUserClosed, Hello, SessionRekey, TurnEnd, TurnNotificationContext, TurnResult
from cc_remote.relay.native_push import NativePushDelivery, NativePushStore
from cc_remote.relay.native_push_router import NativePushRouter


def ask(ask_id="ask-a", *, sid="session-a", **fields):
    return AskUser(sid=sid, seq=1, ask_id=ask_id, question="Synthetic question",
                   allow_text=True, **fields)


def closed(ask_id="ask-a", *, sid="session-a", **fields):
    return AskUserClosed(sid=sid, seq=2, ask_id=ask_id, reason="answered", **fields)


def completion(turn_id="turn-a"):
    return TurnEnd(sid="session-a", seq=3, turn_id=turn_id,
                   result=TurnResult(subtype="success", duration_ms=1, is_error=False),
                   notification_context=TurnNotificationContext(engine="codex", space="code"))


class JournalDispatcher:
    def __init__(self, path):
        self.store = NativePushStore(str(path))
        self.calls = []
        self.block_completions = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.close_failures = 0

    async def notify_turn_end(self, machine_id, **fields):
        self.calls.append(("completion", machine_id, fields["event_id"]))
        if self.block_completions:
            self.entered.set()
            await self.release.wait()

    async def notify_ask_user(self, machine_id, **fields):
        self.calls.append(("attention", machine_id, fields["ask_id"]))
        await self.store.enqueue(NativePushDelivery(
            fields["event_id"], "synthetic-subject", "synthetic-installation",
            "synthetic-revision", machine_id, "attention", fields["ask_id"],
            b'{"synthetic":true}', time.time() + 300,
        ))

    async def close_question(self, machine_id, ask_id):
        self.calls.append(("close", machine_id, ask_id))
        if self.close_failures:
            self.close_failures -= 1
            raise OSError("Synthetic storage failure")
        await self.store.close_question(machine_id, ask_id)

    def rows(self):
        with sqlite3.connect(self.store.path) as db:
            return db.execute("SELECT machine_id,ask_id,status FROM native_deliveries").fetchall()


async def prepare(tmp_path, *, capacity=2):
    target = JournalDispatcher(tmp_path / "native.sqlite3")
    router = NativePushRouter(target, capacity=capacity)
    await router.observe("mac", Hello(role="wrapper", wrapper_generation="generation-a"))
    return router, target


@pytest.mark.asyncio
async def test_full_queue_close_precedes_and_retires_queued_attention(tmp_path):
    router, target = await prepare(tmp_path, capacity=1)
    await router.observe("mac", ask())
    assert router._queue.full()
    await router.observe("mac", closed())
    assert len(router._overflow_closures) == 1
    await router.start()
    try:
        await asyncio.wait_for(router.wait_idle(), timeout=2)
        assert target.calls == [("close", "mac", "ask-a")]
        assert target.rows() == []
        assert not router._questions and not router._closing
        # A duplicate live event after close still cannot enter the real journal:
        # the durable tombstone, rather than an unbounded memory set, rejects it.
        await router.observe("mac", ask())
        await asyncio.wait_for(router.wait_idle(), timeout=2)
        assert target.rows() == []
    finally:
        await router.close()


@pytest.mark.asyncio
async def test_slow_worker_full_queue_keeps_all_admitted_closes_and_coalesces_flood(tmp_path):
    router, target = await prepare(tmp_path)
    await router.start()
    try:
        await router.observe("mac", ask("ask-a"))
        await router.observe("mac", ask("ask-b"))
        await asyncio.wait_for(router.wait_idle(), timeout=2)
        assert len(target.rows()) == 2
        target.block_completions = True
        await router.observe("mac", completion("blocking"))
        await asyncio.wait_for(target.entered.wait(), timeout=2)
        worker = router._worker
        for turn in ("queued-1", "queued-2"):
            await router.observe("mac", completion(turn))
        assert router._queue.full()
        for index in range(100):
            await asyncio.wait_for(router.observe("mac", closed("ask-a")), timeout=0.1)
            await router.observe("mac", closed("ask-b"))
            await router.observe("mac", closed(f"never-admitted-{index}"))
            await router.observe("mac", ask(f"over-capacity-{index}"))
        assert router._worker is worker
        assert len(router._questions) == len(router._closing) == len(router._overflow_closures) == 2
        assert router._queue.qsize() == 2 and not router._idle.is_set()
        target.release.set()
        await asyncio.wait_for(router.wait_idle(), timeout=2)
        assert sorted(target.rows()) == [("mac", "ask-a", "cancelled"), ("mac", "ask-b", "cancelled")]
        kinds = [call[0] for call in target.calls]
        assert kinds == ["attention", "attention", "completion", "close", "close", "completion", "completion"]
        assert not router._questions and not router._overflow_closures
    finally:
        target.release.set()
        await router.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow", [False, True])
async def test_failed_close_keeps_reservation_and_idle_waits_for_persistence(tmp_path, overflow):
    router, target = await prepare(tmp_path, capacity=1 if overflow else 2)
    target.close_failures = 1
    await router.observe("mac", ask())
    await router.observe("mac", closed())
    await router.start()
    try:
        await asyncio.wait_for(router.wait_idle(), timeout=3)
        assert target.close_failures == 0
        assert [call for call in target.calls if call[0] == "close"] == [
            ("close", "mac", "ask-a"), ("close", "mac", "ask-a"),
        ]
        assert all(row[2] == "cancelled" for row in target.rows())
        assert not router._questions and not router._closing and not router._overflow_closures
    finally:
        await router.close()


@pytest.mark.asyncio
async def test_close_scope_and_private_restore_cannot_cancel_another_question(tmp_path):
    router, target = await prepare(tmp_path, capacity=1)
    await router.observe("other-mac", Hello(role="wrapper", wrapper_generation="generation-b"))
    await router.observe("mac", ask())
    await router.observe("other-mac", closed())
    await router.observe("mac", closed(sid="other-session"))
    await router.observe("mac", closed(to="private-client"))
    await router.observe("mac", closed().model_copy(update={"seq": None}))
    assert not router._closing and not router._overflow_closures
    await router.start()
    try:
        await asyncio.wait_for(router.wait_idle(), timeout=2)
        assert target.rows() == [("mac", "ask-a", "pending")]
        await router.observe("mac", closed())
        await asyncio.wait_for(router.wait_idle(), timeout=2)
        assert target.rows() == [("mac", "ask-a", "cancelled")]
    finally:
        await router.close()


@pytest.mark.asyncio
async def test_admission_reserves_close_capacity_until_closed_then_accepts_new_attention(tmp_path):
    router, target = await prepare(tmp_path, capacity=1)
    await router.start()
    try:
        await router.observe("mac", ask())
        await asyncio.wait_for(router.wait_idle(), timeout=2)
        assert router._queue.empty() and len(router._questions) == 1
        await router.observe("mac", ask("ask-b"))
        await asyncio.wait_for(router.wait_idle(), timeout=2)
        assert target.rows() == [("mac", "ask-a", "pending")]
        await router.observe("mac", closed())
        await asyncio.wait_for(router.wait_idle(), timeout=2)
        await router.observe("mac", ask("ask-b"))
        await asyncio.wait_for(router.wait_idle(), timeout=2)
        assert sorted(target.rows()) == [("mac", "ask-a", "cancelled"), ("mac", "ask-b", "pending")]
        assert len(router._questions) == 1
    finally:
        await router.close()


@pytest.mark.parametrize("capacity", [0, -1])
def test_invalid_capacity_cannot_create_unbounded_queue(capacity):
    with pytest.raises(ValueError):
        NativePushRouter(object(), capacity=capacity)


@pytest.mark.asyncio
async def test_rekey_preserves_reserved_close_for_queued_question(tmp_path):
    router, target = await prepare(tmp_path, capacity=1)
    await router.observe("mac", ask(sid="tmp-session"))
    await router.observe("mac", SessionRekey(old_key="tmp-session", session_id="real-session"))
    await router.observe("mac", closed(sid="real-session"))
    assert list(router._overflow_closures) == [("mac", "real-session", "ask-a")]
    await router.start()
    try:
        await asyncio.wait_for(router.wait_idle(), timeout=2)
        assert target.calls == [("close", "mac", "ask-a")]
        assert target.rows() == [] and not router._questions
    finally:
        await router.close()
