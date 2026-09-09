"""Bounded live notification routing; never retains question/response content."""
from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from dataclasses import dataclass

from cc_remote.log import logger

log = logger("cc_remote.relay.native_push_router")


@dataclass(eq=False)
class _QuestionReservation:
    key: tuple[str, str, str]
    deadline: float | None = None


class NativePushRouter:
    def __init__(self, dispatcher, *, capacity: int = 256):
        if capacity < 1:
            raise ValueError("native push router capacity must be positive")
        self.dispatcher = dispatcher
        self._queue: asyncio.Queue[tuple] = asyncio.Queue(maxsize=capacity)
        # Admitting attention also reserves its eventual close slot. If these
        # slots fill, drop NEW attention instead of losing an accepted close.
        # Keys contain identifiers only; no question/options/answer is retained.
        self._questions: dict[tuple[str, str, str], _QuestionReservation] = {}
        self._closing: set[tuple[str, str, str]] = set()
        self._overflow_closures: OrderedDict[tuple[str, str, str], _QuestionReservation] = OrderedDict()
        self._capacity = capacity
        self._idle = asyncio.Event()
        self._idle.set()
        self._worker: asyncio.Task | None = None
        self._generations: OrderedDict[str, str] = OrderedDict()
        self._contexts: OrderedDict[tuple[str, str], dict] = OrderedDict()

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None

    async def wait_idle(self) -> None:
        # Queue.join alone does not include reserved overflow closes/retries.
        await self._idle.wait()

    def _prune_questions(self) -> None:
        now = asyncio.get_running_loop().time()
        for key, question in list(self._questions.items()):
            if question.deadline is not None and question.deadline <= now and key not in self._closing:
                self._questions.pop(key, None)

    def _closed(self, key: tuple[str, str, str], question: _QuestionReservation) -> None:
        if self._questions.get(key) is question:
            self._questions.pop(key, None)
        self._closing.discard(key)
        self._overflow_closures.pop(key, None)

    def _mark_idle(self) -> None:
        if self._queue.empty() and not self._overflow_closures:
            self._idle.set()

    def _remember(self, machine_id: str, sid: str, engine: str, space: str) -> None:
        if engine not in {"claude", "codex"} or space not in {"code", "work"}:
            return
        key = (machine_id, sid)
        self._contexts[key] = {"sid": sid, "engine": engine, "space": space}
        self._contexts.move_to_end(key)
        while len(self._contexts) > 4096:
            self._contexts.popitem(last=False)

    async def observe(self, machine_id: str, msg: object) -> None:
        kind = getattr(msg, "type", "")
        if kind == "hello" and getattr(msg, "role", None) == "wrapper":
            generation = getattr(msg, "wrapper_generation", None)
            if generation and self._generations.get(machine_id) != generation:
                self._contexts = OrderedDict((key, value) for key, value in self._contexts.items()
                                             if key[0] != machine_id)
                self._generations[machine_id] = generation
                self._generations.move_to_end(machine_id)
                while len(self._generations) > 256:
                    old_machine, _ = self._generations.popitem(last=False)
                    self._contexts = OrderedDict((key, value) for key, value in self._contexts.items()
                                                 if key[0] != old_machine)
            return
        if kind == "session_list":
            for session in getattr(msg, "sessions", ()):
                self._remember(machine_id, session.session_id,
                               session.engine or getattr(msg, "engine", ""), session.space)
            return
        if kind == "session_rekey":
            context = self._contexts.pop((machine_id, getattr(msg, "old_key", "")), None)
            if context:
                self._remember(machine_id, msg.session_id, context["engine"], context["space"])
            for key, question in list(self._questions.items()):
                if key[:2] != (machine_id, msg.old_key):
                    continue
                new_key = (machine_id, msg.session_id, key[2])
                self._questions.pop(key)
                question.key = new_key
                self._questions[new_key] = question
                if key in self._closing:
                    self._closing.remove(key)
                    self._closing.add(new_key)
                if key in self._overflow_closures:
                    self._overflow_closures.pop(key)
                    self._overflow_closures[new_key] = question
            return
        # to marks private traffic and all per-client replay/active-ask restore.
        # seq=None restores are also never new live events. No private question
        # can become a device-wide notification through this hook.
        if getattr(msg, "to", None) or kind not in {"turn_end", "ask_user", "ask_user_closed"}:
            return
        sid = getattr(msg, "sid", None)
        seq = getattr(msg, "seq", None)
        generation = self._generations.get(machine_id)
        if not sid or seq is None or not generation:
            return
        context = self._contexts.get((machine_id, sid), {"sid": sid}).copy()
        if kind == "turn_end":
            raw = getattr(msg, "notification_context", None)
            if raw is None:
                return  # buffer/history copy, including wrapper reconnect replay
            context.update({"engine": raw.engine, "space": raw.space})
            self._remember(machine_id, sid, raw.engine, raw.space)
            if raw.parent_session_id:
                context["parent_session_id"] = raw.parent_session_id
            result = getattr(msg, "result", None)
            subtype = getattr(result, "subtype", "").lower()
            outcome = ("interrupted" if subtype in {"error_during_execution", "interrupted", "cancelled", "canceled"}
                       else "failed" if getattr(result, "is_error", False) else "success")
            identity = getattr(msg, "turn_id", None) or str(seq)
            data = ("completion", machine_id, context, outcome)
        else:
            identity = getattr(msg, "ask_id", None)
            if not identity:
                return
            data = (kind, machine_id, context, identity)
        self._prune_questions()
        question_key = (machine_id, sid, identity)
        question = None
        if kind == "ask_user_closed":
            if question_key not in self._questions or question_key in self._closing:
                # An unadmitted question never reached this router's journal.
                # Expired reservations outlive the dispatcher's 5-minute TTL.
                return
            question = self._questions[question_key]
            self._closing.add(question_key)
            if self._queue.full():
                # At most one close per admitted question: this cannot exceed
                # capacity, even under arbitrary close/duplicate event floods.
                self._overflow_closures[question_key] = question
                self._idle.clear()
                return
        elif kind == "ask_user":
            if question_key in self._questions:
                return
            if len(self._questions) >= self._capacity:
                log.warning("native push attention slots at capacity")
                return
            question = _QuestionReservation(question_key)
        event_id = hashlib.sha256(f"{machine_id}\0{generation}\0{sid}\0{kind}\0{identity}".encode()).hexdigest()
        try:
            self._queue.put_nowait((*data, event_id, question))
            if kind == "ask_user":
                self._questions[question_key] = question
            self._idle.clear()
        except asyncio.QueueFull:
            log.warning("native push event queue at capacity")

    async def _run(self) -> None:
        while True:
            if self._overflow_closures:
                key = next(iter(self._overflow_closures))
                question = self._overflow_closures[key]
                try:
                    await self.dispatcher.close_question(key[0], key[2])
                except Exception:
                    # Keep the reserved close through transient storage failure.
                    # One worker retries; observe never waits for DB/network and
                    # no subsequent attention can overtake this tombstone.
                    log.warning("native push question close will retry")
                    await asyncio.sleep(1)
                    continue
                self._closed(question.key, question)
                self._mark_idle()
                continue
            kind, machine_id, context, value, event_id, question = await self._queue.get()
            key = question.key if question is not None else (machine_id, context["sid"], value)
            try:
                if kind == "completion":
                    await self.dispatcher.notify_turn_end(machine_id, outcome=value, context=context, event_id=event_id)
                elif kind == "ask_user":
                    if self._questions.get(key) is question:
                        await self.dispatcher.notify_ask_user(machine_id, context={**context, "sid": key[1]}, ask_id=value, event_id=event_id)
                else:
                    await self.dispatcher.close_question(machine_id, value)
                    self._closed(question.key, question)
            except Exception:
                if kind == "ask_user_closed":
                    self._overflow_closures[question.key] = question
                    log.warning("native push question close will retry")
                else:
                    log.warning("native push event could not be persisted")
            finally:
                if kind == "ask_user" and self._questions.get(question.key) is question:
                    # Start retention AFTER a possibly slow enqueue/fan-out.
                    # One hour exceeds the dispatcher's 300-second attention
                    # lifetime; unanswered/dropped-close events cannot reserve
                    # slots forever, nor be forgotten while retries are live.
                    question.deadline = asyncio.get_running_loop().time() + 3600
                self._queue.task_done()
                self._mark_idle()
