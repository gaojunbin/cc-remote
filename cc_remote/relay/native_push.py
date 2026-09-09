"""Native APNs delivery: private durable routing, live-session authorization.

The database is not an authentication authority. Every delivery (including a
retry after restart) requires the relay's live session/device authorization.
Only fixed alert text and bounded navigation identifiers reach APNs.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, replace
from contextvars import ContextVar
from pathlib import Path
from typing import Awaitable, Callable, Mapping

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from cc_remote.config import valid_machine_id
from cc_remote.log import logger
from cc_remote.relay.origins import canonical_https_origin

log = logger("cc_remote.relay.native_push")

_HOSTS = {
    "sandbox": "https://api.sandbox.push.apple.com",
    "production": "https://api.push.apple.com",
}
_KINDS = {"completion", "attention"}
_REASONS = {
    "BadDeviceToken", "DeviceTokenNotForTopic", "Forbidden", "ExpiredToken",
    "Unregistered", "PayloadTooLarge", "TooManyProviderTokenUpdates",
    "TooManyRequests", "InternalServerError", "ServiceUnavailable", "Shutdown",
    "ExpiredProviderToken", "InvalidProviderToken", "BadTopic", "MissingTopic",
}
_APNS_HTTP_CONTEXT: ContextVar[bool] = ContextVar("cc_remote_apns_http", default=False)


class _APNsRequestLogFilter(logging.Filter):
    """HTTPX logs request URLs at INFO; an APNs URL contains a device token."""

    _device_url = re.compile(
        r"(https://api(?:\.sandbox)?\.push\.apple\.com(?::443)?/3/device/)[^\s\"'<>]+",
        re.IGNORECASE,
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = self._device_url.sub(r"\1<redacted>", message)
        if redacted != message:
            # Replace args too: structured handlers must not retain the original
            # URL object even if their displayed message has already been cleaned.
            record.msg, record.args = redacted, ()
        return True


class _APNsHeaderLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # HPACK logs both header-table values and the complete recoverable
        # encoded block. String redaction cannot safely scrub the binary form.
        return not (_APNS_HTTP_CONTEXT.get() and record.levelno == logging.DEBUG)


def _protect_apns_transport_logs() -> None:
    request_logger = logging.getLogger("httpx")
    if not any(isinstance(item, _APNsRequestLogFilter) for item in request_logger.filters):
        request_logger.addFilter(_APNsRequestLogFilter())
    for name in ("hpack.hpack", "hpack.table"):
        header_logger = logging.getLogger(name)
        if not any(isinstance(item, _APNsHeaderLogFilter) for item in header_logger.filters):
            header_logger.addFilter(_APNsHeaderLogFilter())


def _identifier(value: str, limit: int = 256) -> bool:
    return (isinstance(value, str) and 0 < len(value) <= limit
            and all(ord(char) >= 32 and ord(char) != 127 for char in value))


def _token(value: str) -> bool:
    # Apple explicitly says not to assume a fixed token length.
    return (isinstance(value, str) and 2 <= len(value) <= 1024
            and len(value) % 2 == 0 and re.fullmatch(r"[0-9a-fA-F]+", value) is not None)


@dataclass(frozen=True)
class NativePushInstallation:
    installation_id: str
    subject: str
    session_jti: str
    expires_at: float
    device_token: str
    machine_ids: tuple[str, ...]
    topic: str
    environment: str
    client_id: str
    event_kinds: tuple[str, ...] = ("completion", "attention")
    privacy: str = "generic"
    revision: str = ""
    updated_at: float = 0

    def validate(self) -> None:
        try:
            valid_id = str(uuid.UUID(self.installation_id)) == self.installation_id.lower()
        except (ValueError, AttributeError, TypeError):
            valid_id = False
        if not (valid_id and _identifier(self.subject, 128)
                and _identifier(self.session_jti, 128) and len(self.session_jti) >= 16
                and isinstance(self.expires_at, (int, float))
                and not isinstance(self.expires_at, bool) and math.isfinite(self.expires_at)
                and _token(self.device_token) and self.environment in _HOSTS
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,254}", self.topic)
                and _identifier(self.client_id, 128) and self.privacy == "generic"
                and 1 <= len(self.machine_ids) <= 64
                and len(set(self.machine_ids)) == len(self.machine_ids)
                and all(valid_machine_id(item) for item in self.machine_ids)
                and 1 <= len(self.event_kinds) <= 2
                and len(set(self.event_kinds)) == len(self.event_kinds)
                and set(self.event_kinds) <= _KINDS):
            raise ValueError("invalid native push installation")


@dataclass(frozen=True)
class APNsResponse:
    status: int
    reason: str | None = None
    timestamp: float | None = None  # APNs Unregistered timestamp: milliseconds.
    retry_after: float | None = None


APNsSender = Callable[[str, Mapping[str, str], bytes], Awaitable[APNsResponse]]


class APNsProvider:
    """ES256 JWT cache and one pooled, HTTP/2-only Apple connection transport."""

    def __init__(
        self, team_id: str, key_id: str, topic: str, private_key_pem: bytes, *,
        sender: APNsSender | None = None, clock: Callable[[], float] = time.time,
        environments: tuple[str, ...] = ("sandbox", "production"),
    ) -> None:
        if (not re.fullmatch(r"[A-Z0-9]{10}", team_id)
                or not re.fullmatch(r"[A-Z0-9]{10}", key_id)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,254}", topic)
                or not environments or not set(environments) <= _HOSTS.keys()):
            raise ValueError("invalid APNs provider configuration")
        try:
            key = serialization.load_pem_private_key(private_key_pem, password=None)
        except (ValueError, TypeError):
            raise ValueError("invalid APNs signing key") from None
        if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
            raise ValueError("APNs signing key must use P-256")
        self.team_id, self.key_id, self.topic = team_id, key_id, topic
        self.environments = tuple(environments)
        self._key, self._clock = key, clock
        self._sender = sender or self._http_send
        self._jwt = ""
        self._issued_at = 0
        self._client = None
        _protect_apns_transport_logs()

    def _provider_token(self) -> str:
        now = int(self._clock())
        if self._jwt and 0 <= now - self._issued_at < 50 * 60:
            return self._jwt

        def encode(value: object) -> str:
            return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).rstrip(b"=").decode()

        unsigned = (encode({"alg": "ES256", "kid": self.key_id}) + "."
                    + encode({"iss": self.team_id, "iat": now}))
        der = self._key.sign(unsigned.encode(), ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        signature = base64.urlsafe_b64encode(r.to_bytes(32, "big") + s.to_bytes(32, "big")).rstrip(b"=").decode()
        self._jwt, self._issued_at = unsigned + "." + signature, now
        return self._jwt

    async def send(
        self, device_token: str, payload: bytes, *, environment: str,
        notification_id: str, expires_at: float,
    ) -> APNsResponse:
        if not _token(device_token) or environment not in self.environments:
            raise ValueError("invalid APNs destination")
        if len(payload) > 4096 or not isinstance(json.loads(payload), dict):
            raise ValueError("invalid APNs payload")
        if str(uuid.UUID(notification_id)) != notification_id:
            raise ValueError("invalid APNs notification identifier")
        headers = {
            "authorization": "bearer " + self._provider_token(),
            "apns-topic": self.topic, "apns-push-type": "alert", "apns-priority": "10",
            "apns-id": notification_id, "apns-collapse-id": notification_id,
            "apns-expiration": str(max(0, int(expires_at))),
            "content-type": "application/json",
        }
        # No caller-supplied host, URL, redirect or proxy can receive a provider JWT.
        return await self._sender(_HOSTS[environment] + "/3/device/" + device_token.lower(), headers, payload)

    async def _http_send(self, url: str, headers: Mapping[str, str], payload: bytes) -> APNsResponse:
        context = _APNS_HTTP_CONTEXT.set(True)
        try:
            return await self._http_send_in_context(url, headers, payload)
        finally:
            _APNS_HTTP_CONTEXT.reset(context)

    async def _http_send_in_context(self, url: str, headers: Mapping[str, str], payload: bytes) -> APNsResponse:
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(
                http2=True, http1=False, follow_redirects=False, trust_env=False,
                timeout=httpx.Timeout(10), limits=httpx.Limits(max_connections=2),
            )
        try:
            async with self._client.stream("POST", url, headers=headers, content=payload) as response:
                if response.http_version != "HTTP/2":
                    return APNsResponse(0)
                body = b""
                async for chunk in response.aiter_bytes():
                    body += chunk
                    if len(body) > 4096:
                        return APNsResponse(response.status_code)
                try:
                    fields = json.loads(body) if body else {}
                except (ValueError, UnicodeDecodeError):
                    fields = {}
                if not isinstance(fields, dict):
                    fields = {}
                reason = fields.get("reason")
                timestamp = fields.get("timestamp")
                if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                        or not math.isfinite(timestamp) or timestamp < 0):
                    timestamp = None
                try:
                    retry_after = float(response.headers.get("retry-after", ""))
                    if not math.isfinite(retry_after) or retry_after < 0:
                        retry_after = None
                except ValueError:
                    retry_after = None
                return APNsResponse(response.status_code, reason if isinstance(reason, str) and reason in _REASONS else None,
                                    timestamp, retry_after)
        except httpx.TransportError:
            return APNsResponse(0)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@dataclass(frozen=True)
class NativePushDelivery:
    delivery_id: str
    subject: str
    installation_id: str
    revision: str
    machine_id: str
    event_kind: str
    ask_id: str | None
    payload: bytes
    expires_at: float
    attempts: int = 0


class NativePushStore:
    """Bounded SQLite installation registry and delivery journal, mode 0600."""

    def __init__(self, path: str, *, clock: Callable[[], float] = time.time,
                 max_installations: int = 2048, max_per_subject: int = 64,
                 max_deliveries: int = 10000) -> None:
        if min(max_installations, max_per_subject, max_deliveries) < 1:
            raise ValueError("native push limits must be positive")
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        os.fchmod(fd, 0o600)
        os.close(fd)
        self._clock = clock
        self.max_installations, self.max_per_subject, self.max_deliveries = max_installations, max_per_subject, max_deliveries
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS native_installations (
                    subject TEXT NOT NULL, installation_id TEXT NOT NULL,
                    session_jti TEXT NOT NULL, expires_at REAL NOT NULL,
                    device_token TEXT NOT NULL, topic TEXT NOT NULL,
                    environment TEXT NOT NULL, client_id TEXT NOT NULL,
                    machine_ids TEXT NOT NULL, event_kinds TEXT NOT NULL,
                    revision TEXT NOT NULL, updated_at REAL NOT NULL,
                    PRIMARY KEY(subject, installation_id),
                    UNIQUE(topic, environment, device_token)
                );
                CREATE TABLE IF NOT EXISTS native_deliveries (
                    delivery_id TEXT PRIMARY KEY, subject TEXT NOT NULL,
                    installation_id TEXT NOT NULL, revision TEXT NOT NULL,
                    machine_id TEXT NOT NULL, event_kind TEXT NOT NULL,
                    ask_id TEXT, payload BLOB NOT NULL, expires_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending', next_attempt REAL NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS native_deliveries_due ON native_deliveries(status, next_attempt);
                CREATE TABLE IF NOT EXISTS native_closed_questions (
                    machine_id TEXT NOT NULL, ask_id TEXT NOT NULL,
                    expires_at REAL NOT NULL, PRIMARY KEY(machine_id, ask_id)
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        # DELETE journal inherits the private database's permissions and avoids
        # leaving WAL/SHM sidecars in a pre-existing, broader parent directory.
        db.execute("PRAGMA journal_mode=DELETE")
        return db

    @staticmethod
    def _installation(row: sqlite3.Row) -> NativePushInstallation:
        return NativePushInstallation(
            subject=row["subject"], installation_id=row["installation_id"],
            session_jti=row["session_jti"], expires_at=row["expires_at"],
            device_token=row["device_token"], topic=row["topic"], environment=row["environment"],
            client_id=row["client_id"], machine_ids=tuple(json.loads(row["machine_ids"])),
            event_kinds=tuple(json.loads(row["event_kinds"])), revision=row["revision"], updated_at=row["updated_at"],
        )

    def _prune(self, db: sqlite3.Connection) -> None:
        now = self._clock()
        db.execute("DELETE FROM native_installations WHERE expires_at<=?", (now,))
        db.execute("DELETE FROM native_closed_questions WHERE expires_at<=?", (now,))
        db.execute("UPDATE native_deliveries SET status='expired' WHERE expires_at<=? AND status IN ('pending','retry','sending')", (now,))
        db.execute("DELETE FROM native_deliveries WHERE created_at<? AND status NOT IN ('pending','retry','sending')", (now - 7 * 86400,))

    async def upsert(self, installation: NativePushInstallation) -> NativePushInstallation:
        installation.validate()
        return await asyncio.to_thread(self._upsert, installation)

    def _upsert(self, value: NativePushInstallation) -> NativePushInstallation:
        now = self._clock()
        if value.expires_at <= now:
            raise ValueError("expired native push session")
        value = replace(value, installation_id=value.installation_id.lower(), device_token=value.device_token.lower(),
                        machine_ids=tuple(sorted(value.machine_ids)), event_kinds=tuple(sorted(value.event_kinds)),
                        revision="", updated_at=0)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._prune(db)
            existing = db.execute("SELECT * FROM native_installations WHERE subject=? AND installation_id=?", (value.subject, value.installation_id)).fetchone()
            if existing is not None:
                old = self._installation(existing)
                normalized = replace(old, machine_ids=tuple(sorted(old.machine_ids)),
                                     event_kinds=tuple(sorted(old.event_kinds)), revision="", updated_at=0)
                # Foreground/token callbacks may repeat the same PUT. Rotating
                # this revision would orphan already queued/retrying deliveries.
                # Keep the original registration timestamp for 410 comparisons.
                if normalized == value:
                    return old
            if existing is None:
                total = db.execute("SELECT COUNT(*) FROM native_installations").fetchone()[0]
                own = db.execute("SELECT COUNT(*) FROM native_installations WHERE subject=?", (value.subject,)).fetchone()[0]
                if total >= self.max_installations or own >= self.max_per_subject:
                    raise ValueError("native push installation limit reached")
            replaced = db.execute(
                "SELECT subject,installation_id,revision FROM native_installations "
                "WHERE (subject=? AND installation_id=?) OR (topic=? AND environment=? AND device_token=?)",
                (value.subject, value.installation_id, value.topic, value.environment, value.device_token),
            ).fetchall()
            for row in replaced:
                db.execute(
                    "UPDATE native_deliveries SET status='cancelled' WHERE subject=? AND installation_id=? AND revision=? AND status IN ('pending','retry','sending')",
                    (row["subject"], row["installation_id"], row["revision"]),
                )
            value = replace(value, revision=str(uuid.uuid4()), updated_at=now)
            # Token possession permits transferring that exact APNs endpoint to
            # a new login. Knowing a public installation UUID alone does not:
            # another subject has a different primary key and cannot replace it.
            db.execute("DELETE FROM native_installations WHERE topic=? AND environment=? AND device_token=?", (value.topic, value.environment, value.device_token))
            db.execute("INSERT OR REPLACE INTO native_installations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                value.subject, value.installation_id, value.session_jti, value.expires_at,
                value.device_token, value.topic, value.environment, value.client_id,
                json.dumps(value.machine_ids), json.dumps(value.event_kinds), value.revision, value.updated_at,
            ))
        return value

    async def get_installation(self, subject: str, installation_id: str) -> NativePushInstallation | None:
        def get():
            with self._connect() as db:
                row = db.execute("SELECT * FROM native_installations WHERE subject=? AND installation_id=? AND expires_at>?", (subject, installation_id, self._clock())).fetchone()
            return self._installation(row) if row else None
        return await asyncio.to_thread(get)

    async def for_machine(self, machine_id: str) -> list[NativePushInstallation]:
        def get():
            with self._connect() as db:
                self._prune(db)
                rows = db.execute("SELECT * FROM native_installations").fetchall()
            return [value for row in rows if machine_id in (value := self._installation(row)).machine_ids]
        return await asyncio.to_thread(get)

    async def remove_installation(self, subject: str, installation_id: str, session_jti: str | None = None) -> int:
        def remove():
            with self._connect() as db:
                sql, args = "DELETE FROM native_installations WHERE subject=? AND installation_id=?", [subject, installation_id]
                if session_jti is not None:
                    sql += " AND session_jti=?"
                    args.append(session_jti)
                return db.execute(sql, args).rowcount
        return await asyncio.to_thread(remove)

    async def remove_session(self, session_jti: str) -> int:
        def remove():
            with self._connect() as db:
                return db.execute("DELETE FROM native_installations WHERE session_jti=?", (session_jti,)).rowcount
        return await asyncio.to_thread(remove)

    async def remove_machine(self, machine_id: str) -> int:
        def remove():
            count = 0
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                for row in db.execute("SELECT * FROM native_installations").fetchall():
                    value = self._installation(row)
                    if machine_id not in value.machine_ids:
                        continue
                    count += 1
                    remaining = tuple(item for item in value.machine_ids if item != machine_id)
                    if remaining:
                        db.execute("UPDATE native_installations SET machine_ids=? WHERE subject=? AND installation_id=?", (json.dumps(remaining), value.subject, value.installation_id))
                    else:
                        db.execute("DELETE FROM native_installations WHERE subject=? AND installation_id=?", (value.subject, value.installation_id))
                db.execute("UPDATE native_deliveries SET status='cancelled' WHERE machine_id=? AND status IN ('pending','retry','sending')", (machine_id,))
            return count
        return await asyncio.to_thread(remove)

    async def invalidate(self, value: NativePushInstallation, timestamp: float | None = None) -> int:
        def remove():
            with self._connect() as db:
                sql = "DELETE FROM native_installations WHERE subject=? AND installation_id=? AND revision=? AND device_token=?"
                args: list[object] = [value.subject, value.installation_id, value.revision, value.device_token]
                if timestamp is not None:
                    sql += " AND updated_at<=?"
                    args.append(timestamp / 1000)
                return db.execute(sql, args).rowcount
        return await asyncio.to_thread(remove)

    async def enqueue(self, value: NativePushDelivery) -> bool:
        def insert():
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                self._prune(db)
                if (value.event_kind == "attention" and db.execute(
                    "SELECT 1 FROM native_closed_questions WHERE machine_id=? AND ask_id=? AND expires_at>?",
                    (value.machine_id, value.ask_id, self._clock()),
                ).fetchone()):
                    return False
                if db.execute("SELECT 1 FROM native_deliveries WHERE delivery_id=?", (value.delivery_id,)).fetchone():
                    return False
                total = db.execute("SELECT COUNT(*) FROM native_deliveries").fetchone()[0]
                if total >= self.max_deliveries:
                    # Keep recent terminal receipts for dedupe, but they must
                    # never starve new notifications for the seven-day TTL.
                    # Reclaim only the oldest terminal rows, never active work.
                    db.execute(
                        "DELETE FROM native_deliveries WHERE delivery_id IN ("
                        "SELECT delivery_id FROM native_deliveries "
                        "WHERE status NOT IN ('pending','retry','sending') "
                        "ORDER BY created_at,rowid LIMIT ?)",
                        (total - self.max_deliveries + 1,),
                    )
                    if db.execute("SELECT COUNT(*) FROM native_deliveries").fetchone()[0] >= self.max_deliveries:
                        return False
                db.execute("INSERT INTO native_deliveries (delivery_id,subject,installation_id,revision,machine_id,event_kind,ask_id,payload,expires_at,next_attempt,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                    value.delivery_id, value.subject, value.installation_id, value.revision,
                    value.machine_id, value.event_kind, value.ask_id, value.payload,
                    value.expires_at, self._clock(), self._clock(),
                ))
            return True
        return await asyncio.to_thread(insert)

    async def claim_due(self, limit: int = 16) -> list[NativePushDelivery]:
        def claim():
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                self._prune(db)
                rows = db.execute("SELECT * FROM native_deliveries WHERE status IN ('pending','retry','sending') AND next_attempt<=? AND expires_at>? ORDER BY created_at LIMIT ?", (self._clock(), self._clock(), min(64, max(1, limit)))).fetchall()
                for row in rows:
                    db.execute("UPDATE native_deliveries SET status='sending',attempts=attempts+1,next_attempt=? WHERE delivery_id=?", (self._clock() + 120, row["delivery_id"]))
            return [NativePushDelivery(row["delivery_id"], row["subject"], row["installation_id"], row["revision"], row["machine_id"], row["event_kind"], row["ask_id"], row["payload"], row["expires_at"], row["attempts"] + 1) for row in rows]
        return await asyncio.to_thread(claim)

    async def finish(self, delivery_id: str, status: str, *, next_attempt: float = 0) -> None:
        if status not in {"delivered", "failed", "cancelled", "retry"}:
            raise ValueError("invalid native delivery status")
        def update():
            with self._connect() as db:
                db.execute("UPDATE native_deliveries SET status=?,next_attempt=? WHERE delivery_id=? AND status='sending'", (status, next_attempt, delivery_id))
        await asyncio.to_thread(update)

    async def is_pending(self, delivery_id: str) -> bool:
        def get():
            with self._connect() as db:
                return db.execute("SELECT 1 FROM native_deliveries WHERE delivery_id=? AND status='sending' AND expires_at>?", (delivery_id, self._clock())).fetchone() is not None
        return await asyncio.to_thread(get)

    async def close_question(self, machine_id: str, ask_id: str) -> None:
        def close():
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                self._prune(db)
                db.execute("INSERT OR REPLACE INTO native_closed_questions VALUES (?,?,?)", (machine_id, ask_id, self._clock() + 3600))
                db.execute("DELETE FROM native_closed_questions WHERE rowid NOT IN (SELECT rowid FROM native_closed_questions ORDER BY expires_at DESC LIMIT ?)", (self.max_deliveries,))
                db.execute("UPDATE native_deliveries SET status='cancelled' WHERE machine_id=? AND ask_id=? AND event_kind='attention' AND status IN ('pending','retry','sending')", (machine_id, ask_id))
        await asyncio.to_thread(close)


NativePushSessionCheck = Callable[[NativePushInstallation, str], Awaitable[bool]]
NativePushOrigin = Callable[[NativePushInstallation], Awaitable[str | None]]


class NativePushDispatcher:
    def __init__(self, store: NativePushStore, provider: APNsProvider, *, origin: str,
                 session_active: NativePushSessionCheck, clock: Callable[[], float] = time.time,
                 max_attempts: int = 3, origin_for: NativePushOrigin | None = None) -> None:
        if not (canonical_https_origin(origin) or (origin == "auto" and callable(origin_for))):
            raise ValueError("native push requires an exact HTTPS origin")
        if not callable(session_active) or not 1 <= max_attempts <= 5:
            raise ValueError("native push requires live session authorization")
        self.store, self.provider = store, provider
        self.origin = canonical_https_origin(origin) or "auto"
        self._origin_for = origin_for
        self._session_active, self._clock, self.max_attempts = session_active, clock, max_attempts
        self._wake = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self._drain_lock = asyncio.Lock()
        self._last_storage_warning = float("-inf")

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
        await self.provider.close()

    async def _run(self) -> None:
        while True:
            self._wake.clear()
            try:
                await self.drain_once()
            except Exception:
                # A temporary SQLite error must not permanently kill delivery.
                # Durable leases bound crash retries; don't expose exception text.
                if self._clock() - self._last_storage_warning >= 60:
                    log.warning("native push storage unavailable; delivery will retry")
                    self._last_storage_warning = self._clock()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=2)
            except asyncio.TimeoutError:
                pass

    async def notify_turn_end(self, machine_id: str, *, outcome: str,
                              context: Mapping[str, object] | None, event_id: str) -> None:
        bodies = {"success": "远程任务已完成", "failed": "远程任务执行失败", "interrupted": "远程任务已中断"}
        if outcome not in bodies:
            raise ValueError("invalid native completion outcome")
        await self._enqueue(machine_id, "completion", bodies[outcome], context, event_id)

    async def notify_ask_user(self, machine_id: str, *, context: Mapping[str, object] | None,
                              ask_id: str, event_id: str, target_client_id: str | None = None) -> None:
        if not _identifier(ask_id):
            return
        await self._enqueue(machine_id, "attention", "远程任务需要你确认", context, event_id,
                            ask_id=ask_id, target_client_id=target_client_id)

    async def close_question(self, machine_id: str, ask_id: str) -> None:
        await self.store.close_question(machine_id, ask_id)

    async def _enqueue(self, machine_id: str, kind: str, body: str,
                       context: Mapping[str, object] | None, event_id: str, *,
                       ask_id: str | None = None, target_client_id: str | None = None) -> None:
        if not valid_machine_id(machine_id) or not _identifier(event_id, 512) or not context:
            return
        sid = context.get("parent_session_id") or context.get("session_id") or context.get("sid")
        engine, space = context.get("engine"), context.get("space")
        if not _identifier(sid):
            return
        classified = (isinstance(engine, str) and engine in {"claude", "codex"}
                      and isinstance(space, str) and space in {"code", "work"})
        if kind == "completion" and not classified:
            return
        route = {"v": 1, "machine_id": machine_id,
                 "session_id": sid, "event_kind": kind, "event_id": event_id}
        if classified:
            route.update(engine=engine, space=space)
        if ask_id is not None:
            route["ask_id"] = ask_id
        for installation in await self.store.for_machine(machine_id):
            if (kind not in installation.event_kinds or installation.topic != self.provider.topic
                    or installation.environment not in self.provider.environments
                    or (target_client_id is not None and installation.client_id != target_client_id)):
                continue
            origin = (await self._origin_for(installation)
                      if self.origin == "auto" and self._origin_for else self.origin)
            if not origin or canonical_https_origin(origin) != origin:
                continue
            payload = json.dumps({"aps": {"alert": {"title": "Remote", "body": body}, "sound": "default"},
                                  "cc_remote": {**route, "origin": origin}},
                                 ensure_ascii=False, separators=(",", ":")).encode()
            if len(payload) > 4096:
                continue
            # Stable across registration refresh and replay; no raw token/jti in
            # the identifier or payload. APNs ID/collapse ID reuse bounds retry duplicates.
            identity = json.dumps([installation.subject, installation.installation_id, installation.topic,
                                   installation.environment, machine_id, kind, event_id])
            delivery_id = str(uuid.UUID(bytes=hashlib.sha256(identity.encode()).digest()[:16], version=5))
            ttl = 300 if kind == "attention" else 3600
            await self.store.enqueue(NativePushDelivery(delivery_id, installation.subject,
                installation.installation_id, installation.revision, machine_id, kind, ask_id,
                payload, min(installation.expires_at, self._clock() + ttl)))
        self._wake.set()

    async def drain_once(self) -> int:
        async with self._drain_lock:
            deliveries = await self.store.claim_due()
            # Bounded fan-out; each call also has its own transport timeout.
            for start in range(0, len(deliveries), 4):
                await asyncio.gather(*(self._deliver(item) for item in deliveries[start:start + 4]))
            return len(deliveries)

    async def _deliver(self, item: NativePushDelivery) -> None:
        installation = await self.store.get_installation(item.subject, item.installation_id)
        if (installation is None or installation.revision != item.revision
                or item.machine_id not in installation.machine_ids
                or installation.topic != self.provider.topic
                or installation.environment not in self.provider.environments):
            await self.store.finish(item.delivery_id, "cancelled")
            return
        try:
            authorized = await self._session_active(installation, item.machine_id)
        except Exception:
            authorized = False
        if not authorized:
            await self.store.invalidate(installation)
            await self.store.finish(item.delivery_id, "cancelled")
            return
        # A question may have closed or a token rotated while authorization awaited.
        current = await self.store.get_installation(item.subject, item.installation_id)
        if current != installation or not await self.store.is_pending(item.delivery_id):
            await self.store.finish(item.delivery_id, "cancelled")
            return
        if item.attempts > self.max_attempts:
            await self.store.finish(item.delivery_id, "failed")
            return
        try:
            authorized = await self._session_active(installation, item.machine_id)
        except Exception:
            authorized = False
        if not authorized:
            # Logout may revoke the in-memory registry while its slower SQLite
            # cleanup is pending. Recheck after every storage await and before
            # starting HTTP; exact revision invalidation preserves a newer login.
            await self.store.invalidate(installation)
            await self.store.finish(item.delivery_id, "cancelled")
            return
        try:
            response = await self.provider.send(installation.device_token, item.payload,
                environment=installation.environment, notification_id=item.delivery_id,
                expires_at=item.expires_at)
        except Exception:
            # No key, endpoint token, payload or remote error body is logged.
            response = APNsResponse(0)
        if response.status == 200:
            await self.store.finish(item.delivery_id, "delivered")
        elif response.status == 410:
            await self.store.invalidate(installation, response.timestamp)
            await self.store.finish(item.delivery_id, "failed")
        elif response.status in {0, 429, 500, 502, 503, 504} and response.reason != "TooManyProviderTokenUpdates":
            # Apple advises >=15 minutes for 5XX. Persist the schedule instead
            # of sleeping in the realtime relay callback or bypassing revocation.
            minimum = 900 if response.status >= 500 else 60 * (2 ** (item.attempts - 1))
            delay = max(minimum, response.retry_after or 0)
            next_attempt = self._clock() + delay
            if item.attempts < self.max_attempts and next_attempt < item.expires_at and delay <= 3600:
                await self.store.finish(item.delivery_id, "retry", next_attempt=next_attempt)
            else:
                await self.store.finish(item.delivery_id, "failed")
        else:
            await self.store.finish(item.delivery_id, "failed")
