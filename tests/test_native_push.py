"""No Apple network, production keys, model calls or user credentials."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import sqlite3
import stat
import uuid
from dataclasses import replace

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from cc_remote.relay.native_push import (
    APNsProvider, APNsResponse, NativePushDispatcher, NativePushInstallation,
    NativePushStore,
)


class Clock:
    def __init__(self):
        self.now = 1_800_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def signing_key():
    return ec.generate_private_key(ec.SECP256R1())


def _pem(key):
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _provider(key, sender, clock, **kwargs):
    return APNsProvider("TEAMID0001", "KEYID00001", "com.example.remote", _pem(key),
                        sender=sender, clock=clock, **kwargs)


def _installation(clock, **kwargs):
    value = NativePushInstallation(
        installation_id="00000000-0000-4000-a000-000000000001", subject="alice",
        session_jti="test-session-00000001", expires_at=clock() + 7200,
        device_token="ab" * 32, machine_ids=("mac", "vps"), topic="com.example.remote",
        environment="sandbox", client_id="native-client-one",
    )
    return replace(value, **kwargs)


async def _active(_installation, _machine_id):
    return True


CONTEXT = {"sid": "session-one", "engine": "codex", "space": "code"}


def _rows(store):
    with sqlite3.connect(store.path) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT * FROM native_deliveries")]


def test_provider_es256_signature_headers_cache_and_fixed_apple_hosts(signing_key):
    async def run():
        clock, requests = Clock(), []

        async def sender(url, headers, payload):
            requests.append((url, dict(headers), payload))
            return APNsResponse(200)

        provider = _provider(signing_key, sender, clock)
        identifier = str(uuid.uuid4())
        payload = b'{"aps":{"alert":{"title":"Remote","body":"Done"}}}'
        await provider.send("AB" * 17, payload, environment="sandbox", notification_id=identifier, expires_at=clock() + 300)
        url, headers, sent = requests[-1]
        assert url == "https://api.sandbox.push.apple.com/3/device/" + "ab" * 17
        assert sent == payload
        assert headers["apns-push-type"] == "alert"
        assert headers["apns-topic"] == "com.example.remote"
        assert headers["apns-id"] == headers["apns-collapse-id"] == identifier
        assert headers["apns-expiration"] == str(int(clock() + 300))
        token = headers["authorization"].removeprefix("bearer ")
        header, claims, signature = token.split(".")

        def decode(value):
            return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

        assert json.loads(decode(header)) == {"alg": "ES256", "kid": "KEYID00001"}
        assert json.loads(decode(claims)) == {"iss": "TEAMID0001", "iat": int(clock())}
        raw = decode(signature)
        assert len(raw) == 64
        signing_key.public_key().verify(encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")),
                                        (header + "." + claims).encode(), ec.ECDSA(hashes.SHA256()))
        clock.now += 49 * 60
        await provider.send("ab" * 32, payload, environment="production", notification_id=identifier, expires_at=clock() + 300)
        assert requests[-1][0].startswith("https://api.push.apple.com/")
        assert requests[-1][1]["authorization"] == "bearer " + token
        clock.now += 60
        await provider.send("ab" * 32, payload, environment="sandbox", notification_id=identifier, expires_at=clock() + 300)
        assert requests[-1][1]["authorization"] != "bearer " + token
        for environment, device_token in [("https://attacker.example", "ab"), ("sandbox", "ab/../../secrets"), ("sandbox", "a")]:
            with pytest.raises(ValueError):
                await provider.send(device_token, payload, environment=environment, notification_id=identifier, expires_at=clock() + 300)
        with pytest.raises(ValueError):
            await provider.send("ab", b" " * 4097, environment="sandbox", notification_id=identifier, expires_at=clock() + 300)
        assert len(requests) == 3
        await provider.close()

    asyncio.run(run())


def test_provider_rejects_wrong_signing_curve_and_untrusted_configuration(signing_key):
    for args in [("TEAMID0001", "KEYID00001", "bad topic", _pem(signing_key)),
                 ("TEAMID0001", "KEYID00001", "com.example.remote", b"secret-looking-invalid-key"),
                 ("TEAMID0001", "KEYID00001", "com.example.remote", _pem(ec.generate_private_key(ec.SECP384R1())))]:
        with pytest.raises(ValueError) as caught:
            APNsProvider(*args)
        assert "secret-looking" not in str(caught.value)


def test_provider_http2_response_parsing_is_bounded_and_does_not_follow_redirects(signing_key):
    async def run():
        clock = Clock()
        responses = [
            httpx.Response(410, json={"reason": "Unregistered", "timestamp": 1234567890}, extensions={"http_version": b"HTTP/2"}),
            httpx.Response(429, json={"reason": "TooManyRequests"}, headers={"retry-after": "123"}, extensions={"http_version": b"HTTP/2"}),
            httpx.Response(500, json={"reason": "PRIVATE-SERVER-ERROR", "timestamp": True}, extensions={"http_version": b"HTTP/2"}),
            httpx.Response(200, extensions={"http_version": b"HTTP/1.1"}),
            httpx.Response(302, headers={"location": "https://attacker.example"}, extensions={"http_version": b"HTTP/2"}),
        ]
        seen = []

        def handle(request):
            seen.append(request)
            return responses.pop(0)

        provider = _provider(signing_key, None, clock)
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handle), http2=True, follow_redirects=False, trust_env=False)
        results = []
        for _ in range(5):
            results.append(await provider.send("ab", b'{"aps":{}}', environment="sandbox", notification_id=str(uuid.uuid4()), expires_at=clock() + 30))
        assert results == [APNsResponse(410, "Unregistered", 1234567890),
                           APNsResponse(429, "TooManyRequests", retry_after=123),
                           APNsResponse(500), APNsResponse(0), APNsResponse(302)]
        assert len(seen) == 5 and all(request.url.host == "api.sandbox.push.apple.com" for request in seen)
        await provider.close()

    asyncio.run(run())


def test_default_httpx_info_logs_redact_only_apns_device_urls(signing_key):
    async def run():
        clock = Clock()
        request_logger = logging.getLogger("httpx")
        original_level, original_propagate = request_logger.level, request_logger.propagate
        captured = []

        class Capture(logging.Handler):
            def emit(self, record):
                captured.append((record.getMessage(), record.args))

        handler = Capture()
        request_logger.addHandler(handler)
        request_logger.setLevel(logging.INFO)
        # Keep this regression's deliberately sensitive fixture URLs away from
        # pytest/root handlers even if the protection under test regresses.
        request_logger.propagate = False
        provider = _provider(signing_key, None, clock)

        def respond(_request):
            return httpx.Response(200, extensions={"http_version": b"HTTP/2"})

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(respond), http2=True,
                                            follow_redirects=False, trust_env=False)
        sensitive_token = uuid.uuid4().hex + uuid.uuid4().hex
        try:
            # Calls the production _http_send implementation and HTTPX's real
            # INFO log path. Only the network transport is substituted.
            for environment in ("sandbox", "production"):
                await provider.send(sensitive_token, b'{"aps":{}}', environment=environment,
                                    notification_id=str(uuid.uuid4()), expires_at=clock() + 60)
            await provider._client.get("https://other.example/3/device/keep-this-visible")
            if any(sensitive_token in message or sensitive_token in str(args) for message, args in captured):
                pytest.fail("APNs device token reached the HTTPX log handler", pytrace=False)
            assert len(captured) == 3
            assert all("/3/device/<redacted>" in message and args == () for message, args in captured[:2])
            assert "https://other.example/3/device/keep-this-visible" in captured[-1][0]
            assert all("HTTP Request:" in message for message, _args in captured)
        finally:
            await provider.close()
            request_logger.removeHandler(handler)
            request_logger.setLevel(original_level)
            request_logger.propagate = original_propagate

    asyncio.run(run())


def test_real_hpack_headers_are_hidden_only_in_apns_context(signing_key):
    async def run():
        from hpack import Encoder, NeverIndexedHeaderTuple
        from httpcore import Request
        from httpcore._trace import Trace

        names = ("hpack.hpack", "hpack.table", "httpcore.http2")
        loggers = [logging.getLogger(name) for name in names]
        originals = [(item.level, item.propagate, item.handlers[:]) for item in loggers]
        captured = []

        class Capture(logging.Handler):
            def emit(self, record):
                captured.append((record.name, record.getMessage(), record.args))

        handler = Capture()
        for item in loggers:
            for original_handler in item.handlers[:]:
                item.removeHandler(original_handler)
            item.addHandler(handler)
            item.setLevel(logging.DEBUG)
            item.propagate = False

        provider = _provider(signing_key, None, Clock())
        entered, release = asyncio.Event(), asyncio.Event()
        fixture_token = uuid.uuid4().hex + uuid.uuid4().hex
        observed_authorizations = []

        async def respond(request):
            authorization = request.headers["authorization"]
            observed_authorizations.append(authorization)
            encoder = Encoder()
            # Use the real encoder, including the normally redacted sensitive
            # tuple and its still-recoverable encoded block, then force eviction.
            encoder.encode([(b":method", b"POST"), (b":path", request.url.raw_path),
                            NeverIndexedHeaderTuple(b"authorization", authorization.encode())], huffman=False)
            encoder.header_table_size = 1
            # The installed HTTP core request trace uses method-only repr;
            # verify it remains safe without suppressing useful trace messages.
            traced = Request("POST", str(request.url), headers={"authorization": authorization})
            async with Trace("send_request_headers", loggers[2], traced, {"request": traced, "stream_id": 1}):
                pass
            entered.set()
            await release.wait()
            return httpx.Response(200, extensions={"http_version": b"HTTP/2"})

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(respond), http2=True,
                                            follow_redirects=False, trust_env=False)
        try:
            sending = asyncio.create_task(provider.send(fixture_token, b'{"aps":{}}', environment="sandbox",
                                                        notification_id=str(uuid.uuid4()), expires_at=Clock()() + 60))
            await entered.wait()
            if any(name.startswith("hpack.") for name, _message, _args in captured):
                pytest.fail("APNs HPACK headers reached a DEBUG handler", pytrace=False)
            if any(secret in message or secret in str(args)
                   for secret in [fixture_token, *observed_authorizations]
                   for _name, message, args in captured):
                pytest.fail("APNs transport debug log exposed a secret", pytrace=False)
            assert any(name == "httpcore.http2" and "send_request_headers.started" in message
                       for name, message, _args in captured)
            # This task's unrelated encoding remains visible while the APNs
            # task is suspended, proving the filter is not a global debug mute.
            unrelated = Encoder()
            unrelated.encode([(b":path", b"/public-unrelated")], huffman=False)
            unrelated.header_table_size = 1
            assert any(name == "hpack.hpack" and "/public-unrelated" in message for name, message, _args in captured)
            assert any(name == "hpack.table" and "/public-unrelated" in message for name, message, _args in captured)
            release.set()
            assert (await sending).status == 200
            # Cleanup resets context even when HTTP is cancelled, so subsequent
            # non-APNs debug events in the same task remain available.
            entered.clear()
            release.clear()

            async def cancelled_send():
                try:
                    await provider.send(fixture_token, b'{"aps":{}}', environment="production",
                                        notification_id=str(uuid.uuid4()), expires_at=Clock()() + 60)
                except asyncio.CancelledError:
                    Encoder().encode([(b":path", b"/after-cancellation")], huffman=False)
                    raise

            cancelling = asyncio.create_task(cancelled_send())
            await entered.wait()
            cancelling.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelling
            assert any("/after-cancellation" in message for _name, message, _args in captured)
        finally:
            release.set()
            await provider.close()
            for item, (level, propagate, handlers) in zip(loggers, originals):
                item.removeHandler(handler)
                for original_handler in handlers:
                    item.addHandler(original_handler)
                item.setLevel(level)
                item.propagate = propagate

    asyncio.run(run())


def test_store_private_permissions_scope_rotation_expiry_and_limits(tmp_path):
    async def run():
        clock = Clock()
        store = NativePushStore(str(tmp_path / "private" / "native.sqlite3"), clock=clock,
                                max_installations=3, max_per_subject=1)
        first = await store.upsert(_installation(clock))
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
        assert first.revision and first.updated_at == clock()
        assert await store.for_machine("mac") == [first]
        assert await store.for_machine("other") == []
        with pytest.raises(ValueError, match="limit"):
            await store.upsert(_installation(clock, installation_id=str(uuid.uuid4()), device_token="cc" * 32))
        # A different account knowing only an installation UUID cannot overwrite Alice.
        bob = await store.upsert(_installation(clock, subject="bob", device_token="cc" * 32))
        assert await store.get_installation("alice", first.installation_id) == first
        assert await store.remove_installation("bob", first.installation_id, "wrong-session") == 0
        assert await store.remove_installation("bob", first.installation_id) == 1
        assert await store.get_installation("alice", first.installation_id) == first
        # Possession of the same actual token transfers its endpoint to the new login.
        bob = await store.upsert(replace(bob, device_token=first.device_token))
        assert await store.get_installation("alice", first.installation_id) is None
        assert await store.for_machine("vps") == [bob]
        assert await store.remove_machine("mac") == 1
        assert await store.for_machine("mac") == []
        assert (await store.for_machine("vps"))[0].machine_ids == ("vps",)
        clock.now += 7201
        assert await store.for_machine("vps") == []

    asyncio.run(run())


def test_store_rejects_symlink_and_invalid_registrations(tmp_path):
    target = tmp_path / "untouched"
    target.write_text("preserve me")
    link = tmp_path / "link.sqlite3"
    link.symlink_to(target)
    with pytest.raises(OSError):
        NativePushStore(str(link))
    assert target.read_text() == "preserve me"
    for overrides in [{"privacy": "session"}, {"environment": "evil"}, {"device_token": "ab" * 513},
                      {"machine_ids": ("../other",)}, {"machine_ids": ()}, {"event_kinds": ("anything",)},
                      {"installation_id": "guessable"}, {"expires_at": float("nan")}]:
        with pytest.raises(ValueError):
            _installation(Clock(), **overrides).validate()


def test_dispatch_payload_privacy_device_kind_scope_and_persistent_deduplication(tmp_path, signing_key):
    async def run():
        clock, requests = Clock(), []
        path = str(tmp_path / "native.sqlite3")
        store = NativePushStore(path, clock=clock)
        await store.upsert(_installation(clock))

        async def sender(url, headers, payload):
            requests.append((url, dict(headers), json.loads(payload)))
            return APNsResponse(200)

        provider = _provider(signing_key, sender, clock)
        dispatcher = NativePushDispatcher(store, provider, origin="https://relay.example", session_active=_active, clock=clock)
        context = {**CONTEXT, "display_name": "PRIVATE PROMPT", "question": "SECRET QUESTION", "options": ["API_KEY"], "files": ["/Users/private/file"]}
        await dispatcher.notify_turn_end("not-authorized", outcome="success", context=context, event_id="ignored")
        await dispatcher.notify_turn_end("mac", outcome="success", context=context, event_id="same-event")
        await dispatcher.notify_turn_end("mac", outcome="success", context=context, event_id="same-event")
        assert requests == []  # Hook enqueues only; it never waits for Apple.
        assert await dispatcher.drain_once() == 1
        payload = requests[0][2]
        assert payload["aps"]["alert"] == {"title": "Remote", "body": "远程任务已完成"}
        assert payload["cc_remote"] == {"v": 1, "origin": "https://relay.example", "machine_id": "mac", "session_id": "session-one", "engine": "codex", "space": "code", "event_kind": "completion", "event_id": "same-event"}
        serialized = json.dumps(payload)
        assert not any(secret in serialized for secret in ["PRIVATE", "SECRET", "API_KEY", "alice", "test-session", "ababab", "native-client"])
        assert len(serialized.encode()) <= 4096
        reopened = NativePushStore(path, clock=clock)
        second = NativePushDispatcher(reopened, provider, origin="https://relay.example", session_active=_active, clock=clock)
        await second.notify_turn_end("mac", outcome="success", context=context, event_id="same-event")
        assert await second.drain_once() == 0
        assert len(requests) == 1
        await second.notify_turn_end("vps", outcome="failed", context=context, event_id="same-event")
        await second.drain_once()
        assert len(requests) == 2 and requests[0][1]["apns-id"] != requests[1][1]["apns-id"]
        await dispatcher.close()

    asyncio.run(run())


def test_restart_cannot_authorize_persisted_token_without_live_session(tmp_path, signing_key):
    async def run():
        clock, sent = Clock(), []
        path = str(tmp_path / "native.sqlite3")
        store = NativePushStore(path, clock=clock)
        installation = await store.upsert(_installation(clock))

        async def sender(*_args):
            sent.append(True)
            return APNsResponse(200)

        provider = _provider(signing_key, sender, clock)
        dispatcher = NativePushDispatcher(store, provider, origin="https://relay.example", session_active=_active, clock=clock)
        await dispatcher.notify_turn_end("mac", outcome="success", context=CONTEXT, event_id="before-restart")

        async def no_session(value, machine_id):
            assert value.session_jti == installation.session_jti and machine_id == "mac"
            return False

        restarted = NativePushDispatcher(NativePushStore(path, clock=clock), provider, origin="https://relay.example", session_active=no_session, clock=clock)
        assert await restarted.drain_once() == 1
        assert sent == []
        assert await store.for_machine("mac") == []
        assert _rows(store)[0]["status"] == "cancelled"

    asyncio.run(run())


def test_limited_retry_honors_delay_and_rechecks_revocation(tmp_path, signing_key):
    async def run():
        clock, sent, active = Clock(), [], [True]
        store = NativePushStore(str(tmp_path / "native.sqlite3"), clock=clock)
        await store.upsert(_installation(clock))

        async def sender(*args):
            sent.append(args)
            return APNsResponse(503, "ServiceUnavailable", retry_after=1)

        async def authorized(*_args):
            return active[0]

        provider = _provider(signing_key, sender, clock)
        dispatcher = NativePushDispatcher(store, provider, origin="https://relay.example", session_active=authorized, clock=clock)
        await dispatcher.notify_turn_end("mac", outcome="success", context=CONTEXT, event_id="retry-me")
        await dispatcher.drain_once()
        assert len(sent) == 1 and _rows(store)[0]["next_attempt"] == clock() + 900
        clock.now += 899
        assert await dispatcher.drain_once() == 0
        active[0] = False
        clock.now += 1
        assert await dispatcher.drain_once() == 1
        assert len(sent) == 1 and _rows(store)[0]["status"] == "cancelled"

    asyncio.run(run())


@pytest.mark.parametrize("response,delay,retries", [
    (APNsResponse(429, "TooManyRequests", retry_after=120), 120, 3),
    (APNsResponse(0), 60, 3),
    (APNsResponse(403, "InvalidProviderToken"), 0, 1),
    (APNsResponse(400, "BadDeviceToken"), 0, 1),
    (APNsResponse(413, "PayloadTooLarge"), 0, 1),
    (APNsResponse(429, "TooManyProviderTokenUpdates"), 0, 1),
])
def test_retry_budget_and_nonretryable_responses(tmp_path, signing_key, response, delay, retries):
    async def run():
        clock, sent = Clock(), []
        store = NativePushStore(str(tmp_path / "native.sqlite3"), clock=clock)
        await store.upsert(_installation(clock))

        async def sender(*args):
            sent.append(args)
            return response

        dispatcher = NativePushDispatcher(store, _provider(signing_key, sender, clock), origin="https://relay.example", session_active=_active, clock=clock)
        await dispatcher.notify_turn_end("mac", outcome="success", context=CONTEXT, event_id="retry-budget")
        await dispatcher.drain_once()
        if delay:
            assert _rows(store)[0]["next_attempt"] == clock() + delay
        for _ in range(5):
            clock.now += 300
            await dispatcher.drain_once()
        assert len(sent) == retries
        assert _rows(store)[0]["status"] == "failed"
        assert len({args[1]["apns-id"] for args in sent}) == 1

    asyncio.run(run())


def test_410_cannot_invalidate_later_registration_even_with_same_token(tmp_path, signing_key):
    async def run():
        clock = Clock()
        store = NativePushStore(str(tmp_path / "native.sqlite3"), clock=clock)
        old = await store.upsert(_installation(clock))
        entered, release = asyncio.Event(), asyncio.Event()

        async def sender(*_args):
            entered.set()
            await release.wait()
            return APNsResponse(410, "Unregistered", timestamp=clock() * 1000)

        dispatcher = NativePushDispatcher(store, _provider(signing_key, sender, clock), origin="https://relay.example", session_active=_active, clock=clock)
        await dispatcher.notify_turn_end("mac", outcome="success", context=CONTEXT, event_id="old-token-send")
        task = asyncio.create_task(dispatcher.drain_once())
        await entered.wait()
        clock.now += 1
        new = await store.upsert(replace(old, session_jti="new-session-00000001"))
        assert old.revision != new.revision
        release.set()
        await task
        assert await store.get_installation(new.subject, new.installation_id) == new
        assert await store.invalidate(new, timestamp=(clock() - 1) * 1000) == 0
        assert await store.invalidate(new, timestamp=clock() * 1000) == 1

    asyncio.run(run())


def test_token_rotation_or_question_close_during_auth_never_sends(tmp_path, signing_key):
    async def run():
        clock, sent = Clock(), []
        store = NativePushStore(str(tmp_path / "native.sqlite3"), clock=clock)
        old = await store.upsert(_installation(clock))

        async def sender(*args):
            sent.append(args)
            return APNsResponse(200)

        async def rotating(_value, _machine_id):
            await store.upsert(replace(old, device_token="cd" * 32))
            return True

        dispatcher = NativePushDispatcher(store, _provider(signing_key, sender, clock), origin="https://relay.example", session_active=rotating, clock=clock)
        await dispatcher.notify_turn_end("mac", outcome="success", context=CONTEXT, event_id="rotation")
        await dispatcher.drain_once()
        assert sent == []

        async def closing(_value, _machine_id):
            await dispatcher.close_question("mac", "ask-current")
            return True

        dispatcher._session_active = closing
        await dispatcher.notify_ask_user("mac", context=CONTEXT, ask_id="ask-current", event_id="closing")
        await dispatcher.drain_once()
        assert sent == []

    asyncio.run(run())


def test_identical_registration_preserves_queued_delivery_and_retry(tmp_path, signing_key):
    async def run():
        clock, sent = Clock(), []
        store = NativePushStore(str(tmp_path / "native.sqlite3"), clock=clock)
        first = await store.upsert(_installation(clock))
        assert first.machine_ids == tuple(sorted(first.machine_ids))
        assert first.event_kinds == tuple(sorted(first.event_kinds))

        async def sender(*args):
            sent.append(args)
            return APNsResponse(429, "TooManyRequests", retry_after=60) if len(sent) == 1 else APNsResponse(200)

        dispatcher = NativePushDispatcher(store, _provider(signing_key, sender, clock), origin="https://relay.example", session_active=_active, clock=clock)
        await dispatcher.notify_turn_end("mac", outcome="success", context=CONTEXT, event_id="queued-refresh")
        clock.now += 5
        repeated = replace(first, machine_ids=tuple(reversed(first.machine_ids)),
                           event_kinds=tuple(reversed(first.event_kinds)), revision="untrusted-client-revision", updated_at=clock())
        assert await store.upsert(repeated) == first
        assert _rows(store)[0]["status"] == "pending"
        assert await dispatcher.drain_once() == 1 and len(sent) == 1
        assert _rows(store)[0]["status"] == "retry"
        clock.now += 60
        assert await store.upsert(repeated) == first
        assert _rows(store)[0]["status"] == "retry"
        assert await dispatcher.drain_once() == 1 and len(sent) == 2
        assert _rows(store)[0]["status"] == "delivered"
        assert sent[0][1]["apns-id"] == sent[1][1]["apns-id"]
        # An identical refresh isn't a fresh token registration and cannot mask
        # APNs invalidation occurring after the original registration timestamp.
        assert await store.invalidate(first, timestamp=(first.updated_at + 1) * 1000) == 1

    asyncio.run(run())


@pytest.mark.parametrize("change", [
    {"session_jti": "changed-session-000001"}, {"expires_at": 1_800_009_000},
    {"device_token": "cd" * 32}, {"topic": "com.example.another"},
    {"environment": "production"}, {"client_id": "new-client"},
    {"machine_ids": ("vps",)}, {"event_kinds": ("attention",)},
])
def test_material_registration_changes_cancel_old_delivery(tmp_path, signing_key, change):
    async def run():
        clock, sent = Clock(), []
        store = NativePushStore(str(tmp_path / "native.sqlite3"), clock=clock)
        first = await store.upsert(_installation(clock))

        async def sender(*args):
            sent.append(args)
            return APNsResponse(200)

        dispatcher = NativePushDispatcher(store, _provider(signing_key, sender, clock), origin="https://relay.example", session_active=_active, clock=clock)
        await dispatcher.notify_turn_end("mac", outcome="success", context=CONTEXT, event_id="replaced-binding")
        clock.now += 5
        changed = await store.upsert(replace(first, **change))
        assert changed.revision != first.revision and changed.updated_at == clock()
        assert _rows(store)[0]["status"] == "cancelled"
        assert await dispatcher.drain_once() == 0 and not sent
        assert await store.invalidate(first, timestamp=clock() * 1000) == 0
        assert await store.get_installation(changed.subject, changed.installation_id) == changed

    asyncio.run(run())


@pytest.mark.parametrize("replace_binding", [False, True])
def test_revocation_during_pending_read_prevents_new_http_request(tmp_path, signing_key, monkeypatch, replace_binding):
    async def run():
        clock, sent, active = Clock(), [], [True]
        store = NativePushStore(str(tmp_path / "native.sqlite3"), clock=clock)
        installation = await store.upsert(_installation(clock))
        checks = []
        replacement = []

        async def sender(*args):
            sent.append(args)
            return APNsResponse(200)

        async def authorized(*_args):
            checks.append(active[0])
            return active[0]

        real_pending = store.is_pending

        async def revoke_during_storage_read(delivery_id):
            result = await real_pending(delivery_id)
            # SessionRegistry revoke is immediate; DB removal deliberately lags.
            active[0] = False
            assert await store.get_installation(installation.subject, installation.installation_id) is not None
            if replace_binding:
                replacement.append(await store.upsert(replace(installation, session_jti="new-login-session-0001")))
            return result

        monkeypatch.setattr(store, "is_pending", revoke_during_storage_read)
        dispatcher = NativePushDispatcher(store, _provider(signing_key, sender, clock), origin="https://relay.example", session_active=authorized, clock=clock)
        await dispatcher.notify_turn_end("mac", outcome="success", context=CONTEXT, event_id="logout-window")
        await dispatcher.drain_once()
        assert checks == [True, False] and not sent
        assert _rows(store)[0]["status"] == "cancelled"
        current = await store.get_installation(installation.subject, installation.installation_id)
        assert current == (replacement[0] if replace_binding else None)

    asyncio.run(run())


def test_attention_fallback_event_filter_and_persistent_close_tombstone(tmp_path, signing_key):
    async def run():
        clock, sent = Clock(), []
        path = str(tmp_path / "native.sqlite3")
        store = NativePushStore(path, clock=clock)
        await store.upsert(_installation(clock, event_kinds=("attention",)))

        async def sender(_url, _headers, payload):
            sent.append(json.loads(payload))
            return APNsResponse(429, "TooManyRequests")

        provider = _provider(signing_key, sender, clock)
        dispatcher = NativePushDispatcher(store, provider, origin="https://relay.example", session_active=_active, clock=clock)
        await dispatcher.notify_turn_end("mac", outcome="success", context=CONTEXT, event_id="filtered-kind")
        await dispatcher.notify_ask_user("mac", context={"sid": "session-one", "engine": "codex", "question": "SECRET"}, ask_id="ask-open", event_id="new-question")
        assert _rows(store)[0]["expires_at"] == clock() + 300
        await dispatcher.drain_once()
        assert len(sent) == 1
        route = sent[0]["cc_remote"]
        assert route["event_kind"] == "attention" and route["ask_id"] == "ask-open"
        assert "engine" not in route and "space" not in route and "SECRET" not in json.dumps(sent)
        await dispatcher.close_question("mac", "ask-open")
        clock.now += 120
        assert await dispatcher.drain_once() == 0
        assert len(sent) == 1
        await dispatcher.close_question("mac", "closed-before-enqueue")
        reopened = NativePushStore(path, clock=clock)
        restarted = NativePushDispatcher(reopened, provider, origin="https://relay.example", session_active=_active, clock=clock)
        await restarted.notify_ask_user("mac", context=CONTEXT, ask_id="closed-before-enqueue", event_id="late-replayed-question")
        assert await restarted.drain_once() == 0
        await restarted.notify_ask_user("mac", context=CONTEXT, ask_id="private", event_id="private-event", target_client_id="another-client")
        assert await restarted.drain_once() == 0

    asyncio.run(run())


def test_worker_recovers_from_storage_failure_with_fixed_warning(tmp_path, signing_key, monkeypatch):
    async def run():
        clock, sent, warnings = Clock(), [], []
        store = NativePushStore(str(tmp_path / "native.sqlite3"), clock=clock)
        await store.upsert(_installation(clock))

        async def sender(*args):
            sent.append(args)
            return APNsResponse(200)

        class SafeLogger:
            def warning(self, message):
                warnings.append(message)

        monkeypatch.setattr("cc_remote.relay.native_push.log", SafeLogger())
        dispatcher = NativePushDispatcher(store, _provider(signing_key, sender, clock), origin="https://relay.example", session_active=_active, clock=clock)
        await dispatcher.notify_turn_end("mac", outcome="success", context=CONTEXT, event_id="disk-recovery")
        real_claim = store.claim_due
        failed = asyncio.Event()
        calls = 0

        async def intermittent_claim():
            nonlocal calls
            calls += 1
            if calls == 1:
                failed.set()
                raise OSError("PRIVATE-TOKEN PRIVATE-DB-PATH")
            return await real_claim()

        monkeypatch.setattr(store, "claim_due", intermittent_claim)
        await dispatcher.start()
        await failed.wait()
        await asyncio.sleep(0)
        assert not dispatcher._worker.done() and calls == 1 and not sent
        assert warnings == ["native push storage unavailable; delivery will retry"]
        assert "PRIVATE" not in str(warnings)
        # A new event wakes the bounded wait; ordinary retry also happens after
        # its 2-second timeout. No real sleep or Apple connection is needed here.
        dispatcher._wake.set()
        for _ in range(200):
            if sent:
                break
            await asyncio.sleep(0.005)
        assert len(sent) == 1 and calls == 2
        await dispatcher.close()

    asyncio.run(run())


def test_delivery_capacity_and_worker_lifecycle(tmp_path, signing_key):
    async def run():
        clock, sent = Clock(), []
        store = NativePushStore(str(tmp_path / "native.sqlite3"), clock=clock, max_deliveries=2)
        await store.upsert(_installation(clock))

        async def sender(*args):
            sent.append(args)
            return APNsResponse(200)

        dispatcher = NativePushDispatcher(store, _provider(signing_key, sender, clock), origin="https://relay.example", session_active=_active, clock=clock)
        for index in range(10):
            await dispatcher.notify_turn_end("mac", outcome="success", context=CONTEXT, event_id=f"event-{index}")
        assert len(_rows(store)) == 2
        await dispatcher.start()
        task = dispatcher._worker
        await dispatcher.start()
        assert task is dispatcher._worker
        for _ in range(200):
            if len(sent) == 2:
                break
            await asyncio.sleep(0.005)
        assert len(sent) == 2
        await dispatcher.close()
        assert task.done()

    asyncio.run(run())


def test_terminal_journal_capacity_never_blocks_new_pending_work(tmp_path, signing_key):
    async def run():
        clock, sent = Clock(), []
        store = NativePushStore(str(tmp_path / "native.sqlite3"), clock=clock, max_deliveries=3)
        await store.upsert(_installation(clock))

        async def sender(_url, _headers, payload):
            sent.append(json.loads(payload)["cc_remote"]["event_id"])
            return APNsResponse(200)

        dispatcher = NativePushDispatcher(store, _provider(signing_key, sender, clock), origin="https://relay.example", session_active=_active, clock=clock)

        async def enqueue(event_id):
            clock.now += 1
            await dispatcher.notify_turn_end("mac", outcome="success", context=CONTEXT, event_id=event_id)

        def event_ids():
            return {json.loads(row["payload"])["cc_remote"]["event_id"] for row in _rows(store)}

        for index in range(3):
            await enqueue(f"old-{index}")
        await dispatcher.drain_once()
        assert len(sent) == 3 and all(row["status"] == "delivered" for row in _rows(store))
        await enqueue("old-2")
        assert len(_rows(store)) == 3 and await dispatcher.drain_once() == 0
        # Duplicate checking precedes eviction. New work reclaims only the
        # oldest terminal receipt, retaining the remaining recent dedupe window.
        await enqueue("new-0")
        assert event_ids() == {"old-1", "old-2", "new-0"}
        await enqueue("new-1")
        assert event_ids() == {"old-2", "new-0", "new-1"}
        await enqueue("new-2")
        assert event_ids() == {"new-0", "new-1", "new-2"}
        assert all(row["status"] == "pending" for row in _rows(store))
        await enqueue("cannot-evict-pending")
        assert event_ids() == {"new-0", "new-1", "new-2"}
        assert await dispatcher.drain_once() == 3
        assert set(sent) == {"old-0", "old-1", "old-2", "new-0", "new-1", "new-2"}
        await enqueue("still-delivers")
        assert len(_rows(store)) == 3
        assert await dispatcher.drain_once() == 1
        assert sent[-1] == "still-delivers"

    asyncio.run(run())
