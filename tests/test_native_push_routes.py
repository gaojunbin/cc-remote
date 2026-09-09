"""Native HTTP/live-routing acceptance with an injected Apple sender; no models."""
from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from cc_remote.config import RelayConfig, validate_relay_config
from cc_remote.protocol import (
    AskUser, AskUserClosed, Hello, SessionInfo, SessionList, SessionRekey,
    TurnEnd, TurnNotificationContext, TurnResult,
)
from cc_remote.relay import server
from cc_remote.relay.native_push import APNsResponse
from cc_remote.relay.native_push_router import NativePushRouter


class Provider:
    topic = "com.example.remote"
    environments = ("sandbox", "production")

    def __init__(self):
        self.calls = []

    async def send(self, device_token, payload, **options):
        self.calls.append((device_token, json.loads(payload), options))
        return APNsResponse(200)

    async def close(self):
        pass


@pytest.fixture(autouse=True)
def reset_limit():
    server._login_limiter.reset()
    yield
    server._login_limiter.reset()


def config(tmp_path, **overrides):
    values = dict(login_password="fixture login password", wrapper_token="w" * 48,
                  session_secret="s" * 48, public_origin="https://remote.example",
                  device_db_path=str(tmp_path / "devices.sqlite3"),
                  apns_team_id="TEAM123456", apns_key_id="KEY1234567",
                  apns_topic=Provider.topic, apns_key_path="/test-only/AuthKey.p8",
                  apns_db_path=str(tmp_path / "native.sqlite3"))
    values.update(overrides)
    return RelayConfig(**values)


def registration(**overrides):
    body = dict(installation_id=str(uuid.uuid4()), device_token="ab" * 32,
                environment="sandbox", machine_ids=["mac"], client_id="native-client",
                privacy="generic", event_kinds=["completion", "attention"])
    body.update(overrides)
    return body


def login(client, cfg, username=""):
    response = client.post("/api/login", json={"username": username, "password": cfg.login_password})
    assert response.status_code == 200


def test_native_registration_requires_exact_origin_auth_and_valid_payload(tmp_path):
    cfg = config(tmp_path)
    app = server.create_app(cfg, native_push_provider=Provider())
    with TestClient(app, base_url=cfg.public_origin) as client:
        body = registration()
        assert client.get("/api/native-push/config").status_code == 401
        assert client.put("/api/native-push/installation", json=body,
                          headers={"Origin": cfg.public_origin}).status_code == 401
        login(client, cfg)
        capability = client.get("/api/native-push/config")
        assert capability.headers["cache-control"] == "no-store"
        assert capability.json() == {"enabled": True, "event_kinds": ["completion", "attention"],
                                     "environments": ["sandbox", "production"], "topic": Provider.topic}
        assert client.put("/api/native-push/installation", json=body).status_code == 403
        assert client.put("/api/native-push/installation", json=body,
                          headers={"Origin": "https://other.example"}).status_code == 403
        client.headers["Origin"] = cfg.public_origin
        for changes in ({"device_token": "odd"}, {"device_token": "ab" * 513}, {"environment": "other"},
                        {"machine_ids": ["mac", "mac"]}, {"machine_ids": []}, {"machine_ids": [None]},
                        {"privacy": "session"}, {"event_kinds": [{}]}, {"event_kinds": []},
                        {"installation_id": 42}, {"client_id": "client\nunsafe"}, {"topic": "attacker"}):
            assert client.put("/api/native-push/installation", json={**body, **changes}).status_code == 400
        response = client.put("/api/native-push/installation", json=body)
        assert response.json() == {"ok": True, "installation_id": body["installation_id"]}
        assert response.headers["cache-control"] == "no-store"
        assert "device_token" not in response.text
        stored = client.portal.call(app.state.native_push_store.for_machine, "mac")
        assert len(stored) == 1 and stored[0].subject == "legacy"
        assert stored[0].session_jti and stored[0].topic == Provider.topic
        assert client.put("/api/native-push/installation", content=b"x" * (16 * 1024 + 1)).status_code == 413
        assert client.delete("/api/native-push/installation/" + body["installation_id"]).json() == {"ok": True}
        assert client.portal.call(app.state.native_push_store.for_machine, "mac") == []


def test_disabled_native_push_is_optional_and_does_not_read_a_key(tmp_path):
    cfg = config(tmp_path, apns_team_id="", apns_key_id="", apns_topic="", apns_key_path="")
    app = server.create_app(cfg)
    with TestClient(app, base_url=cfg.public_origin, headers={"Origin": cfg.public_origin}) as client:
        login(client, cfg)
        assert client.get("/api/native-push/config").json() == {
            "enabled": False, "event_kinds": [], "environments": [], "topic": ""}
        assert client.put("/api/native-push/installation", json=registration()).status_code == 503


def test_native_authorization_rechecks_logout_after_device_lookup(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    provider = Provider()
    app = server.create_app(cfg, native_push_provider=provider)
    with TestClient(app, base_url=cfg.public_origin, headers={"Origin": cfg.public_origin}) as client:
        login(client, cfg)
        assert client.put("/api/native-push/installation", json=registration()).status_code == 200
        installation = client.portal.call(app.state.native_push_store.for_machine, "mac")[0]

        async def device_lookup_during_logout(claims, machine_id, devices):
            # Ownership read yields to logout; database cleanup can be slower.
            await app.state.sessions.revoke(claims.jti)
            return True

        monkeypatch.setattr(server, "_claims_allow_machine", device_lookup_during_logout)
        active = client.portal.call(app.state.native_push_dispatcher._session_active, installation, "mac")
        assert active is False
        assert provider.calls == []


def test_native_devices_and_unregister_are_account_and_login_scoped(tmp_path):
    users = {"alice": {"password": "fixture login password", "machines": ["mac"]},
             "bob": {"password": "fixture login password", "machines": ["vps"]}}
    cfg = config(tmp_path, login_users_json=json.dumps(users))
    app = server.create_app(cfg, native_push_provider=Provider())
    with TestClient(app, base_url=cfg.public_origin, headers={"Origin": cfg.public_origin}) as client:
        login(client, cfg, "alice")
        body = registration()
        assert client.put("/api/native-push/installation", json={**body, "machine_ids": ["vps"]}).status_code == 403
        assert client.put("/api/native-push/installation", json=body).status_code == 200
        old_cookie = client.cookies.get("cc_remote_session")
        login(client, cfg, "alice")
        assert client.put("/api/native-push/installation", json={**body, "device_token": "cd" * 32}).status_code == 200
        # Delayed cleanup by a prior login must not delete the new registration.
        response = client.delete("/api/native-push/installation/" + body["installation_id"],
                                 headers={"Cookie": "cc_remote_session=" + old_cookie})
        assert response.status_code == 200
        assert len(client.portal.call(app.state.native_push_store.for_machine, "mac")) == 1
        login(client, cfg, "bob")
        assert client.delete("/api/native-push/installation/" + body["installation_id"]).status_code == 200
        assert len(client.portal.call(app.state.native_push_store.for_machine, "mac")) == 1


def completion(seq=10, **changes):
    return TurnEnd(sid="session", seq=seq, turn_id="native-turn", result=TurnResult(
        subtype="success", duration_ms=1, is_error=False), notification_context=TurnNotificationContext(
            engine="codex", space="code", display_name="PRIVATE-TITLE"), **changes)


def test_real_relay_live_hook_privacy_dedup_logout_and_restart(tmp_path):
    cfg, provider = config(tmp_path), Provider()
    app = server.create_app(cfg, native_push_provider=provider)
    with TestClient(app, base_url=cfg.public_origin, headers={"Origin": cfg.public_origin}) as client:
        login(client, cfg)
        body = registration()
        assert client.put("/api/native-push/installation", json=body).status_code == 200

        async def emit():
            hub = app.state.hub
            await hub._on_wrapper_msg(Hello(role="wrapper", wrapper_generation="generation"), "mac")
            await hub._on_wrapper_msg(completion(), "mac")
            await hub._on_wrapper_msg(completion(), "mac")
            await hub._on_wrapper_msg(completion(to="native-client"), "mac")
            await hub._on_wrapper_msg(AskUser(sid="session", seq=11, ask_id="private-ask", question="PRIVATE-QUESTION",
                                             allow_text=True, to="another-client"), "mac")
            await hub._on_wrapper_msg(AskUser(sid="session", seq=12, ask_id="public-ask", question="PRIVATE-QUESTION", allow_text=True), "mac")
            await app.state.native_push_router.wait_idle()
            await app.state.native_push_dispatcher.drain_once()

        client.portal.call(emit)
        assert len(provider.calls) == 2
        encoded = json.dumps(provider.calls)
        assert "PRIVATE-QUESTION" not in encoded and "PRIVATE-TITLE" not in encoded and "private-ask" not in encoded
        route = next(call[1]["cc_remote"] for call in provider.calls
                     if call[1]["cc_remote"]["event_kind"] == "completion")
        assert route["origin"] == cfg.public_origin and route["machine_id"] == "mac"
        assert route["engine"] == "codex" and route["space"] == "code"
        assert client.post("/api/logout").status_code == 200
        assert client.portal.call(app.state.native_push_store.for_machine, "mac") == []
        login(client, cfg)
        assert client.put("/api/native-push/installation", json=body).status_code == 200
        cookie = client.cookies.get("cc_remote_session")
    # The persisted installation is not an independent authorization grant.
    restarted = server.create_app(cfg, native_push_provider=Provider())
    with TestClient(restarted, base_url=cfg.public_origin) as client:
        assert client.get("/api/native-push/config", headers={"Cookie": "cc_remote_session=" + cookie}).status_code == 401


@pytest.mark.asyncio
async def test_router_replay_context_rekey_and_ordering():
    class Dispatcher:
        def __init__(self): self.calls = []
        async def notify_turn_end(self, machine_id, **fields): self.calls.append(("completion", machine_id, fields))
        async def notify_ask_user(self, machine_id, **fields): self.calls.append(("attention", machine_id, fields))
        async def close_question(self, machine_id, ask_id): self.calls.append(("close", machine_id, ask_id))
    target = Dispatcher()
    router = NativePushRouter(target)
    await router.start()
    try:
        await router.observe("mac", Hello(role="wrapper", wrapper_generation="g"))
        await router.observe("mac", SessionList(engine="claude", sessions=[SessionInfo(session_id="old", engine="claude", space="work")], to="browser"))
        await router.observe("mac", SessionRekey(old_key="old", session_id="new"))
        ask = AskUser(sid="new", seq=5, ask_id="ask", question="PRIVATE", allow_text=True)
        await router.observe("mac", ask)
        await router.observe("mac", AskUserClosed(sid="new", seq=6, ask_id="ask", reason="answered"))
        await router.observe("mac", ask.model_copy(update={"to": "browser", "seq": None}))
        await router.observe("mac", completion().model_copy(update={"notification_context": None}))
        await router.wait_idle()
        assert [item[0] for item in target.calls] == ["attention", "close"]
        assert target.calls[0][2]["context"] == {"sid": "new", "engine": "claude", "space": "work"}
        assert "PRIVATE" not in repr(target.calls)
    finally:
        await router.close()


@pytest.mark.parametrize("change", [dict(apns_team_id="bad"), dict(apns_key_path="relative.p8"),
                                   dict(apns_key_id=""), dict(apns_environments="sandbox,sandbox"),
                                   dict(apns_topic="a/../b"), dict(apns_environments="sandbox,other")])
def test_native_config_fails_closed(tmp_path, change):
    with pytest.raises(ValueError):
        validate_relay_config(config(tmp_path, **change))
