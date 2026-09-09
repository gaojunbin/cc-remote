"""HTTPS proxy access over an HTTP Docker bridge, without a configured domain."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect

from cc_remote.config import validate_relay_config
from cc_remote.protocol import Hello, serialize
from cc_remote.relay import server
from cc_remote.relay.auth import SESSION_COOKIE_NAME
from cc_remote.relay.origins import canonical_https_origin
from tests.test_auth import _cfg
from tests.test_native_push_routes import Provider, completion, config, registration


class HTTPUpstream:
    """Model NPM preserving Host while contacting a container over HTTP."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in {"http", "websocket"}:
            scope = {**scope, "scheme": "http" if scope["type"] == "http" else "ws",
                     "client": ("172.19.0.1", 43000), "server": ("172.19.0.2", 8765)}
        await self.app(scope, receive, send)


@pytest.fixture(autouse=True)
def reset_limiter():
    server._login_limiter.reset()
    yield
    server._login_limiter.reset()


@pytest.mark.parametrize("origin", ["https://remote.example", "https://remote.example:9443"])
def test_proxy_login_session_websocket_and_logout_over_http_upstream(origin):
    cfg = _cfg(public_origin="auto")
    app = server.create_app(cfg)
    with TestClient(HTTPUpstream(app), base_url=origin) as client:
        headers = {"Origin": origin, "X-Forwarded-Host": "attacker.example",
                   "X-Forwarded-Proto": "http"}
        response = client.post("/api/login", headers=headers, json={"password": cfg.login_password})
        assert response.status_code == 200
        cookie = response.headers["set-cookie"].lower()
        assert "secure" in cookie and "httponly" in cookie and "samesite=strict" in cookie
        assert client.get("/api/session").status_code == 200
        with client.websocket_connect(origin.replace("https:", "wss:") + "/ws", headers=headers) as ws:
            ws.send_text(serialize(Hello(role="client", client_id="proxy-test")))
            assert json.loads(ws.receive_text())["code"] == "wrapper_offline"
            ws.close()
        assert cfg.public_origin == "auto"
        response = client.post("/api/logout", headers=headers)
        assert "secure" in response.headers["set-cookie"].lower()
        assert client.get("/api/session").status_code == 401


@pytest.mark.parametrize("origin", [
    "https://attacker.example", "http://remote.example", "null",
    "https://remote.example:8443", "https://remote.example/path",
    "https://remote.example?query", "https://user@remote.example",
])
def test_auto_origin_rejects_cross_site_http_and_malformed_requests(origin):
    cfg = _cfg(public_origin="auto")
    with TestClient(HTTPUpstream(server.create_app(cfg)), base_url="https://remote.example") as client:
        response = client.post("/api/login", headers={"Origin": origin},
                               json={"password": cfg.login_password})
        assert response.status_code == 403
        assert "set-cookie" not in response.headers
        login = client.post("/api/login", json={"password": cfg.login_password})
        assert login.status_code == 200
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect("wss://remote.example/ws", headers={"Origin": origin}):
                pass
        assert error.value.code == 1008


def test_auto_sessions_cannot_move_to_another_valid_host():
    cfg = _cfg(public_origin="auto")
    with TestClient(HTTPUpstream(server.create_app(cfg)), base_url="https://first.example") as client:
        login = client.post("/api/login", json={"password": cfg.login_password})
        cookie = f"{SESSION_COOKIE_NAME}={login.cookies.get(SESSION_COOKIE_NAME)}"
        second = {"Host": "second.example", "Origin": "https://second.example", "Cookie": cookie}
        assert client.get("/api/session", headers=second).status_code == 401
        assert client.get("/api/devices", headers=second).status_code == 401
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("wss://second.example/ws", headers=second):
                pass
        assert client.post("/api/logout", headers=second).status_code == 200
        assert client.get("/api/session", headers={"Cookie": cookie}).status_code == 200


@pytest.mark.parametrize("hosts", [[], ["a.example", "b.example"], ["a.example:0"],
    ["a.example:"], ["a.example:99999"], ["user@a.example"], ["a.example/path"],
    ["a.example#"], ["a.example?"], ["a.example\\evil"], ["a.example b.example"],
    ["a.example\n"], ["[::1"], ["bad..example"]])
def test_auto_origin_rejects_missing_duplicate_or_invalid_host(hosts):
    req = Request({"type": "http", "scheme": "http", "path": "/api/login", "query_string": b"",
                   "server": ("172.19.0.2", 8765), "headers": [
                       (b"host", host.encode()) for host in hosts]})
    assert not server._request_origin_allowed(req, _cfg(public_origin="auto"))


@pytest.mark.parametrize(("value", "expected"), [
    ("https://REMOTE.example:443", "https://remote.example"),
    ("https://[2001:db8::1]:9443", "https://[2001:db8::1]:9443"),
    ("https://127.0.0.1", "https://127.0.0.1"),
])
def test_https_origin_canonicalization(value, expected):
    assert canonical_https_origin(value) == expected


@pytest.mark.parametrize("overrides", [
    {"login_password": "short"}, {"session_secret": ""}, {"wrapper_token": ""},
    {"allow_insecure_http": True}, {"allow_private_origins": True},
])
def test_auto_mode_keeps_config_validation_and_https_requirement(overrides):
    with pytest.raises(ValueError):
        validate_relay_config(_cfg(public_origin="auto", **overrides))


def test_apns_uses_each_login_origin_without_learning_a_global_domain(tmp_path):
    cfg, provider = config(tmp_path, public_origin="auto"), Provider()
    app = server.create_app(cfg, native_push_provider=provider)
    with TestClient(HTTPUpstream(app), base_url="https://first.example") as client:
        for domain, token in (("first.example", "ab" * 32), ("second.example", "cd" * 32)):
            origin = "https://" + domain
            assert client.post(origin + "/api/login", json={"password": cfg.login_password}).status_code == 200
            assert client.put(origin + "/api/native-push/installation", headers={"Origin": origin},
                              json=registration(device_token=token)).status_code == 200

        async def emit():
            await app.state.hub._on_wrapper_msg(Hello(role="wrapper", wrapper_generation="generation"), "mac")
            await app.state.hub._on_wrapper_msg(completion(), "mac")
            await app.state.native_push_router.wait_idle()
            await app.state.native_push_dispatcher.drain_once()

        client.portal.call(emit)
        assert len(provider.calls) == 2
        assert {token: payload["cc_remote"]["origin"] for token, payload, _ in provider.calls} == {
            "ab" * 32: "https://first.example", "cd" * 32: "https://second.example"}
        assert cfg.public_origin == "auto"
