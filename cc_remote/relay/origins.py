"""HTTPS origins for deployments whose reverse proxy owns the public domain."""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

from starlette.requests import HTTPConnection


def canonical_https_origin(value: str) -> str | None:
    if (not value or len(value) > 2048
            or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
            or "\\" in value or "%" in value):
        return None
    try:
        parsed = urlsplit(value)
        hostname, port = parsed.hostname, parsed.port
        if (parsed.scheme != "https" or not hostname or parsed.username is not None
                or parsed.password is not None or parsed.path or parsed.query or parsed.fragment
                or "?" in value or "#" in value or parsed.netloc.endswith(":")):
            return None
        try:
            address = ipaddress.ip_address(hostname)
            host = f"[{address}]" if address.version == 6 else str(address)
        except ValueError:
            host = hostname.encode("idna").decode("ascii").lower()
            if len(host) > 253 or not all(
                re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                for label in host.removesuffix(".").split(".")
            ):
                return None
        if port == 0:
            return None
    except (ValueError, UnicodeError):
        return None
    return f"https://{host}" + (f":{port}" if port not in (None, 443) else "")


def request_https_origin(req: HTTPConnection) -> str | None:
    """Use only Host; HTTPS is a deployment contract, never a forwarded claim."""
    hosts = req.headers.getlist("host")
    if len(hosts) != 1:
        return None
    return canonical_https_origin("https://" + hosts[0])
