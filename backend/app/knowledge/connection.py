"""Explicit local-service transport policy; no DNS lookup or provider import."""
from __future__ import annotations

from ipaddress import ip_address, ip_network
import re
from urllib.parse import urlsplit


_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_PRIVATE_NETWORKS = tuple(ip_network(value) for value in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7",
))
_SERVICE_NAME = re.compile(r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?")


def service_origin(value):
    """Normalize a credential-free origin without relaxing transport policy."""
    if (not isinstance(value, str) or not value or len(value) > 2048
            or any(ord(character) <= 32 or ord(character) == 127 for character in value)):
        raise ValueError("Invalid memory service origin")
    try:
        url = urlsplit(value)
        port = url.port
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username is not None or url.password is not None
                or url.query or url.fragment or url.path not in {"", "/"}
                or port == 0 or url.netloc.endswith(":")):
            raise ValueError
        host = "[" + url.hostname + "]" if ":" in url.hostname else url.hostname
        if port is not None:
            host += ":" + str(port)
        return url.scheme + "://" + host
    except ValueError:
        raise ValueError("Invalid memory service origin") from None


def private_http_origin(value):
    """Only an explicit local address or private single-label service name."""
    origin = service_origin(value)
    url = urlsplit(origin)
    if url.scheme != "http":
        raise ValueError("A private HTTP opt-in must name an HTTP origin")
    try:
        address = ip_address(url.hostname)
    except ValueError:
        private = _SERVICE_NAME.fullmatch(url.hostname) is not None
    else:
        private = address.is_loopback or any(address in network for network in _PRIVATE_NETWORKS)
    if not private:
        raise ValueError("A private HTTP opt-in cannot name a public or link-local endpoint")
    return origin


def hindsight_origin(value, *, approved_http_origin=None):
    origin = service_origin(value)
    approved = private_http_origin(approved_http_origin) if approved_http_origin is not None else None
    url = urlsplit(origin)
    if url.scheme == "http" and url.hostname not in _LOOPBACK and origin != approved:
        raise ValueError("Non-loopback HTTP requires its exact approved private origin")
    return origin
