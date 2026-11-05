"""Task-local transport log suppression for credential-bearing provider traffic."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging


PRIVATE_HTTP_ACTIVE = ContextVar("jarvis_private_http_active", default=False)


class TransportPrivacyFilter(logging.Filter):
    def filter(self, record):
        return not PRIVATE_HTTP_ACTIVE.get()


TRANSPORT_PRIVACY_FILTER = TransportPrivacyFilter()
TRANSPORT_LOGGERS = (
    "httpx", "httpcore.connection", "httpcore.http11", "httpcore.http2",
    "hpack.hpack", "hpack.table",
)


@contextmanager
def private_http_logs():
    for name in TRANSPORT_LOGGERS:
        logging.getLogger(name).addFilter(TRANSPORT_PRIVACY_FILTER)
    token = PRIVATE_HTTP_ACTIVE.set(True)
    try:
        yield
    finally:
        PRIVATE_HTTP_ACTIVE.reset(token)
