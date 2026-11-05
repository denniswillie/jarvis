"""Bounded private exception observations; no payloads, SQL, secrets or locals."""
import asyncio
import re
import time
import traceback


class CanonicalRefreshDiagnostics:
    def __init__(self, *, limit=64):
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("Invalid diagnostic bound")
        self.limit, self.records, self.omitted = limit, [], 0

    def append(self, record):
        if len(self.records) >= self.limit:
            self.omitted += 1
        else:
            self.records.append(record)

    def observe(self, enrichment):
        worker = enrichment.worker

        def wrap(original, phase):
            async def observed(*args, **kwargs):
                started = time.monotonic()
                try:
                    return await original(*args, **kwargs)
                except BaseException as error:
                    code = getattr(error, "code", None)
                    self.append({"phase": phase, "session_id": str(enrichment.scope.session_id),
                        "exception_type": type(error).__name__,
                        "source_error_code": code if isinstance(code, str) and re.fullmatch(r"[a-z0-9_]{1,100}", code) else None,
                        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                        "shutdown_cancellation": isinstance(error, asyncio.CancelledError)
                            and (enrichment._closed or worker._closed),
                        "frames": [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                                   for frame in traceback.extract_tb(error.__traceback__)[-12:]]})
                    raise
            return observed

        worker.read = wrap(worker.read, "canonical_read")
        worker.apply = wrap(worker.apply, "canonical_apply")
        enrichment._lookup = wrap(enrichment._lookup, "lookup_source_resolution")
        original_invalidate = worker.invalidate

        async def invalidate(reason):
            self.append({"phase": "canonical_invalidate", "session_id": str(enrichment.scope.session_id),
                "reason": reason, "canonical_exception_type": worker.last_error,
                "failure_count": worker.failure_count, "selected_count": len(enrichment._selected)})
            return await original_invalidate(reason)
        worker.invalidate = invalidate

    def document(self):
        return {"format": "jarvis.canonical_refresh.private_diagnostics.v1",
                "records": self.records, "omitted_records": self.omitted,
                "exception_messages_and_locals_included": False}
