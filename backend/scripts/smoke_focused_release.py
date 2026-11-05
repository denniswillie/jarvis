from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx


@dataclass(frozen=True, slots=True)
class SmokeCheck:
    path: str
    expected_status: int
    require_ready: bool
    authenticated: bool


CHECKS = (
    SmokeCheck("/health", 200, False, False),
    SmokeCheck("/ready", 200, True, False),
    SmokeCheck("/diagnostics/config", 200, True, True),
    SmokeCheck("/tasks/providers", 200, False, True),
    SmokeCheck("/attention/push-status", 200, False, True),
)


def focused_payload(path, payload, *, require_push=False):
    """Readiness only: not proof of recall, completed work or physical delivery."""
    if path == "/diagnostics/config":
        jobs = payload.get("background_jobs", {}) if isinstance(payload, dict) else {}
        return all(jobs.get(name) is False for name in
            ("memory_job_runner", "memory_maintenance_scheduler")), []
    if path == "/tasks/providers":
        return isinstance(payload, list) and any(
            isinstance(row, dict) and row.get("backend_kind") == "frontier"
            and row.get("configured") is True and row.get("worker_ready") is True
            for row in payload), []
    if path == "/attention/push-status":
        if not isinstance(payload, dict):
            return False, ["push_status_unavailable"]
        running = payload.get("state") == "running"
        warnings = [] if running else ["phone_push_not_running"]
        if payload.get("provider_access") != "verified":
            warnings.append("actual_apns_delivery_unverified")
        known = running or (payload.get("state") == "unavailable"
                            and payload.get("reason") == "push_disabled")
        return known and (running or not require_push), warnings
    return True, []


async def run_smoke(
    *,
    base_url: str,
    token: str,
    allow_degraded: bool,
    require_push: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[dict[str, object]]:
    public_headers = {"ngrok-skip-browser-warning": "true"}
    authenticated_headers = {
        **public_headers,
        "Authorization": f"Bearer {token}",
    }
    results: list[dict[str, object]] = []
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=15,
        transport=transport,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        for check in CHECKS:
            try:
                response = await client.get(
                    check.path,
                    headers=(
                        authenticated_headers
                        if check.authenticated
                        else public_headers
                    ),
                )
                payload = response.json()
                status = payload.get("status") if isinstance(payload, dict) else None
                passed = response.status_code == check.expected_status
                if check.require_ready and not allow_degraded:
                    passed = passed and status in {"ok", "ready"}
                focused, warnings = focused_payload(check.path, payload, require_push=require_push)
                passed = passed and focused
                results.append(
                    {
                        "path": check.path,
                        "http_status": response.status_code,
                        "service_status": status,
                        "passed": passed,
                        "warnings": warnings,
                    }
                )
            except (httpx.HTTPError, ValueError) as error:
                results.append(
                    {
                        "path": check.path,
                        "http_status": None,
                        "service_status": None,
                        "passed": False,
                        "error_type": type(error).__name__,
                    }
                )
    return results


async def run_smoke_series(
    *,
    base_url: str,
    token: str,
    allow_degraded: bool,
    require_push: bool = False,
    attempts: int,
    interval_seconds: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[dict[str, object]]:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if interval_seconds < 0:
        raise ValueError("interval_seconds must not be negative")

    snapshots: list[dict[str, object]] = []
    for attempt in range(1, attempts + 1):
        checks = await run_smoke(
            base_url=base_url,
            token=token,
            allow_degraded=allow_degraded,
            require_push=require_push,
            transport=transport,
        )
        passed = all(bool(check["passed"]) for check in checks)
        snapshots.append(
            {
                "attempt": attempt,
                "checked_at": datetime.now(UTC).isoformat(),
                "passed": passed,
                "checks": checks,
            }
        )
        if passed:
            break
        if attempt < attempts and interval_seconds:
            await asyncio.sleep(interval_seconds)
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description="Check focused Jarvis release readiness, not model or physical acceptance")
    parser.add_argument(
        "--base-url",
        default=os.getenv("JARVIS_BASE_URL", "http://127.0.0.1:9000"),
    )
    parser.add_argument(
        "--token",
        default=os.getenv("JARVIS_STATIC_BEARER_TOKEN"),
    )
    parser.add_argument("--allow-degraded", action="store_true")
    parser.add_argument("--require-push", action="store_true",
        help="Fail unless the APNs software producer is running; real delivery still needs device evidence.")
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Maximum readiness attempts; successful checks stop retries early.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=2.0,
        help="Delay between failed readiness attempts.",
    )
    args = parser.parse_args()
    if not args.token:
        parser.error("--token or JARVIS_STATIC_BEARER_TOKEN is required")
    attempts = asyncio.run(
        run_smoke_series(
            base_url=args.base_url,
            token=args.token,
            allow_degraded=args.allow_degraded,
            require_push=args.require_push,
            attempts=args.attempts,
            interval_seconds=args.interval_seconds,
        )
    )
    passed = bool(attempts[-1]["passed"])
    print(json.dumps({"passed": passed, "attempts": attempts}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
