"""Observed failures are evidence, not already-passing regression tests."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Regression:
    case_id: str
    scenario: str
    oracle: str
    limitation: str


CASES = (
    Regression("R01", "GPT committed turn with no answer until reprompt", "bounded explicit turn outcome", "historical root cause unknown"),
    Regression("R02", "lost item found, cancelled, then replaced", "one active target and one delivered find-once event", "native intent choice also needs evaluation"),
    Regression("R03", "work stopped, not completed; missing item later found", "retrieved state reflects evidence and time", "harmful historical retrieval not demonstrated"),
    Regression("R04", "flat branded object may be a box", "acknowledge possible match and request another view", "native recognition quality, not deterministic"),
    Regression("R05", "valid scene with invalid optional references", "safe scene retained, invalid candidate rejected", "do not reconstruct rejected raw outputs"),
    Regression("R06", "no camera frames arrive during view questions", "truthful unavailable evidence and diagnostic gap", "historical capture cause unknown"),
    Regression("R07", "incomplete return or wrong elif indentation", "no confident approval and useful correct next step", "nearest retained frame is not exact model input"),
    Regression("R08", "clear search frame followed by invalid detections", "valid target binding and measured first-audio delay", "historical first audible onset unknown"),
    Regression("R09", "general and tracked notifications compete", "single authoritative delivered outcome", "historical rejection cause not persisted"),
    Regression("R10", "barge-in and reset during unfinished exchange", "truthful continuity with no repeated side effects", "do not equate reconnect with network failure"),
    Regression("R11", "image selected and input released", "distinct selected, written, acknowledged evidence", "availability does not establish recognition"),
)
