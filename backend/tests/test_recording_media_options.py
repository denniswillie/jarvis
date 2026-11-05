from uuid import uuid4

import pytest

from app.app_controls import AppCommand
from app.files.recording_ingress import RecordingClientReport
from app.recordings.voice import LeanRecordingDispatcher
from app.recordings.admission import RecordingAdmission
from app.live_state.admission import CommandAdmission


def dispatcher():
    return LeanRecordingDispatcher(owner_id="test", session_id=uuid4(), admissions=None, policy=None,
        provider_session_key=lambda: "test", is_active=lambda: True, on_command=lambda value: True)


def capabilities(version=2):
    fields = dict(type="recording.capabilities", version=version, generation_id=str(uuid4()), available=True,
                  include_jarvis_audio=True, max_bytes=100_000, max_duration_ms=1_800_000)
    if version == 2:
        fields.update(include_media_audio=True, voiceover=True)
    return fields


def test_v1_is_unchanged_and_v2_selects_music_without_an_extra_question():
    value = dispatcher()
    assert value.configure_device(capabilities(1))
    assert value._normalize(dict(action="start", include_jarvis_audio=False)) == ("start", {"include_jarvis_audio": False})
    assert value.configure_device(capabilities(2))
    for include in (True, False):
        _, args = value._normalize(dict(action="start", include_jarvis_audio=include))
        assert args == dict(include_jarvis_audio=include, recording_mode="ordinary", include_media_audio=True)


def test_only_explicit_voiceover_silences_media():
    value = dispatcher()
    assert value.configure_device(capabilities())
    _, args = value._normalize(dict(action="start", include_jarvis_audio=False, recording_mode="voiceover"))
    assert args == dict(include_jarvis_audio=False, recording_mode="voiceover", include_media_audio=False)
    with pytest.raises(ValueError):
        value._normalize(dict(action="start", include_jarvis_audio=True, recording_mode="voiceover"))
    with pytest.raises(ValueError):
        AppCommand(action="recording.start", include_jarvis_audio=True, recording_mode="voiceover")
    AppCommand(action="recording.start", include_jarvis_audio=False, recording_mode="voiceover")


def test_bad_capabilities_do_not_enable_media():
    value = dispatcher()
    bad = capabilities(); bad["include_media_audio"] = "true"
    assert not value.configure_device(bad)
    with pytest.raises(ValueError):
        value._normalize(dict(action="start", include_jarvis_audio=False, recording_mode="voiceover"))


def test_legacy_report_serialization_does_not_acquire_new_null_fields():
    report = dict(generation_id=uuid4(), timeline_end_ms=100, video_frames=1,
                  source_audio_frames=4800, padded_audio_frames=0)
    old = RecordingClientReport(**report).model_dump(mode="json")
    assert "include_media_audio" not in old and "recording_mode" not in old
    new = RecordingClientReport(**report, recording_mode="ordinary", include_media_audio=True).model_dump(mode="json")
    assert {key: val for key, val in new.items() if key not in ("recording_mode", "include_media_audio")} == old
    with pytest.raises(ValueError):
        RecordingClientReport(**report, recording_mode="voiceover", include_media_audio=True)


def test_admission_keeps_media_options_in_original_command_fingerprint():
    # This is the pure validation boundary; database behavior remains unchanged.
    admission = object.__new__(RecordingAdmission)
    admission.owner_id = "test"
    admission.access_check = None
    base = dict(owner_id="test", session_id=uuid4(), provider_session_key="test", provider_call_id="call",
                tool_name="live_recording", capability="recordings.start")
    old = CommandAdmission(**base, normalized_arguments={"include_jarvis_audio": False})
    new = CommandAdmission(**base, normalized_arguments=dict(include_jarvis_audio=False,
        recording_mode="ordinary", include_media_audio=True))
    _, old_hash = admission._command(old, "recordings.start")
    args, new_hash = admission._command(new, "recordings.start")
    assert old_hash != new_hash and args["include_media_audio"] is True
