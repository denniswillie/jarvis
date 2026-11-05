from copy import deepcopy
from datetime import datetime, timezone

import pytest

from app.perception.working_memory import normalize_owner_working_memory


def test_shared_companion_fields_do_not_break_legacy_projection():
    state = {
        "companion_live_state": {"revision": 3},
        "companion_installed_plugins": ["shotcut", "vlc"],
    }
    original = deepcopy(state)
    assert normalize_owner_working_memory(state, now=datetime.now(timezone.utc)) == {}
    assert state == original


def test_unrecognized_owner_fields_still_rejected():
    with pytest.raises(ValueError, match="unsupported owner working-memory fields"):
        normalize_owner_working_memory(
            {"unrecognized": {}}, now=datetime.now(timezone.utc)
        )


def test_owned_fields_still_validated_alongside_plugin_inventory():
    with pytest.raises(ValueError, match="active_goal_ids must be an object"):
        normalize_owner_working_memory(
            {"companion_installed_plugins": ["shotcut"], "active_goal_ids": None},
            now=datetime.now(timezone.utc),
        )
