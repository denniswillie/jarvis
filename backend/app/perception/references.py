"""Request-local visual identifiers. Durable contracts continue to use real IDs."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from uuid import UUID

from app.perception.contracts import FastPerceptionOutputError, FastPerceptionRequest, bound_perception_questions


REFERENCE_CONTRACT_VERSION = 3
# Optional extension on brief-bearing requests; legacy v3 requests are unchanged.
ASSIGNMENT_REFERENCE_VERSION = 1


class PerceptionReferenceError(FastPerceptionOutputError):
    """Only application-owned error codes, never model content or raw IDs."""


def _uuid(value) -> str:
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        raise PerceptionReferenceError("invalid_context_reference") from None


def _mapping(prefix, values, *, short_handles=True):
    values = tuple(values)
    if len(values) != len(set(values)):
        raise PerceptionReferenceError("duplicate_context_reference")
    result = {(f"{prefix}:{index}" if short_handles else value): value for index, value in enumerate(values)}
    if any(handle in values and handle != value for handle, value in result.items()):
        raise PerceptionReferenceError("ambiguous_context_reference")
    return MappingProxyType(result)


def _subjects(request):
    values = []
    items = [("visual_focus", request.active_focus)] if isinstance(request.active_focus, Mapping) else []
    items += [("conditional_reminder", item) for item in request.active_reminders]
    for kind, item in items:
        if item.get("owner_id") not in (None, request.owner_id):
            raise PerceptionReferenceError("reference_owner_mismatch")
        if kind == "visual_focus" and item.get("session_id") is not None:
            if _uuid(item["session_id"]) != str(request.session_id):
                raise PerceptionReferenceError("reference_session_mismatch")
        if item.get("id") is not None:
            values.append((kind, _uuid(item["id"])))
    if len(values) != len(set(values)):
        raise PerceptionReferenceError("duplicate_context_reference")
    return tuple(values)


class PerceptionReferences:
    def __init__(self, request: FastPerceptionRequest, *, short_handles: bool = True) -> None:
        self._short_handles = short_handles
        self._scope = (request.owner_id, request.session_id, request.window_id)
        self.frames = _mapping("frame", (str(frame.frame_id) for frame in request.frames), short_handles=short_handles)
        self.entities = _mapping("entity", sorted(request.existing_entity_ids), short_handles=short_handles)
        subjects = _subjects(request)
        self.focus = _mapping("focus", (identifier for kind, identifier in subjects if kind == "visual_focus"), short_handles=short_handles)
        self.reminders = _mapping("reminder", (identifier for kind, identifier in subjects if kind == "conditional_reminder"), short_handles=short_handles)
        self._subjects = subjects
        self._brief = request.perception_brief
        self._brief_binding = request.perception_brief_binding
        self._brief_input_digest = request.perception_brief_input_digest
        self._questions = MappingProxyType(bound_perception_questions(request))
        check_versions = {}
        for question in self._questions.values():
            if question.check_ref is not None:
                check_versions.setdefault(str(question.check_ref.source_id), question.check_ref)
        self.questions = _mapping("question", (str(value) for value in self._questions), short_handles=short_handles)
        self.checks = _mapping("check", check_versions, short_handles=short_handles)
        self._check_versions = MappingProxyType(check_versions)

    def require_scope(self, request: FastPerceptionRequest) -> None:
        if (self._scope != (request.owner_id, request.session_id, request.window_id)
                or tuple(self.frames.values()) != tuple(str(frame.frame_id) for frame in request.frames)
                or frozenset(self.entities.values()) != request.existing_entity_ids
                or self._subjects != _subjects(request)
                or self._brief != request.perception_brief
                or self._brief_binding != request.perception_brief_binding
                or self._brief_input_digest != request.perception_brief_input_digest):
            raise PerceptionReferenceError("reference_scope_mismatch")

    @staticmethod
    def _resolve(value, mapping, code, *, uuid_value=False):
        if isinstance(value, str) and value in mapping:
            return mapping[value]
        if uuid_value:
            try:
                value = str(UUID(str(value)))
            except (ValueError, TypeError, AttributeError):
                raise PerceptionReferenceError(code) from None
        if isinstance(value, str) and value in mapping.values():
            return value
        raise PerceptionReferenceError(code)

    def frame(self, value):
        return self._resolve(value, self.frames, "unknown_frame_reference", uuid_value=True)

    def entity(self, value):
        return self._resolve(value, self.entities, "unknown_entity_reference")

    def question(self, value):
        return self._resolve(value, self.questions, "unknown_question_reference", uuid_value=True)

    def check(self, value):
        return self._resolve(value, self.checks, "unknown_check_reference", uuid_value=True)

    def finding_identity(self, question_value, check_value):
        """Map only an explicitly correct pair, never guess or repair a check.

        Returning the original SourceDependency is how content revision and
        lifecycle generation survive the model's short request-local handles.
        A frame/focus/reminder handle cannot act as a question or check handle.
        """
        question_id = self.question(question_value)
        question = self._questions[UUID(question_id)]
        if question.check_ref is None:
            if check_value is not None:
                raise PerceptionReferenceError("question_check_mismatch")
            return question_id, None
        check_id = self.check(check_value)
        if check_id != str(question.check_ref.source_id):
            raise PerceptionReferenceError("question_check_mismatch")
        return question_id, self._check_versions[check_id]

    def subject(self, kind, value):
        if kind not in ("visual_focus", "conditional_reminder"):
            raise PerceptionReferenceError("unknown_subject_type")
        mapping = self.focus if kind == "visual_focus" else self.reminders
        return self._resolve(value, mapping, "unknown_subject_reference", uuid_value=True)

    def subject_identity(self, value):
        """Resolve one bound reference, never infer a target from model prose.

        Native handles already identify their registry. Durable UUIDs remain
        compatible only when unambiguous across the supplied subject registries.
        """
        matches = []
        for kind, mapping in (("visual_focus", self.focus), ("conditional_reminder", self.reminders)):
            try:
                identifier = self._resolve(value, mapping, "unknown_subject_reference", uuid_value=True)
            except PerceptionReferenceError:
                continue
            matches.append((kind, identifier))
        if len(matches) != 1:
            raise PerceptionReferenceError("ambiguous_subject_reference" if matches else "unknown_subject_reference")
        return matches[0]

    @staticmethod
    def _reference(value, mapping):
        return next(handle for handle, identifier in mapping.items() if identifier == str(value))

    def frame_label(self, frame_id):
        return self._reference(frame_id, self.frames)

    def context(self, request: FastPerceptionRequest, context: dict) -> dict:
        self.require_scope(request)
        if request.perception_brief is not None and "perception_assignment" in context:
            assignment = context["perception_assignment"]
            context = {**context, "perception_assignment": {
                **assignment,
                "assignment_reference_version": ASSIGNMENT_REFERENCE_VERSION,
                "questions": [
                    {**question, "id": self._reference(question["id"], self.questions),
                     "check_id": (self._reference(question["check_ref"]["source_id"], self.checks)
                                  if question.get("check_ref") is not None else None)}
                    for question in assignment["questions"]
                ],
            }}
        if not self._short_handles:
            return context
        result = {key: value for key, value in context.items() if key not in {"owner_id", "session_id", "window_id"}}
        result["contract_version"] = REFERENCE_CONTRACT_VERSION
        result["primary_frame_id"] = self.frame_label(request.primary_frame.frame_id)
        result["frames"] = [
            {**frame, "frame_id": self.frame_label(frame["frame_id"]),
             "attachment_label": f"FRAME_{index:02d} id={self.frame_label(frame['frame_id'])}"}
            for index, frame in enumerate(context["frames"])
        ]
        result["allowed_existing_entity_ids"] = list(self.entities)
        state = dict(context["previous_working_state"])
        if isinstance(state.get("entities"), list):
            state["entities"] = [
                {**entity, "entity_id": self._reference(entity["entity_id"], self.entities)}
                if isinstance(entity, Mapping) and entity.get("entity_id") in self.entities.values() else entity
                for entity in state["entities"]
            ]
        result["previous_working_state"] = state

        def target(item, mapping):
            if not isinstance(item, Mapping) or item.get("id") is None:
                return item
            return {**{key: value for key, value in item.items() if key not in {"owner_id", "session_id"}},
                    "id": self._reference(_uuid(item["id"]), mapping)}

        result["active_focus"] = target(request.active_focus, self.focus)
        result["active_reminders"] = [target(item, self.reminders) for item in request.active_reminders]
        return result
