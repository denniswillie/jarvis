"""Explicit-focus inputs using existing owners and removable receipts.

``visual_focus`` and ``episode`` use semantic counters on the canonical focus.
``visual_focus_subject`` remains the separate attention delivery-generation fence.
Neither an observational question nor a literal finding authorizes completion.

The exact canonical tool receipt supplies removable provenance; a new generic
payload containing a fresh read of a focus would not supply that provenance.
Current applicability still requires validate(), not merely a surviving historic
receipt. Callers retaining results must check it at their owner/effect boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from uuid import UUID, uuid5

from app.knowledge.models import SourceError
from app.live_state.canonical import CONTEXT_EVENT, context_dependency
from app.live_state.models import SourceDependency
from app.live_state.perception import ObservationalQuestion
from app.memory.prospective.repository import _focus_from_record, _lock_owner
from app.memory.prospective.models import VisualFocus
from app.tasks.context_repository import RECORD_SQL, context_record


FOCUS_SUBJECT = "visual_focus_subject"
FOCUS_CHECK = "visual_focus"
FOCUS_NAMESPACES = (FOCUS_SUBJECT, FOCUS_CHECK, "episode")
_FOCUS_SQL = """select f.*,s.generation as subject_generation,
    s.state as subject_state,s.expires_at as subject_expires_at,
    s.descriptor_digest,
    s.descriptor_digest=encode(sha256(convert_to(jsonb_build_object(
        'target',f.target,'description',f.description,'purpose',f.purpose,
        'mode',f.mode,'expires_at',f.expires_at)::text,'UTF8')),'hex') as descriptor_matches
    from jarvis_visual_focuses f left join jarvis_attention_visual_subjects s
      on s.owner_id=f.owner_id and s.subject_type='visual_focus' and s.subject_id=f.id
    where f.owner_id=$1 and f.completed_at is null and f.cleared_at is null
      and f.expires_at>$2 order by f.created_at desc,f.id limit 2"""


def _unavailable(reason):
    return SourceError("perception_focus_" + reason, "Canonical visual-focus proof is unavailable.")


@dataclass(frozen=True)
class CanonicalFocusInput:
    question: ObservationalQuestion
    source: SourceDependency
    payload_id: UUID
    expires_at: datetime
    descriptor_digest: str
    focus: VisualFocus

    @property
    def dependencies(self):
        return self.question.focus_ref, self.question.check_ref, self.question.episode_or_object_ref, self.source


def focus_dependencies(brief):
    """Partition ONLY episodes explicitly bound to a visual-focus question.

    Main's brief binder must send these to CanonicalFocusInputs.validate, not
    the responsibility episode owner. An arbitrary ``episode`` is not a focus.
    Keep the subject dependency for the existing attention output fence.
    """
    references = set()
    for question in brief.questions:
        if question.check_ref is not None and question.check_ref.source_namespace == FOCUS_CHECK:
            references.update((question.check_ref, question.episode_or_object_ref, question.focus_ref))
    return tuple(ref for ref in brief.dependencies if ref in references)


class CanonicalFocusInputs:
    """Bounded queries, through the same transaction and original-source locks."""

    def __init__(self, canonical):
        self.canonical = canonical

    async def _load(self, connection, *, brief=None, lock=False):
        scope, now = self.canonical.scope, self.canonical.clock()
        rows = await connection.fetch(_FOCUS_SQL + (" for share of f" if lock else ""),
                                      scope.owner_id, now)
        if not rows:
            return None
        if len(rows) != 1:
            raise _unavailable("ambiguous")
        row = rows[0]
        generation = row.get("subject_generation")
        revision, episode_revision = row.get("revision"), row.get("episode_revision")
        if (row["owner_id"] != scope.owner_id or not isinstance(row["id"], UUID)
                or row["completed_at"] is not None or row["cleared_at"] is not None
                or not row["created_at"] <= now < row["expires_at"]
                or row["mode"] not in ("find_once", "monitor")
                or type(generation) is not int or generation < 1
                or type(revision) is not int or revision < 1
                or type(episode_revision) is not int or episode_revision < 1
                or row.get("subject_state") != "active"
                or row.get("subject_expires_at") != row["expires_at"]
                or row.get("descriptor_matches") is not True
                or not isinstance(row.get("descriptor_digest"), str)):
            raise _unavailable("owner_version_unavailable")
        dependency = SourceDependency(source_namespace=FOCUS_SUBJECT,
            source_id=row["id"], revision=generation, generation=generation)
        check = SourceDependency(source_namespace=FOCUS_CHECK, source_id=row["id"], revision=revision)
        episode = SourceDependency(source_namespace="episode", source_id=row["id"], revision=episode_revision)
        if brief is not None and any(ref not in brief.dependencies for ref in (dependency, check, episode)):
            raise _unavailable("changed")
        fields = {key: row[key] for key in ("target", "description", "purpose", "mode")}
        if any(not isinstance(value, str) or not value.strip() or "\x00" in value
               for value in fields.values()):
            raise _unavailable("descriptor_invalid")
        text = ("Inspect the current view for the requested target described by this data: "
                + json.dumps({"target": fields["target"], "description": fields["description"]},
                             ensure_ascii=True)
                + ". Report visible matching features, location and uncertainty; if absent or unclear, "
                  "say so. Do not infer that the search or a physical action is completed.")
        if len(text) > 1000:
            raise _unavailable("question_budget")
        ids = ([source.source_id for source in brief.dependencies
                if source.source_namespace == CONTEXT_EVENT] if brief is not None else None)
        records = await connection.fetch(RECORD_SQL + """ where e.owner_id=$1 and e.session_id=$2
            and e.kind='tool_outcome' and e.trust='tool_receipt' and e.finality in ('final','revised')
            and p.availability='available'
            and p.content->>'tool_name' in ('set_visual_focus','get_visual_focus')
            and p.content#>>'{result,visual_focus,id}'=$3
            and ($4::uuid[] is null or e.id=any($4::uuid[]))
            order by e.sequence desc limit 16""", scope.owner_id, scope.session_id, str(row["id"]), ids)
        for raw in records:
            record = context_record(raw)
            content = record.content or {}
            result = content.get("result")
            rendered = result.get("visual_focus") if isinstance(result, dict) else None
            if (record.owner_id != scope.owner_id or record.session_id != scope.session_id
                    or not record.is_current or record.availability != "available"
                    or record.kind != "tool_outcome" or record.trust != "tool_receipt"
                    or record.finality not in ("final", "revised") or record.recorded_at > now
                    or content.get("tool_name") not in ("set_visual_focus", "get_visual_focus")
                    or not isinstance(rendered, dict) or result.get("status") != "success"
                    or rendered.get("id") != str(row["id"])
                    or any(rendered.get(key) != value for key, value in fields.items())
                    or rendered.get("episode_revision", 1) != episode_revision
                    or rendered.get("completed_at") is not None or rendered.get("cleared_at") is not None):
                continue
            try:
                if any(datetime.fromisoformat(rendered[key]) != row[key]
                       for key in ("created_at", "expires_at")):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            source = context_dependency(record)
            question = ObservationalQuestion(
                id=uuid5(row["id"], f"perception:{generation}:{revision}:{episode_revision}:{record.id}:{record.revision}"),
                question=text, priority="explicit_inspection", focus_ref=dependency,
                check_ref=check, episode_or_object_ref=episode)
            if brief is not None and (source not in brief.dependencies or question not in brief.questions):
                continue
            return CanonicalFocusInput(question, source, record.payload_id,
                                       row["expires_at"], row["descriptor_digest"], _focus_from_record(row))
        raise _unavailable("receipt_unavailable")

    async def _locked(self, connection, *, brief=None):
        first = await self._load(connection, brief=brief)
        if first is None:
            if brief is not None:
                raise _unavailable("changed")
            return None
        # Source locks precede the existing prospective owner/row locks. No
        # external work or model call runs while any of these locks are held.
        await self.canonical._lock_roots(connection, (first.payload_id,))
        await _lock_owner(connection, self.canonical.scope.owner_id)
        current = await self._load(connection, brief=brief, lock=True)
        if current != first:
            raise _unavailable("changed")
        return current

    async def read(self):
        async with self.canonical._transaction() as connection:
            return await self._locked(connection)

    async def validate(self, connection, brief):
        dependencies = tuple(source for source in brief.dependencies if source.source_namespace == FOCUS_SUBJECT)
        if len(dependencies) != 1:
            raise _unavailable("dependency_invalid")
        # The same current owner and exact removable receipt are consulted on
        # each bind/guard; global Live State revisions are deliberately ignored.
        current = await self._locked(connection, brief=brief)
        if set(focus_dependencies(brief)) != set(current.dependencies) - {current.source}:
            raise _unavailable("dependency_invalid")
        return current
