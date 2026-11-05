"""Exact reported quantities over canonical sources, not another memory engine."""
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation, localcontext
import json
from typing import ClassVar, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.knowledge.models import SourceError, SourcePublication
from app.knowledge.read_provenance import SourceReadProvenance
from app.tasks.models import TaskError, TaskVoiceSource, digest
from app.tools.inputs import failed, success
from app.tools.invocation import current_tool_invocation
from app.tools.provenance import SourceBoundToolResult, source_read_arguments
from app.tools.registry import ToolDefinition

NAMESPACE = "exact:measurements:v1"


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Measurement(StrictInput):
    subject: str = Field(min_length=1, max_length=160)
    metric: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=60,
        description="Exact decimal string, not a floating-point JSON number.")
    unit: str = Field(min_length=1, max_length=80)
    measured_on: date = Field(description="Reported calendar date, not note creation time.")
    notes: str = Field(default="", max_length=1000)

    @field_validator("value")
    @classmethod
    def exact_decimal(cls, value):
        try:
            number = Decimal(value)
        except InvalidOperation as error:
            raise ValueError("Use a finite decimal string.") from error
        if not number.is_finite() or not -12 <= number.as_tuple().exponent <= 12:
            raise ValueError("Use a finite decimal with exponent between -12 and 12.")
        normalized = format(number, "f")
        if len(Decimal(normalized).as_tuple().digits) > 30:
            raise ValueError("Use at most 30 digits after expanding scientific notation.")
        return normalized


class SaveMeasurement(StrictInput):
    measurement: Measurement
    request_slot: int = Field(default=1, ge=1, le=32, strict=True)


def measurement_content(measurement):
    record = measurement.model_dump(mode="json")
    return {"format": "jarvis.measurement.v1", "measurement": record,
        "text": f"Reported measurement on {record['measured_on']}: subject {record['subject']}; "
                f"metric {record['metric']}; value {record['value']} {record['unit']}. Notes: {record['notes']}"}


def measurement_correction(replacement):
    """Typed replacement through the existing authorized source-editing path."""
    try:
        return measurement_content(Measurement.model_validate_json(replacement))
    except (ValidationError, TypeError) as error:
        raise SourceError("measurement_requires_structured_correction",
            "For an exact measurement, replacement must be a JSON object with subject, metric, value "
            "(decimal string), unit, measured_on (YYYY-MM-DD), and optional notes. Preserve unrelated fields.") from error


class MeasurementQuery(StrictInput):
    subject: str | None = Field(default=None, min_length=1, max_length=160)
    metric: str | None = Field(default=None, min_length=1, max_length=120)
    unit: str | None = Field(default=None, min_length=1, max_length=80)
    from_date: date | None = None
    through_date: date | None = None
    offset: int = Field(default=0, ge=0, le=100000, strict=True)
    limit: int = Field(default=16, ge=1, le=16, strict=True)

    @model_validator(mode="after")
    def date_order(self):
        if self.from_date and self.through_date and self.from_date > self.through_date:
            raise ValueError("from_date must not follow through_date.")
        return self


class MeasurementReference(StrictInput):
    measurement_id: UUID
    revision: int = Field(ge=1, strict=True)


class MeasurementCalculation(StrictInput):
    records: list[MeasurementReference] = Field(min_length=1, max_length=32)
    operation: Literal["sum", "difference", "minimum", "maximum"]

    @model_validator(mode="after")
    def distinct_records(self):
        if len({r.measurement_id for r in self.records}) != len(self.records):
            raise ValueError("Each measurement must occur only once.")
        if self.operation == "difference" and len(self.records) != 2:
            raise ValueError("Difference requires [before, after] and computes after minus before.")
        return self


def calculate_values(measurements, operation):
    if len({(m.subject, m.metric, m.unit) for m in measurements}) != 1:
        raise SourceError("measurement_series_mismatch",
            "Select one subject, metric and exact unit. No unit conversion was assumed.")
    numbers = [Decimal(m.value) for m in measurements]
    with localcontext() as context:
        context.prec = 128
        if operation == "sum":
            value = sum(numbers, Decimal(0))
        elif operation == "difference":
            value = numbers[1] - numbers[0]
        elif operation == "minimum":
            value = min(numbers)
        else:
            value = max(numbers)
    return format(value, "f")


class MeasurementRepository:
    def __init__(self, sources, requests, *, owner_id, area_id, selections=None,
                 access_check=None, task_evidence=None):
        if sources.pool is not requests.pool:
            raise ValueError("Measurement sources and requests must share a database.")
        self.sources, self.requests = sources, requests
        self.owner_id, self.area_id = owner_id, area_id
        self.selections, self.access_check = selections, access_check
        self.task_evidence = task_evidence
        self.provenance = SourceReadProvenance(sources)

    def authorize(self, owner):
        if owner != self.owner_id:
            raise SourceError("measurement_owner_mismatch", "Measurements are outside this owner.", 403)
        if self.access_check is not None:
            self.access_check(owner)

    async def _scope(self, connection, session_id):
        self.authorize(self.owner_id)
        if session_id is not None and not await connection.fetchval(
                "select exists(select 1 from jarvis_sessions where owner_id=$1 and id=$2)",
                self.owner_id, session_id):
            raise SourceError("measurement_session_mismatch", "The session belongs to another owner.", 403)
        area = self.area_id
        if self.selections is not None:
            selected = await self.selections.read_in_transaction(connection, self.owner_id, session_id)
            area = UUID(selected["area_id"])
        await self.sources._area(connection, self.owner_id, area)
        return area

    async def _load(self, connection, area, identity):
        namespace = await connection.fetchval(
            "select namespace from jarvis_knowledge_sources where owner_id=$1 and area_id=$2 and id=$3",
            self.owner_id, area, identity)
        if namespace != NAMESPACE:
            raise SourceError("measurement_not_found", "No exact measurement exists in this scope.", 404)
        source = await self.sources._read(connection, self.owner_id, area, identity)
        if await connection.fetchval("select lock_jarvis_copy_source($1,$2)",
                                     self.owner_id, source.payload_id) != "available":
            raise SourceError("measurement_evidence_unavailable", "The measurement's original evidence is unavailable.")
        try:
            measurement = Measurement.model_validate(source.content["measurement"])
        except (KeyError, TypeError, ValidationError) as error:
            raise SourceError("measurement_original_invalid",
                "This source no longer contains a valid exact measurement; do not guess its values.") from error
        return source, measurement

    async def _result(self, connection, area, records, **extra):
        metadata = []
        for source, _ in records:
            item = source.original_evidence()
            item.pop("content")
            metadata.append(item)
        if self.task_evidence is not None:
            citations = await self.task_evidence.bind_in_transaction(connection, metadata)
            metadata = [item | {"task_evidence": citation}
                        for item, citation in zip(metadata, citations, strict=True)]
        self.authorize(self.owner_id)
        return {
            "records": [{"measurement_id":str(source.source_id), "revision":source.revision,
                         "measurement":measurement.model_dump(mode="json"), "source":item}
                        for (source, measurement), item in zip(records, metadata, strict=True)],
            "sources":metadata, "knowledge_scope":{"area_id":str(area)},
            "meaning":"Exact stored records formulated from user requests, not independent physical verification. "
                      "measured_on is the reported calendar date; source timestamps describe storage. "
                      "Historical originals remain available through explain_memory/list_memory_versions.",
            **extra,
        }

    async def save(self, session_id, request: SaveMeasurement):
        invocation = current_tool_invocation()
        if (session_id is None or self.task_evidence is not None or invocation.task is not None
                or invocation.provider_call_id is None or invocation.request_event_id is None):
            raise SourceError("measurement_request_required",
                "An exact finalized conversational request is required for measurement edits.", 403)
        marker = {"event_id":str(invocation.request_event_id), "revision":invocation.request_revision,
                  "slot":request.request_slot}
        parameters = digest(request.model_dump(mode="json"))
        async with self.sources.transaction(self.owner_id) as connection:
            area = await self._scope(connection, session_id)
            await self.requests.lock_owner(connection, self.owner_id)
            event = await self.requests._voice_source(connection, self.owner_id, TaskVoiceSource(
                invocation.request_event_id, invocation.request_revision, session_id))
            prior = await connection.fetchrow("""select s.id,s.area_id,v.revision,p.content->>'request_digest' as request_digest
                from jarvis_knowledge_sources s join jarvis_knowledge_source_versions v
                    on v.owner_id=s.owner_id and v.source_id=s.id
                join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
                where s.owner_id=$1 and s.namespace=$2 and p.content->'request'=$3::jsonb
                order by v.recorded_at limit 1""", self.owner_id, NAMESPACE, json.dumps(marker))
            if prior is not None:
                if prior["area_id"] != area or prior["request_digest"] != parameters:
                    raise SourceError("measurement_request_conflict",
                        "This request slot already wrote another measurement or scope; do not repeat it with changed arguments.")
                stored = await self._load(connection, area, prior["id"])
                if stored[0].revision != prior["revision"]:
                    raise SourceError("measurement_request_superseded",
                        "This request was already applied and later corrected. Do not restore its old values.")
                return await self._result(connection, area, [stored], request_replayed=True)
            content = measurement_content(request.measurement) | {
                "request": marker, "request_digest": parameters,
                "meaning": "Model-formulated from the linked user request; not independent measurement or proof."}
            if not await connection.fetchval("select lock_jarvis_copy_sources($1,$2::uuid[])",
                                             self.owner_id, [event["payload_id"]]):
                raise SourceError("measurement_request_unavailable", "The original request was removed.")
            payload = await connection.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available',$2::jsonb) returning id""", self.owner_id, json.dumps(content))
            await connection.execute("""insert into jarvis_context_payload_dependencies
                (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
                self.owner_id, event["payload_id"], payload)
            key = str(uuid5(NAMESPACE_URL, "jarvis:measurement:" + digest({"owner":self.owner_id, **marker})))
            source = await self.sources.publish_in_transaction(connection, self.owner_id, SourcePublication(
                area_id=area, namespace=NAMESPACE, source_key=key, payload_id=payload,
                source_type="measurement", trust="tool_receipt", source_time=datetime.now(UTC),
                time_basis="Storage time only; the reported measurement calendar date is measurement.measured_on.",
                session_id=session_id, expected_revision=0))
            return await self._result(connection, area, [(source, request.measurement)], request_replayed=False)

    async def listing(self, session_id, query: MeasurementQuery):
        async with self.sources.transaction(self.owner_id) as connection:
            area = await self._scope(connection, session_id)
            rows = await connection.fetch("""select s.id from jarvis_knowledge_sources s
                join jarvis_knowledge_source_versions v
                    on v.owner_id=s.owner_id and v.source_id=s.id and v.revision=s.current_revision
                join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
                where s.owner_id=$1 and s.area_id=$2 and s.namespace=$3 and s.status='active'
                    and p.availability='available'
                    and ($4::text is null or p.content->'measurement'->>'subject'=$4)
                    and ($5::text is null or p.content->'measurement'->>'metric'=$5)
                    and ($6::text is null or p.content->'measurement'->>'unit'=$6)
                    and ($7::text is null or p.content->'measurement'->>'measured_on'>=$7)
                    and ($8::text is null or p.content->'measurement'->>'measured_on'<=$8)
                order by p.content->'measurement'->>'measured_on',s.id limit $9 offset $10""",
                self.owner_id, area, NAMESPACE, query.subject, query.metric, query.unit,
                query.from_date.isoformat() if query.from_date else None,
                query.through_date.isoformat() if query.through_date else None, query.limit+1, query.offset)
            records = [await self._load(connection, area, row["id"]) for row in rows[:query.limit]]
            more = len(rows) > query.limit
            return await self._result(connection, area, records, has_more=more,
                next_offset=query.offset+query.limit if more else None, filters=query.model_dump(mode="json"),
                coverage="Current page only. Corrections can reorder pages; calculations require exact record revisions.")

    async def calculate(self, session_id, request: MeasurementCalculation):
        async with self.sources.transaction(self.owner_id) as connection:
            area = await self._scope(connection, session_id)
            records = [await self._load(connection, area, item.measurement_id) for item in request.records]
            if any(source.revision != item.revision for (source, _), item in zip(records, request.records, strict=True)):
                raise SourceError("measurement_revision_conflict", "A selected record changed; list current revisions.")
            measurements = [record for _, record in records]
            value = calculate_values(measurements, request.operation)
            return await self._result(connection, area, records, calculation={
                "operation":request.operation, "value":value, "unit":measurements[0].unit,
                "subject":measurements[0].subject, "metric":measurements[0].metric,
                "record_count":len(records), "coverage":"Only the explicitly selected records, not an inferred lifetime total."})

    async def capture_result(self, result, capture_context):
        capture = capture_context.capture
        self.authorize(capture.owner_id)
        if self.sources.pool is not capture.repository.pool:
            raise SourceError("measurement_capture_scope", "Measurement capture requires its original database.")
        area = UUID(result["knowledge_scope"]["area_id"])
        async def materialize(connection, originals):
            record = await capture.bound_tool_outcome(connection, capture_context.call_id,
                capture_context.tool_name, result, originals, publication_area_id=area)
            self.authorize(self.owner_id)
            return SourceBoundToolResult(result, record), record.payload_id
        return await self.provenance.bind(self.owner_id, area, result["sources"], materialize)


@dataclass(frozen=True)
class MeasurementToolModule:
    repository: MeasurementRepository
    name: ClassVar[str] = "measurements"

    def build_tools(self, context):
        self.repository.authorize(context.owner_id)
        def tool(name, model, method, description, *, mutation=False):
            async def invoke(arguments):
                try:
                    request = model.model_validate(arguments)
                    return success(await method(context.session_id, request))
                except ValidationError as error:
                    return failed("invalid_measurement_input", "; ".join(
                        ".".join(map(str, item["loc"])) + ": " + item["msg"]
                        for item in error.errors(include_input=False, include_url=False)))
                except (SourceError, TaskError) as error:
                    return failed(error.code, error.message)
            return ToolDefinition(name=name, input_schema=model.model_json_schema(), description=description,
                handler=invoke, mutates_state=mutation, audit_arguments=source_read_arguments,
                result_capture=self.repository.capture_result if self.repository.task_evidence is None else None,
                capture_after_commit=mutation)
        tools = [
            tool("list_measurements", MeasurementQuery, self.repository.listing,
                "List exact dated measurements in the current knowledge area. Filters use the stored subject, metric "
                "and unit labels exactly; omit filters to discover them. Follow next_offset while has_more is true. "
                "A page is not a complete history. Values are user-requested records, not verified physical facts."),
            tool("calculate_measurements", MeasurementCalculation, self.repository.calculate,
                "Compute an exact decimal sum, difference, minimum or maximum from 1-32 current measurement IDs "
                "and revisions returned by list_measurements. All must share subject, metric and exact unit. "
                "Difference takes [before, after] and returns after minus before. No implicit unit conversion. "
                "The result covers only selected records; never call a partial selection an all-time total."),
        ]
        if context.session_id is not None and self.repository.task_evidence is None:
            tools.append(tool("save_measurement", SaveMeasurement, self.repository.save,
                "Record an exact quantitative value requested by the user, with its reported calendar date, unit, "
                "subject and metric. Reuse the same series labels. This tool only creates NEW entries. Delegate corrections "
                "to the background expert with memory_edit; its durable result survives source-context refresh. "
                "Never create another entry instead of correcting one. Use different request_slot values for "
                "independent writes in one request; preserve every argument on retry. This is not a reminder, "
                "a diagnosis, or proof that an action happened. General project progress belongs in memory/notes.",
                mutation=True))
        return tools
