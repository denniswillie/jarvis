-- Trusted per-version session association. No evidence-body or vendor-document rewrite.
create table jarvis_knowledge_source_sessions (
    owner_id text not null,
    source_id uuid not null,
    source_revision bigint not null check(source_revision>0),
    session_id uuid not null,
    created_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,source_id,source_revision),
    foreign key(owner_id,source_id,source_revision)
        references jarvis_knowledge_source_versions(owner_id,source_id,revision),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id) on delete cascade
);
create index jarvis_source_sessions_by_session
    on jarvis_knowledge_source_sessions(owner_id,session_id,source_id,source_revision);
create trigger jarvis_source_session_immutable before update on jarvis_knowledge_source_sessions
for each row execute function reject_jarvis_foundation_update();

-- Existing canonical associations were already admitted by the context boundary.
insert into jarvis_knowledge_source_sessions(owner_id,source_id,source_revision,session_id)
select cs.owner_id,cs.source_id,cs.source_revision,e.session_id
from jarvis_knowledge_context_sources cs
join jarvis_context_events e on e.owner_id=cs.owner_id and e.id=cs.event_id and e.revision=cs.event_revision
join jarvis_sessions j on j.owner_id=e.owner_id and j.id=e.session_id
on conflict(owner_id,source_id,source_revision) do nothing;

-- Only persisted importer receipts establish this provenance. Do not infer
-- session ownership from arbitrary model text, a vendor result or a changed legacy row.
insert into jarvis_knowledge_source_sessions(owner_id,source_id,source_revision,session_id)
select distinct r.owner_id,r.source_id,r.source_revision,j.id
from jarvis_legacy_import_records r
join jarvis_legacy_import_runs run on run.owner_id=r.owner_id and run.id=r.run_id
join jarvis_knowledge_source_versions v
    on v.owner_id=r.owner_id and v.source_id=r.source_id and v.revision=r.source_revision
join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id and p.availability='available'
join jarvis_sessions j on j.owner_id=r.owner_id and j.id::text=p.content#>>'{record,session_id}'
where r.disposition in ('imported','already_present','suppressed')
    and p.content#>>'{origin,system}'='jarvis_legacy'
    and p.content#>>'{origin,table}'=r.population
    and p.content#>>'{origin,key}'=r.origin_key
    and p.content#>>'{origin,mapping_version}'=run.mapping_version::text
on conflict(owner_id,source_id,source_revision) do nothing;
