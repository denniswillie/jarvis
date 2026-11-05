-- Forward-only: an exact native video command may mint only its managed
-- offline execution and two-file recipe/output grant family. Notes and ordinary
-- explicit user requests retain the 0046 guard verbatim below the new branch.
create function jarvis_media_command_grant_valid(g jarvis_operation_grants) returns boolean
language plpgsql set search_path from current as $$
declare
    e record; t record; r record; parent record;
    b jsonb; c jsonb; q jsonb; pin jsonb; scope jsonb;
begin
    if g.capability not in ('sandbox.execute','files.publish_generated','files.publish_snapshot')
        or g.request_revision<>1 or g.scope_revision<>1 or g.status<>'active'
        or g.expires_at<=clock_timestamp() then return false; end if;
    select ce.*,p.content as body into e from jarvis_context_events ce
    join jarvis_context_payloads p on p.owner_id=ce.owner_id and p.id=ce.payload_id
    where ce.owner_id=g.owner_id and ce.id=g.request_event_id and ce.revision=1
        and ce.source_namespace='voice_command' and ce.kind='command_request' and ce.trust='derived'
        and ce.finality='final' and ce.session_id is not null and p.availability='available'
        and not exists(select 1 from jarvis_context_events n where n.owner_id=ce.owner_id
            and n.source_namespace=ce.source_namespace and n.source_event_id=ce.source_event_id and n.revision>ce.revision)
    for share of ce,p;
    if not found then return false; end if;
    b:=e.body; c:=b->'command'; q:=b->'request'; pin:=b->'media_input';
    if jsonb_typeof(b) is distinct from 'object'
        or not (b ?& array['format','command','operation','task_id','base_revision','area_id','request','mode','parameters_digest','media_input'])
        or b-array['format','command','operation','task_id','base_revision','area_id','request','mode','parameters_digest','media_input']<>'{}'::jsonb
        or b->>'format' is distinct from 'jarvis.task_command.v1' or b->>'operation' is distinct from 'create'
        or b->'base_revision' is distinct from '0'::jsonb
        or b->'mode' is distinct from '"background"'::jsonb
        or coalesce(b->>'parameters_digest','') !~ '^[0-9a-f]{64}$'
        or jsonb_typeof(c) is distinct from 'object'
        or not(c ?& array['owner_id','session_id','provider_session_key','provider_call_id','operation_id','tool_name','capability','arguments_digest','arguments'])
        or c-array['owner_id','session_id','provider_session_key','provider_call_id','operation_id','tool_name','capability','arguments_digest','arguments']<>'{}'::jsonb
        or c->'owner_id' is distinct from to_jsonb(g.owner_id) or c->>'session_id' is distinct from e.session_id::text
        or c->>'operation_id' is distinct from e.source_event_id or c->>'tool_name' is distinct from 'start_media_task'
        or c->>'capability' is distinct from 'tasks.create'
        or jsonb_typeof(c->'provider_session_key') is distinct from 'string'
        or jsonb_typeof(c->'provider_call_id') is distinct from 'string'
        or coalesce(c->>'provider_session_key','') !~ '[^[:space:]]'
        or coalesce(c->>'provider_call_id','') !~ '[^[:space:]]'
        or coalesce(c->>'arguments_digest','') !~ '^[0-9a-f]{64}$'
        or jsonb_typeof(c->'arguments') is distinct from 'object'
        or (c->'arguments')-array['instruction','recording_id','saved']<>'{}'::jsonb
        or jsonb_typeof(c#>'{arguments,instruction}') is distinct from 'string'
        or coalesce(length(c#>>'{arguments,instruction}'),0) not between 1 and 8000
        or coalesce(c#>>'{arguments,instruction}','') !~ '[^[:space:]]'
        or jsonb_typeof(q) is distinct from 'object'
        or not(q ?& array['request_id','title','requested_outcome','session_id','knowledge_area_id','backend_kind','execution_environment','publish_artifacts','edit_memory','edit_note','grant_ids'])
        or q-array['request_id','title','requested_outcome','session_id','knowledge_area_id','backend_kind','execution_environment','publish_artifacts','edit_memory','edit_note','grant_ids']<>'{}'::jsonb
        or q->>'request_id' is distinct from e.source_event_id or q->>'session_id' is distinct from e.session_id::text
        or q->>'requested_outcome' is distinct from c#>>'{arguments,instruction}'
        or q->>'backend_kind' is distinct from 'frontier' or q->>'execution_environment' is distinct from 'offline_sandbox'
        or q->'publish_artifacts' is distinct from 'true'::jsonb or q->'edit_note' is distinct from 'false'::jsonb
        or q->'edit_memory' is distinct from 'false'::jsonb or q->'grant_ids' is distinct from '[]'::jsonb
        or q->>'knowledge_area_id' is distinct from b->>'area_id' then return false; end if;
    select bt.*,d.content as description into t from jarvis_background_tasks bt
    join jarvis_task_requests tr on tr.owner_id=bt.owner_id and tr.task_id=bt.id
    join jarvis_context_payloads d on d.owner_id=bt.owner_id and d.id=bt.description_payload_id
    where bt.owner_id=g.owner_id and bt.id::text=b->>'task_id' and bt.session_id=e.session_id
        and bt.backend_kind='frontier' and bt.knowledge_area_id::text=b->>'area_id'
        and tr.request_id::text=e.source_event_id and tr.operation='create'
        and tr.accepted_revision=1 and tr.accepted_revision<=bt.revision and tr.parameters_digest=b->>'parameters_digest'
        and d.availability='available' and d.content->>'requested_outcome'=q->>'requested_outcome'
        and d.content->>'title'=q->>'title' and d.content->'mode'=b->'mode'
        and d.content->'media_input'=pin
        and d.content->'instruction_request'=jsonb_build_object('event_id',e.id::text,'revision',1,'session_id',bt.session_id::text)
        and d.content#>>'{managed_plugin,format}'='jarvis.managed-media.v1'
        and d.content#>>'{managed_plugin,binding,owner_id}'=g.owner_id
        and d.content#>>'{managed_plugin,binding,plugin_id}'='video_editor'
        and d.content#>>'{managed_plugin,binding,configuration_id}'=d.content#>>'{managed_plugin,bundle,configuration_id}'
    for share of bt,d;
    if not found then return false; end if;
    select rec.*,v.payload_id as source_payload_id,fp.content_sha256,fp.size_bytes into r
    from jarvis_recordings rec
    join jarvis_knowledge_areas a on a.owner_id=rec.owner_id and a.id=rec.area_id
    join jarvis_knowledge_source_versions v on v.owner_id=rec.owner_id and v.document_id=rec.version_id
        and v.source_id=rec.source_id and v.area_id=rec.area_id
    join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
    join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
    join jarvis_context_payloads cp on cp.owner_id=rec.owner_id and cp.id=rec.completion_payload_id
    join jarvis_context_events ce on ce.owner_id=rec.owner_id and ce.id=rec.completion_event_id and ce.payload_id=cp.id
    join jarvis_context_payloads ap on ap.owner_id=rec.owner_id and ap.id=rec.admission_payload_id
    join jarvis_context_events ae on ae.owner_id=rec.owner_id and ae.id=rec.admission_event_id and ae.payload_id=ap.id
    join jarvis_file_publications fp on fp.owner_id=rec.owner_id and fp.id=rec.publication_id
        and fp.file_id=rec.file_id and fp.area_id=rec.area_id
    join jarvis_context_payloads fpp on fpp.owner_id=fp.owner_id and fpp.id=fp.metadata_payload_id
    where rec.owner_id=g.owner_id and rec.session_id=e.session_id and rec.area_id=t.knowledge_area_id
        and rec.recording_id::text=c#>>'{arguments,recording_id}' and not rec.revoked and rec.capture_closed
        and a.status='active' and s.status='active' and v.revision=s.current_revision
        and p.availability='available' and cp.availability='available' and ap.availability='available'
        and fpp.availability='available' and fp.phase='available' and fp.upload_settled and not fp.delete_requested
        and fp.operation_key='recording:' || rec.recording_id::text and fp.transfer_policy='mp4_capture_v1'
        and p.content->>'recording_id'=rec.recording_id::text and p.content->>'content_sha256'=fp.content_sha256
        and not exists(select 1 from jarvis_context_events n where n.owner_id=ae.owner_id
            and n.source_namespace=ae.source_namespace and n.source_event_id=ae.source_event_id and n.revision>ae.revision)
    for share of rec,a,v,s,p,cp,ap,fp,fpp,ce,ae;
    if not found then return false; end if;
    if c#>'{arguments,saved}' is distinct from jsonb_build_object('file_id',r.file_id::text,
        'source_id',r.source_id::text,'version_id',r.version_id,'receipt_event_id',r.completion_event_id::text)
        or pin is distinct from jsonb_build_object('file_id',r.file_id::text,'source_id',r.source_id::text,
            'version_id',r.version_id,'receipt_event_id',r.completion_event_id::text,'owner_id',r.owner_id,
            'session_id',r.session_id::text,'recording_id',r.recording_id::text,'area_id',r.area_id::text,
            'source_payload_id',r.source_payload_id::text,'receipt_payload_id',r.completion_payload_id::text,
            'admission_payload_id',r.admission_payload_id::text,'content_sha256',r.content_sha256,'size_bytes',r.size_bytes)
        then return false; end if;
    if exists(select 1 from jarvis_task_grants tg join jarvis_operation_grants og
        on og.owner_id=tg.owner_id and og.id=tg.grant_id where tg.owner_id=g.owner_id and tg.task_id=t.id
        and (og.capability not in ('sandbox.execute','files.publish_generated','files.publish_snapshot')
            or og.request_event_id<>e.id or og.request_revision<>1)) then return false; end if;
    select p.content into scope from jarvis_context_payloads p
    where p.owner_id=g.owner_id and p.id=g.scope_payload_id and p.availability='available' for share;
    if not found or jsonb_typeof(scope) is distinct from 'object' then return false; end if;
    if g.capability in ('sandbox.execute','files.publish_generated') then
        if t.revision<>1 or t.status<>'queued' or t.request_event_id<>e.id or t.request_revision<>1
            then return false; end if;
        if g.capability='sandbox.execute' then
            return g.operation_key='managed-video-editor:' || t.id::text
                and scope-array['template','lifetime_seconds','command_seconds','output_bytes','allowed_domains','internet_access']='{}'::jsonb
                and scope->>'template'='base' and scope->'allowed_domains'='[]'::jsonb
                and coalesce(scope->'internet_access','false'::jsonb)='false'::jsonb
                and jsonb_typeof(scope->'lifetime_seconds')='number' and (scope->>'lifetime_seconds')::integer between 60 and 3600
                and jsonb_typeof(scope->'command_seconds')='number' and (scope->>'command_seconds')::integer between 1 and 300
                and jsonb_typeof(scope->'output_bytes')='number' and (scope->>'output_bytes')::integer between 1024 and 256000;
        end if;
        return g.operation_key='artifact-task:' || t.id::text
            and scope-array['max_files','max_file_bytes','max_total_bytes','formats','mp4_max_file_bytes']='{}'::jsonb
            and scope->'formats'='["json","mp4"]'::jsonb and scope->'max_files'='2'::jsonb
            and jsonb_typeof(scope->'max_file_bytes')='number' and (scope->>'max_file_bytes')::bigint between 1 and 8388608
            and jsonb_typeof(scope->'max_total_bytes')='number' and (scope->>'max_total_bytes')::bigint between 1 and 134217728
            and jsonb_typeof(scope->'mp4_max_file_bytes')='number' and (scope->>'mp4_max_file_bytes')::bigint between 1 and 33554432;
    end if;
    if not(scope ?& array['request','inspection','parent_grant_id','task_id','evidence_payload_id'])
        or scope-array['request','inspection','parent_grant_id','task_id','evidence_payload_id','transfer_limit_bytes']<>'{}'::jsonb
        or scope->>'task_id' is distinct from t.id::text
        or scope#>>'{request,format}' not in ('json','mp4')
        or scope#>>'{request,format}' is distinct from scope#>>'{inspection,format}'
        or g.operation_key !~ ('^artifact-snapshot:' || replace(t.id::text,'-','') || ':[0-9a-f]{64}$')
        then return false; end if;
    select og.*,p.content as scope into parent from jarvis_task_grants tg
    join jarvis_operation_grants og on og.owner_id=tg.owner_id and og.id=tg.grant_id
    join jarvis_context_payloads p on p.owner_id=og.owner_id and p.id=og.scope_payload_id
    where tg.owner_id=g.owner_id and tg.task_id=t.id and og.id::text=scope->>'parent_grant_id'
        and og.capability='files.publish_generated' and og.operation_key='artifact-task:' || t.id::text
        and og.request_event_id=e.id and og.request_revision=1 and og.scope_revision=1
        and og.status='active' and og.expires_at>clock_timestamp() and p.availability='available' for share of og,p;
    if not found or g.expires_at>parent.expires_at then return false; end if;
    return exists(select 1 from jarvis_context_payloads p where p.owner_id=g.owner_id
        and p.id::text=scope->>'evidence_payload_id' and p.availability='available');
end;
$$;

-- Admit only the account-note grant family from a durable native-command
-- create receipt. Ordinary grants and capability preferences retain their
-- existing finalized-user-input requirement. No trigger is disabled.
create or replace function guard_jarvis_explicit_request() returns trigger
language plpgsql set search_path from current as $$
declare
    target_payload uuid;
    request_event record;
    task_record record;
    parent_record record;
    body jsonb;
    command jsonb;
    request jsonb;
    scope jsonb;
    expected_parent jsonb;
    owner_hash bytea;
    canonical_note uuid;
begin
    -- Preserve authority/input-revision serialization from the original guard.
    perform 1 from jarvis_context_streams where owner_id=new.owner_id for update;
    if not exists (
        select 1 from jarvis_context_events e join jarvis_context_payloads p
            on p.owner_id=e.owner_id and p.id=e.payload_id
        where e.owner_id=new.owner_id and e.id=new.request_event_id and e.revision=new.request_revision
            and e.trust='user_input' and e.kind in ('user_request','user_transcript')
            and e.finality in ('final','revised') and p.availability='available'
            and not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
                and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
                and newer.revision>e.revision)
    ) then
        -- Preferences never receive the command-derived exception. Keep this
        -- branch separate: their NEW record has no grant-only scope columns.
        if tg_table_name<>'jarvis_operation_grants' then
            raise exception 'Explicit authority requires a current finalized user request' using errcode='23514';
        end if;
        if new.capability in ('sandbox.execute','files.publish_generated','files.publish_snapshot')
            and jarvis_media_command_grant_valid(new) is true then
            new.created_at:=clock_timestamp();
            return new;
        end if;
        if new.capability not in ('notes.edit_account','notes.publish_account_edit')
            or new.request_revision<>1 or new.scope_revision<>1 or new.status<>'active'
            or new.expires_at<=clock_timestamp() then
            raise exception 'Explicit authority requires a current finalized user request' using errcode='23514';
        end if;

        select e.*,p.content as body into request_event
        from jarvis_context_events e join jarvis_context_payloads p
            on p.owner_id=e.owner_id and p.id=e.payload_id
        where e.owner_id=new.owner_id and e.id=new.request_event_id and e.revision=1
            and e.source_namespace='voice_command' and e.kind='command_request'
            and e.trust='derived' and e.finality='final' and p.availability='available'
            and e.session_id is not null
            and not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
                and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
                and newer.revision>e.revision)
        for share of e,p;
        if not found then
            raise exception 'Account-note grants require the original available command' using errcode='23514';
        end if;
        body:=request_event.body;
        command:=body->'command';
        request:=body->'request';
        if jsonb_typeof(body) is distinct from 'object'
            or not (body ?& array['format','command','operation','task_id','base_revision','area_id','request','mode','parameters_digest'])
            or body-array['format','command','operation','task_id','base_revision','area_id','request','mode','parameters_digest']<>'{}'::jsonb
            or body->>'format' is distinct from 'jarvis.task_command.v1'
            or body->>'operation' is distinct from 'create'
            or jsonb_typeof(body->'parameters_digest') is distinct from 'string'
            or coalesce(body->>'parameters_digest','') !~ '^[0-9a-f]{64}$'
            or body->'base_revision' is distinct from '0'::jsonb
            or body->>'base_revision' is distinct from '0'
            or (body->'mode' is distinct from 'null'::jsonb and body->'mode' is distinct from '"background"'::jsonb)
            or jsonb_typeof(command) is distinct from 'object'
            or not (command ?& array['owner_id','session_id','provider_session_key','provider_call_id','operation_id','tool_name','capability','arguments_digest','arguments'])
            or command-array['owner_id','session_id','provider_session_key','provider_call_id','operation_id','tool_name','capability','arguments_digest','arguments']<>'{}'::jsonb
            or command->'owner_id' is distinct from to_jsonb(new.owner_id)
            or command->>'session_id' is distinct from request_event.session_id::text
            or command->>'operation_id' is distinct from request_event.source_event_id
            or command->>'tool_name' is distinct from 'live_note'
            or command->>'capability' is distinct from 'tasks.create'
            or jsonb_typeof(command->'provider_session_key') is distinct from 'string'
            or jsonb_typeof(command->'provider_call_id') is distinct from 'string'
            or coalesce(length(btrim(command->>'provider_session_key')),0) not between 1 and 512
            or coalesce(length(btrim(command->>'provider_call_id')),0) not between 1 and 200
            or coalesce(command->>'provider_session_key','') !~ '[^[:space:]]'
            or coalesce(command->>'provider_call_id','') !~ '[^[:space:]]'
            or jsonb_typeof(command->'arguments_digest') is distinct from 'string'
            or coalesce(command->>'arguments_digest','') !~ '^[0-9a-f]{64}$'
            or jsonb_typeof(command->'arguments') is distinct from 'object'
            or (command->'arguments')-array['instruction']<>'{}'::jsonb
            or jsonb_typeof(command#>'{arguments,instruction}') is distinct from 'string'
            or coalesce(length(command#>>'{arguments,instruction}'),0) not between 1 and 8000
            or coalesce(command#>>'{arguments,instruction}','') !~ '[^[:space:]]'
            or jsonb_typeof(request) is distinct from 'object'
            or not (request ?& array['request_id','title','requested_outcome','session_id','knowledge_area_id','backend_kind','execution_environment','publish_artifacts','edit_memory','edit_note','grant_ids'])
            or request-array['request_id','title','requested_outcome','session_id','knowledge_area_id','backend_kind','execution_environment','publish_artifacts','edit_memory','edit_note','grant_ids']<>'{}'::jsonb
            or request->>'request_id' is distinct from request_event.source_event_id
            or request->>'session_id' is distinct from request_event.session_id::text
            or jsonb_typeof(request->'requested_outcome') is distinct from 'string'
            or request->>'requested_outcome' is distinct from command#>>'{arguments,instruction}'
            or jsonb_typeof(request->'title') is distinct from 'string'
            or coalesce(length(request->>'title'),0) not between 1 and 200
            or coalesce(request->>'title','') !~ '[^[:space:]]'
            or request->>'backend_kind' is distinct from 'frontier'
            or request->>'execution_environment' is distinct from 'none'
            or request->'publish_artifacts' is distinct from 'false'::jsonb
            or request->'edit_memory' is distinct from 'false'::jsonb
            or request->'edit_note' is distinct from 'true'::jsonb
            or request->'grant_ids' is distinct from '[]'::jsonb then
            raise exception 'Account-note grants require an exact English note create command' using errcode='23514';
        end if;

        -- The task and immutable create receipt exist before the parent grant
        -- is inserted. An unreceipted context event cannot mint authority.
        select t.*,d.content as description into task_record
        from jarvis_background_tasks t
        join jarvis_task_requests r on r.owner_id=t.owner_id and r.task_id=t.id
        join jarvis_context_payloads d on d.owner_id=t.owner_id and d.id=t.description_payload_id
        join jarvis_sessions s on s.owner_id=t.owner_id and s.id=t.session_id
        join jarvis_knowledge_areas a on a.owner_id=t.owner_id and a.id=t.knowledge_area_id
        where t.owner_id=new.owner_id and t.id::text=body->>'task_id'
            and t.session_id=request_event.session_id and t.backend_kind='frontier'
            and t.knowledge_area_id::text=body->>'area_id' and a.status='active'
            and request->>'knowledge_area_id'=t.knowledge_area_id::text
            and r.request_id::text=request_event.source_event_id and r.operation='create'
            and r.accepted_revision=1 and r.accepted_revision<=t.revision
            and r.parameters_digest=body->>'parameters_digest'
            and d.availability='available'
            and d.content->>'requested_outcome'=request->>'requested_outcome'
            and d.content->>'title'=request->>'title'
            and d.content->'mode'=body->'mode'
            and d.content->'instruction_request'=jsonb_build_object(
                'event_id',request_event.id::text,'revision',1,'session_id',t.session_id::text)
        for share of t,d,s,a;
        if not found then
            raise exception 'Account-note grants require the matching task create receipt and effective instruction' using errcode='23514';
        end if;
        if exists(select 1 from jarvis_task_grants tg join jarvis_operation_grants g
            on g.owner_id=tg.owner_id and g.id=tg.grant_id
            where tg.owner_id=new.owner_id and tg.task_id=task_record.id
                and (g.capability not in ('notes.edit_account','notes.publish_account_edit')
                    or g.request_event_id<>request_event.id or g.request_revision<>1)) then
            raise exception 'Account-note grants cannot carry unrelated task authority' using errcode='23514';
        end if;

        select p.content into scope from jarvis_context_payloads p
        where p.owner_id=new.owner_id and p.id=new.scope_payload_id and p.availability='available'
        for share;
        if not found or jsonb_typeof(scope) is distinct from 'object' then
            raise exception 'Authority scope requires an available immutable payload' using errcode='23514';
        end if;
        -- Python uuid5(NAMESPACE_URL, 'jarvis.account-note:v1:' +
        -- canonical_json([owner], ensure_ascii=False)). A one-string JSON array
        -- needs no separator rewriting; preserve Unicode and exact owner bytes.
        owner_hash:=digest(decode('6ba7b8119dad11d180b400c04fd430c8','hex') ||
            convert_to('jarvis.account-note:v1:' || jsonb_build_array(new.owner_id)::text,'UTF8'),'sha1');
        owner_hash:=set_byte(owner_hash,6,(get_byte(owner_hash,6) & 15) | 80);
        owner_hash:=set_byte(owner_hash,8,(get_byte(owner_hash,8) & 63) | 128);
        canonical_note:=encode(substring(owner_hash from 1 for 16),'hex')::uuid;
        expected_parent:=jsonb_build_object('task_id',task_record.id::text,
            'artifact_id',canonical_note::text,'area_id',task_record.knowledge_area_id::text,'max_publications',1);

        if new.capability='notes.edit_account' then
            if scope<>expected_parent or scope->>'max_publications'<>'1'
                or new.operation_key<>'account-note-task:' || replace(task_record.id::text,'-','')
                or task_record.revision<>1 or task_record.status<>'queued'
                or task_record.request_event_id<>request_event.id or task_record.request_revision<>1
                or exists(select 1 from jarvis_task_grants tg where tg.owner_id=new.owner_id and tg.task_id=task_record.id) then
                raise exception 'Account-note parent scope must match its fresh accepted task' using errcode='23514';
            end if;
        else
            -- Do not duplicate the document/patch validator in SQL. The child
            -- is a frozen bounded scope; its exact candidate and evidence are
            -- checked again by the task/action and note publication owners.
            if not (scope ?& array['task_id','artifact_id','base_version_id','changes','evidence','candidate_digest'])
                or scope-array['task_id','artifact_id','base_version_id','changes','evidence','candidate_digest']<>'{}'::jsonb
                or scope->>'task_id' is distinct from task_record.id::text
                or scope->>'artifact_id' is distinct from canonical_note::text
                or (scope->'base_version_id' is distinct from 'null'::jsonb and (
                    jsonb_typeof(scope->'base_version_id') is distinct from 'string'
                    or length(scope->>'base_version_id') not between 1 and 200))
                or jsonb_typeof(scope->'changes') is distinct from 'array'
                or jsonb_typeof(scope->'evidence') is distinct from 'array'
                or jsonb_typeof(scope->'candidate_digest') is distinct from 'string'
                or coalesce(scope->>'candidate_digest','') !~ '^[0-9a-f]{64}$'
                or new.operation_key !~ '^account-note-edit:[0-9a-f]{64}$' then
                raise exception 'Account-note publication scope must name its exact canonical target' using errcode='23514';
            end if;
            if jsonb_array_length(scope->'changes') not between 1 and 32
                or jsonb_array_length(scope->'evidence')>128 then
                raise exception 'Account-note publication scope exceeds its bounds' using errcode='23514';
            end if;
            select g.id,g.expires_at into parent_record
            from jarvis_task_grants tg join jarvis_operation_grants g
                on g.owner_id=tg.owner_id and g.id=tg.grant_id
            join jarvis_context_payloads p on p.owner_id=g.owner_id and p.id=g.scope_payload_id
            where tg.owner_id=new.owner_id and tg.task_id=task_record.id
                and g.capability='notes.edit_account' and g.request_event_id=request_event.id and g.request_revision=1
                and g.scope_revision=1 and g.status='active' and g.expires_at>clock_timestamp()
                and g.operation_key='account-note-task:' || replace(task_record.id::text,'-','')
                and p.availability='available' and p.content=expected_parent and p.content->>'max_publications'='1'
            for share of g,p;
            if not found or new.expires_at>parent_record.expires_at then
                raise exception 'Account-note publication requires its attached active exact parent' using errcode='23514';
            end if;
            if (select count(*) from jarvis_task_grants tg join jarvis_operation_grants g
                on g.owner_id=tg.owner_id and g.id=tg.grant_id
                where tg.owner_id=new.owner_id and tg.task_id=task_record.id and g.capability='notes.edit_account')<>1 then
                raise exception 'Account-note publication has competing parents' using errcode='23514';
            end if;
        end if;
    end if;
    -- Original scope-availability and creation-time behavior for every caller.
    if tg_table_name='jarvis_operation_grants' then target_payload:=new.scope_payload_id;
    else target_payload:=new.policy_payload_id; end if;
    if not exists(select 1 from jarvis_context_payloads where owner_id=new.owner_id
        and id=target_payload and availability='available') then
        raise exception 'Authority scope requires an available immutable payload' using errcode='23514';
    end if;
    new.created_at:=clock_timestamp();
    return new;
end;
$$;
