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
