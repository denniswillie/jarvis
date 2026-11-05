-- 0050: exact retained/corrected targets on the existing media grant predicate.
-- Shared task/source/grant checks; explicit legacy and continuation target branches.
-- No tables, new ledger, policy defaults or changes to the explicit-request guard.
create function jarvis_media_accepted_proof(o text, area uuid, task uuid, action uuid,
    plan text, ref jsonb, kind text, edit uuid, lock_rows boolean default true) returns jsonb
language plpgsql set search_path from current as $$
declare
    result jsonb;
    -- Only the fixed lock suffix varies; all input values remain bound parameters.
    lookup_sql constant text := $media_proof$
    select jsonb_build_object('action_id',a.id::text,'grant_id',g.id::text,
        'scope_payload_id',s.id::text,'receipt_payload_id',p.id::text,
        'parent_scope_payload_id',parentp.id::text,'source_payload_id',original.id::text,
        'metadata_payload_id',metadata.id::text,'parameters_digest',a.parameters_digest)
    from jarvis_task_actions a
    join jarvis_background_tasks t on t.owner_id=a.owner_id and t.id=a.task_id
    join jarvis_task_grants attached on attached.owner_id=a.owner_id and attached.task_id=a.task_id and attached.grant_id=a.grant_id
    join jarvis_operation_grants g on g.owner_id=a.owner_id and g.id=a.grant_id
    join jarvis_context_payloads s on s.owner_id=g.owner_id and s.id=g.scope_payload_id
    join jarvis_context_payloads p on p.owner_id=a.owner_id and p.id=a.receipt_payload_id
    join jarvis_task_grants parent_attached on parent_attached.owner_id=t.owner_id and parent_attached.task_id=t.id
        and parent_attached.grant_id::text=s.content->>'parent_grant_id'
    join jarvis_operation_grants parentg on parentg.owner_id=parent_attached.owner_id and parentg.id=parent_attached.grant_id
        and parentg.capability='files.publish_generated'
    join jarvis_context_payloads parentp on parentp.owner_id=parentg.owner_id and parentp.id=parentg.scope_payload_id
    join jarvis_files f on f.owner_id=t.owner_id and f.id::text=$6->>'file_id'
    join jarvis_file_publications publication on publication.owner_id=f.owner_id and publication.file_id=f.id
        and publication.area_id=t.knowledge_area_id and publication.operation_key='task-generated-file:' || a.id::text
    join jarvis_context_payloads metadata on metadata.owner_id=publication.owner_id and metadata.id=publication.metadata_payload_id
    join jarvis_knowledge_sources source on source.owner_id=t.owner_id and source.area_id=t.knowledge_area_id
        and source.namespace='product:files' and source.source_key=f.id::text
    join jarvis_knowledge_source_versions v on v.owner_id=source.owner_id and v.source_id=source.id
        and v.revision=source.current_revision and v.document_id=$6->>'version_id'
    join jarvis_context_payloads original on original.owner_id=v.owner_id and original.id=v.payload_id
    join jarvis_knowledge_areas ar on ar.owner_id=t.owner_id and ar.id=t.knowledge_area_id
    where a.owner_id=$1 and a.task_id=$3 and a.id=$4 and a.status='accepted' and a.submission_token is not null
        and t.knowledge_area_id=$2 and ar.status='active' and source.status='active'
        and g.capability='files.publish_snapshot' and a.parameters_payload_id=g.scope_payload_id
        and a.operation_key='tool-grant:' || replace(g.id::text,'-','')
        and p.content->>'parameters_digest'=a.parameters_digest and p.content->>'tool'='publish_generated_file'
        and p.content#>>'{result,status}'='success' and p.content#>>'{result,file,id}'=$6->>'file_id'
        and p.content#>>'{result,file,memory_source,document_id}'=$6->>'version_id'
        and s.content->>'task_id'=$3::text and s.content#>>'{request,format}'=$7
        and s.content#>>'{inspection,format}'=$7
        and s.content#>>'{request,operation_key}'=(case when $7='json' then 'media-recipe:' else 'media-output:' end) || $5
        and s.content#>>'{inspection,content_sha256}'=$6->>'content_sha256'
        and s.content#>>'{inspection,size_bytes}'=f.size_bytes::text
        and f.filename=s.content#>>'{request,filename}' and f.description=s.content#>>'{request,description}'
        and f.filename='edit-' || replace($8::text,'-','') || '-' || left($5,12) || '.' || $7
        and f.content_type=(case when $7='json' then 'application/json' else 'video/mp4' end)
        and f.metadata#>>'{task_artifact,task_id}'=$3::text and f.metadata#>>'{task_artifact,action_id}'=$4::text
        and f.metadata#>>'{task_artifact,area_id}'=$2::text
        and f.metadata#>>'{generated_artifact,operation_key}'=s.content#>>'{request,operation_key}'
        and f.metadata#>>'{generated_artifact,evidence_payload_id}'=s.content->>'evidence_payload_id'
        and original.content->>'content_sha256'=$6->>'content_sha256'
        and original.content#>>'{record,id}'=f.id::text
        and publication.content_sha256=$6->>'content_sha256' and publication.size_bytes=f.size_bytes
        and publication.storage_bucket=f.storage_bucket and publication.storage_path=f.storage_path
        and publication.phase='available' and publication.upload_settled and not publication.delete_requested
        and original.availability='available' and metadata.availability='available'
        and s.availability='available' and p.availability='available' and parentp.availability='available'
        and ($7<>'json' or ($6->>'content_sha256'=$5 and f.size_bytes<=131072))
    $media_proof$;
begin
    if kind not in ('json','mp4') or plan !~ '^[0-9a-f]{64}$'
        or jsonb_typeof(ref) is distinct from 'object'
        or not(ref ?& array['file_id','version_id','content_sha256'])
        or ref-array['file_id','version_id','content_sha256']<>'{}'::jsonb then return null; end if;
    execute lookup_sql || (case when lock_rows then
        ' for share of a,t,g,s,p,parentg,parentp,f,publication,metadata,source,v,original,ar' else '' end)
        into result using o,area,task,action,plan,ref,kind,edit;
    return result;
end;
$$;

create function jarvis_media_correction_lineage(o text, capture uuid, recipe uuid, output uuid) returns boolean
language sql set search_path from current as $$
    with recursive ancestors(root,id) as (
        values(recipe,recipe),(output,output)
        union
        select a.root,d.source_payload_id from ancestors a
        join jarvis_context_payload_dependencies d on d.owner_id=o and d.dependent_payload_id=a.id
    )
    select exists(select 1 from ancestors where root=recipe and id=capture)
       and exists(select 1 from ancestors where root=output and id=recipe)
       and not exists(select 1 from ancestors a left join jarvis_context_payloads p on p.owner_id=o and p.id=a.id
                      where p.id is null or p.availability<>'available');
$$;

create or replace function jarvis_media_command_grant_valid(g jarvis_operation_grants) returns boolean
language plpgsql set search_path from current as $$
declare
    e record; t record; r record; parent record;
    b jsonb; c jsonb; q jsonb; pin jsonb; scope jsonb; original_pin jsonb;
    capture_session text; recording_key text;
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
    if jsonb_typeof(pin) is distinct from 'object' then return false; end if;
    if not(pin ? 'profile') then
        -- Original 0049 profile: same-session recording and exact original shape.
        if (c->'arguments')-array['instruction','recording_id','saved']<>'{}'::jsonb then return false; end if;
        capture_session:=e.session_id::text;
        recording_key:=c#>>'{arguments,recording_id}';
        original_pin:=pin;
    else
        if jsonb_typeof(pin) is distinct from 'object'
            or not(pin ?& array['profile','original','capture_session_id','command_session_id'])
            or pin->>'command_session_id' is distinct from e.session_id::text
            or pin#>>'{original,session_id}' is distinct from pin->>'capture_session_id' then return false; end if;
        if pin->>'profile'='retained_recording.v1' then
            if pin-array['profile','original','capture_session_id','command_session_id']<>'{}'::jsonb
                or (c->'arguments')-array['instruction','recording_id','saved','recording_profile']<>'{}'::jsonb
                or c#>>'{arguments,recording_profile}' is distinct from 'retained_recording.v1'
                or c#>>'{arguments,recording_id}' is distinct from pin#>>'{original,recording_id}'
                or c#>'{arguments,saved}' is distinct from ((pin->'original')-array['owner_id','session_id','recording_id',
                    'area_id','source_payload_id','receipt_payload_id','admission_payload_id','content_sha256','size_bytes'])
                then return false; end if;
        elsif pin->>'profile'='accepted_output.v1' then
            if not(pin ?& array['target','previous_recipe','edit_id','recipe_action_id','recipe_proof','output_proof'])
                or pin-array['profile','original','capture_session_id','command_session_id','target','previous_recipe',
                             'edit_id','recipe_action_id','recipe_proof','output_proof']<>'{}'::jsonb
                or (c->'arguments')-array['instruction','accepted_output']<>'{}'::jsonb
                or c#>'{arguments,accepted_output}' is distinct from pin->'target'
                or jsonb_typeof(pin->'target') is distinct from 'object'
                or not((pin->'target') ?& array['task_id','action_id','render_plan_sha256','file'])
                or (pin->'target')-array['task_id','action_id','render_plan_sha256','file']<>'{}'::jsonb
                or pin#>>'{previous_recipe,content_sha256}' is distinct from pin#>>'{target,render_plan_sha256}'
                or pin->>'recipe_action_id'=pin#>>'{target,action_id}' then return false; end if;
        else return false; end if;
        capture_session:=pin->>'capture_session_id';
        recording_key:=pin#>>'{original,recording_id}';
        original_pin:=pin->'original';
    end if;
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
    where rec.owner_id=g.owner_id and rec.session_id::text=capture_session and rec.area_id=t.knowledge_area_id
        and rec.recording_id::text=recording_key and not rec.revoked and rec.capture_closed
        and a.status='active' and s.status='active' and v.revision=s.current_revision
        and p.availability='available' and cp.availability='available' and ap.availability='available'
        and fpp.availability='available' and fp.phase='available' and fp.upload_settled and not fp.delete_requested
        and fp.operation_key='recording:' || rec.recording_id::text and fp.transfer_policy='mp4_capture_v1'
        and p.content->>'recording_id'=rec.recording_id::text and p.content->>'content_sha256'=fp.content_sha256
        and not exists(select 1 from jarvis_context_events n where n.owner_id=ae.owner_id
            and n.source_namespace=ae.source_namespace and n.source_event_id=ae.source_event_id and n.revision>ae.revision)
    for share of rec,a,v,s,p,cp,ap,fp,fpp,ce,ae;
    if not found then return false; end if;
    if not(pin ? 'profile') and c#>'{arguments,saved}' is distinct from
        jsonb_build_object('file_id',r.file_id::text,'source_id',r.source_id::text,
            'version_id',r.version_id,'receipt_event_id',r.completion_event_id::text) then return false; end if;
    if original_pin is distinct from jsonb_build_object('file_id',r.file_id::text,'source_id',r.source_id::text,
            'version_id',r.version_id,'receipt_event_id',r.completion_event_id::text,'owner_id',r.owner_id,
            'session_id',r.session_id::text,'recording_id',r.recording_id::text,'area_id',r.area_id::text,
            'source_payload_id',r.source_payload_id::text,'receipt_payload_id',r.completion_payload_id::text,
            'admission_payload_id',r.admission_payload_id::text,'content_sha256',r.content_sha256,'size_bytes',r.size_bytes)
        then return false; end if;
    if pin->>'profile'='accepted_output.v1' then
        if pin#>>'{target,task_id}'=t.id::text
            or jarvis_media_accepted_proof(g.owner_id,t.knowledge_area_id,(pin#>>'{target,task_id}')::uuid,
                (pin->>'recipe_action_id')::uuid,pin#>>'{target,render_plan_sha256}',pin->'previous_recipe',
                'json',(pin->>'edit_id')::uuid) is distinct from pin->'recipe_proof'
            or jarvis_media_accepted_proof(g.owner_id,t.knowledge_area_id,(pin#>>'{target,task_id}')::uuid,
                (pin#>>'{target,action_id}')::uuid,pin#>>'{target,render_plan_sha256}',pin#>'{target,file}',
                'mp4',(pin->>'edit_id')::uuid) is distinct from pin->'output_proof'
            or not jarvis_media_correction_lineage(g.owner_id,r.source_payload_id,
                (pin#>>'{recipe_proof,source_payload_id}')::uuid,(pin#>>'{output_proof,source_payload_id}')::uuid)
            then return false; end if;
    end if;
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
