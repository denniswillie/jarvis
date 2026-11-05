-- Existing imported-message identities permit content-free historical repair.
-- No original table rewrite, inferred mapping or change to import snapshot hashes.
create or replace function guard_jarvis_message_source() returns trigger
language plpgsql set search_path from current as $$
begin
    if tg_op='UPDATE' and old.source_payload_id is not null then
        if new.source_payload_id is distinct from old.source_payload_id or new.owner_id<>old.owner_id then
            raise exception 'A source-bound message cannot detach or change its source' using errcode='23514';
        end if;
        if new.content is distinct from old.content and new.content<>'' then
            raise exception 'A source-bound message can only redact its copied text' using errcode='23514';
        end if;
        if new.content='' then return new; end if;
    end if;
    if new.source_payload_id is not null
       and lock_jarvis_copy_source(new.owner_id,new.source_payload_id)<>'available' then
        -- A historical original may outlive an earlier deletion. Permit only
        -- content-free repair of that exact imported row, never a new copy or
        -- restoration of removed evidence. The host checks its full recorded
        -- input digest before choosing to erase it.
        if tg_op='UPDATE' and old.source_payload_id is null and new.content=''
           and (to_jsonb(new)-array['content','metadata','source_payload_id'])
               = (to_jsonb(old)-array['content','metadata','source_payload_id'])
           and exists(
               select 1 from jarvis_legacy_import_records r
               join jarvis_knowledge_source_versions v on v.owner_id=r.owner_id
                   and v.source_id=r.source_id and v.revision=r.source_revision
               join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
               where r.owner_id=new.owner_id and r.population='jarvis_messages'
                   and r.origin_key=new.id::text and r.disposition in ('imported','already_present')
                   and p.id=new.source_payload_id and p.availability in ('deleted','expired','unavailable')
                   and new.metadata=jsonb_build_object(
                       'response_kind','source_bound_unavailable','source_status',p.availability,
                       'source_payload_id',p.id)
           ) then
            return new;
        end if;
        raise exception 'A message copy requires an available owned source' using errcode='23514';
    end if;
    if new.metadata->>'response_kind'='expert_tts' and not exists(
        select 1 from jarvis_attention_intents i where i.owner_id=new.owner_id
            and i.id::text=new.metadata->>'intent_id' and i.source_payload_id=new.source_payload_id) then
        raise exception 'Expert message provenance must match its exact owned source' using errcode='23514';
    end if;
    return new;
end $$;

