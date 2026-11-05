-- Batch the existing copy-source fence without per-source network round trips.
-- Creation ordering matches the dependency DAG, including ancestor/copy pairs.
create function lock_jarvis_copy_sources(p_owner text,p_sources uuid[]) returns boolean
language plpgsql set search_path from current as $$
declare source record; locked integer:=0;
begin
    if p_sources is null or cardinality(p_sources)=0 then return false; end if;
    for source in select id from jarvis_context_payloads
        where owner_id=p_owner and id=any(p_sources) order by created_at,id loop
        if lock_jarvis_copy_source(p_owner,source.id)<>'available' then return false; end if;
        locked:=locked+1;
    end loop;
    -- Missing, foreign, repeated and NULL references are not an authorized set.
    return locked=cardinality(p_sources);
end $$;

