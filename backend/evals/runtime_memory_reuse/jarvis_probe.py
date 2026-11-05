"""Two bounded native Jarvis memory controls on a disposable local database.

This reuses the existing pipeline harness. It is not the full shared-corpus
comparison: only coding/uncertainty controls are seeded, with no real media.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import inspect
import json
import os
from pathlib import Path
import sys

from environment import root_path, save


async def run(root):
    os.chdir(root)
    sys.path.insert(0, "/Users/denniswillie/gitbio/jarvis/backend")
    import asyncpg
    from dotenv import dotenv_values
    from app.settings import JarvisSettings
    from app.runtime.concurrency import PriorityAsyncLimiter
    from app.model_providers.gemini_multimodal_embeddings import GeminiMultimodalEmbeddingModel, gemini_multimodal_embedding_config_from_settings
    from evals.glasses_reliability.memory_pipeline_runtime import PipelineRuntime, FixtureLedger
    from evals.glasses_reliability.memory_grounding_probe import Case
    from evals.glasses_reliability.memory_pipeline_probe import PipelineRecordingClient, snapshot

    config = json.loads((root / "private-config.json").read_text())
    dsn = f"postgresql://postgres:{config['pg_password']}@127.0.0.1:{config['pg_port']}/jarvis_baseline"
    values = dotenv_values("/Users/denniswillie/gitbio/jarvis/backend/.env")
    settings = JarvisSettings(_env_file=None, SUPABASE_DB_URL=dsn,
                             OPENAI_API_KEY=values.get("OPENAI_API_KEY"), GEMINI_API_KEY=values.get("GEMINI_API_KEY"),
                             OPENAI_CONSOLIDATION_MODEL="gpt-5.4-mini")
    directory = root / "baseline"
    directory.mkdir(exist_ok=True)
    ledger = FixtureLedger(directory / "fixture.json")
    report = {"evidence": "native Jarvis extraction/retrieval control", "full_shared_corpus_comparison": False,
              "runs": [], "jobs": [], "errors": [], "cleanup": "run-owned local DB removed by environment teardown"}
    corpus = json.loads(Path(__file__).with_name("fixtures.json").read_text())

    def persist():
        save(root / "reports/jarvis-baseline.json", report)

    client = embeddings = pool = None
    try:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3, command_timeout=60)
        client = PipelineRecordingClient(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_CONSOLIDATION_MODEL,
                                         timeout_seconds=90, limiter=PriorityAsyncLimiter(1))
        embeddings = GeminiMultimodalEmbeddingModel(gemini_multimodal_embedding_config_from_settings(settings))
        runtime = PipelineRuntime(pool=pool, settings=settings, client=client, embedding_model=embeddings, ledger=ledger)
        owner = sorted(ledger.owners)[0]
        report["owner"] = owner
        report["models"] = {"consolidation": client.model, "embedding": embeddings.model}
        for source, question in ((corpus["sources"][0], corpus["questions"][0]), (corpus["sources"][6], corpus["questions"][4])):
            base = datetime.now(UTC)
            case = Case(name=source["id"], task=question["query"], user_text=source["text"],
                        observation="No additional camera evidence is available in this synthetic control.")
            seeded = await runtime.seed_event(owner=owner, case=case, ended_at=base)
            row = {"id": question["id"], "seeded": str(seeded)}
            report["runs"].append(row)
            await asyncio.wait_for(runtime.drain(owner=owner, journal=report["jobs"], persist=persist), timeout=240)
            row["snapshot"] = await snapshot(runtime, owner=owner, base=base, query=question["query"])
            persist()
        report["native_calls"] = client.native_calls
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        for resource in (client, embeddings):
            if resource is not None:
                method = getattr(resource, "aclose", None) or getattr(resource, "close", None)
                if method:
                    result = method()
                    if inspect.isawaitable(result):
                        await result
        if pool is not None:
            await pool.close()
        persist()
    print(json.dumps({"runs": len(report["runs"]), "errors": report["errors"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    asyncio.run(run(root_path(parser.parse_args().root)))
