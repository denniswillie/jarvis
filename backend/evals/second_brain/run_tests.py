"""Opt-in real DB tests; fixtures use disposable schemas, not production tables."""
import argparse
import os
import unittest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-db", action="store_true")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--pattern", default="test_second_brain*.py")
    args = parser.parse_args()
    if args.live_db:
        if not args.manifest_dir:
            parser.error("--live-db requires --manifest-dir")
        from app.settings import JarvisSettings, is_configured_secret
        dsn = JarvisSettings().SUPABASE_DB_URL
        if not is_configured_secret(dsn):
            parser.error("SUPABASE_DB_URL is not configured")
        os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"] = dsn
        os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"] = args.manifest_dir
    suite = unittest.defaultTestLoader.discover("tests", pattern=args.pattern)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
