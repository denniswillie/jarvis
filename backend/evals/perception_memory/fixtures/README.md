# Synthetic visual fixtures

These JPEGs are deterministic test scenes generated with pinned
`Pillow==11.3.0` by `python -m evals.perception_memory.generate_fixtures`.
The generator uses Pillow's bundled default font rather than a host font, so
the same source produces the same bytes on macOS and Linux. They contain simple
shapes and text only; no personal photographs or production data are used.

`manifest.json` records dimensions, descriptions, and SHA-256 hashes. A hash
change is a deliberate baseline change and should be reviewed with the scenario
expectations. The fixtures cover static scenes, readable labels, short actions,
scene transitions, object locations, a distinctive artwork, and a conditional
brushing/flossing reminder.
