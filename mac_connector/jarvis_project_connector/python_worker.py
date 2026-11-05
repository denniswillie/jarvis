"""Single-request Jedi process: snapshot files only, no project imports executed."""
import json
from pathlib import Path
import sys

import jedi


def main():
    payload = json.load(sys.stdin)
    request, files = payload["request"], payload["files"]
    # Parent-owned directory is removed even when this process times out or crashes.
    root = Path(sys.argv[1])
    jedi.settings.use_filesystem_cache = False
    jedi.settings.cache_directory = str(root / ".jedi-cache")
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    project = jedi.Project(root, environment_path=sys.executable, sys_path=[str(root)],
                           smart_sys_path=False, load_unsafe_extensions=False)
    if request["operation"] == "symbols":
        names = project.search(request["query"], all_scopes=True)
    else:
        path = root / request["path"]
        script = jedi.Script(files[request["path"]], path=path, project=project)
        if request["operation"] == "definition":
            names = script.goto(request["line"], request["column"]-1, follow_imports=True)
        else:
            names = script.get_references(request["line"], request["column"]-1,
                                           scope="project", include_builtins=False)
    items, truncated = [], False
    for name in names:
        if name.module_path is None or name.line is None:
            continue
        try:
            path = name.module_path.relative_to(root).as_posix()
        except ValueError:
            continue
        if path not in files:
            continue
        if len(items) >= request["limit"]:
            truncated = True
            break
        items.append({"path": path, "line": name.line, "column": name.column+1,
                      "name": name.name, "kind": name.type,
                      "is_definition": name.is_definition()})
    print(json.dumps({"items": items, "truncated": truncated}))


if __name__ == "__main__":
    main()
