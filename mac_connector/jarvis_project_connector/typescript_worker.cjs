// The maintained compiler receives only an in-memory snapshot; it cannot load repo plugins.
"use strict";
const fs = require("node:fs");
const ts = require(process.argv[2]);
const payload = JSON.parse(fs.readFileSync(0, "utf8"));
const request = payload.request;
const prefix = "/jarvis-project/";
const files = new Map(Object.entries(payload.files).map(([name, text]) => [prefix + name, text]));
const options = {target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.CommonJS,
  moduleResolution: ts.ModuleResolutionKind.Node10, allowJs: true, jsx: ts.JsxEmit.Preserve,
  noLib: true, noEmit: true};
const host = {
  getCompilationSettings: () => options,
  getScriptFileNames: () => [...files.keys()],
  getScriptVersion: () => "1",
  getScriptSnapshot: name => files.has(name) ? ts.ScriptSnapshot.fromString(files.get(name)) : undefined,
  getCurrentDirectory: () => prefix,
  getDefaultLibFileName: () => "",
  fileExists: name => files.has(name),
  readFile: name => files.get(name),
  readDirectory: () => [...files.keys()],
  directoryExists: name => [...files.keys()].some(file => file.startsWith(name.replace(/\/$/, "") + "/")),
  useCaseSensitiveFileNames: () => true,
};
const service = ts.createLanguageService(host);
const rows = [];
function add(name, span, kind, label, isDefinition) {
  if (!files.has(name)) return;
  const text = files.get(name);
  const before = text.slice(0, span.start);
  const lines = before.split("\n");
  rows.push({path: name.slice(prefix.length), line: lines.length,
    column: [...lines[lines.length-1]].length + 1, kind, name: label || "",
    is_definition: Boolean(isDefinition)});
}
try {
  if (request.operation === "symbols") {
    for (const item of service.getNavigateToItems(request.query, request.limit + 1)) {
      add(item.fileName, item.textSpan, item.kind, item.name, true);
    }
  } else {
    const fileName = prefix + request.path;
    const lines = files.get(fileName).split("\n");
    let position = 0;
    for (let i = 0; i < request.line - 1; i++) position += lines[i].length + 1;
    position += [...lines[request.line-1]].slice(0, request.column-1).join("").length;
    if (request.operation === "definition") {
      for (const item of service.getDefinitionAtPosition(fileName, position) || []) {
        add(item.fileName, item.textSpan, item.kind, item.name, true);
      }
    } else {
      for (const group of service.findReferences(fileName, position) || []) {
        for (const item of group.references) {
          add(item.fileName, item.textSpan, group.definition.kind, group.definition.name, item.isDefinition);
        }
      }
    }
  }
  process.stdout.write(JSON.stringify({items: rows.slice(0, request.limit),
    truncated: rows.length > request.limit, engine_version: ts.version}));
} finally {
  service.dispose();
}
