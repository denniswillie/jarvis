import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {CaptionText} from '../src/Clip.mjs';
import {Root} from '../src/Root.mjs';
import {FIXTURE_LIMITS, validateClipProps, segmentVideoProps} from '../src/props.mjs';

function fixture(name = 'mr01-props-10s') {
  return JSON.parse(readFileSync(new URL(`../fixtures/${name}.json`, import.meta.url), 'utf8'));
}

test('accepts Python-produced 10s and 12s props without mutation or rounding', () => {
  for (const [name, frames] of [['mr01-props-10s', 240], ['mr02-props-12s', 288]]) {
    const original = fixture(name);
    const before = JSON.stringify(original);
    const validated = validateClipProps(original);
    assert.equal(validated.recipe.durationInFrames, frames);
    assert.equal(JSON.stringify(original), before);
    assert.deepEqual(validated, original);
    validated.recipe.segments[0].fromFrame = 0;
    assert.equal(original.recipe.segments[0].fromFrame, 48);
  }
});

test('uses half-open composition-frame trims and contiguous output positions', () => {
  const {recipe} = validateClipProps(fixture());
  const first = segmentVideoProps(recipe.segments[0], recipe.audio, '/local/inputs/hash.mp4');
  const second = segmentVideoProps(recipe.segments[1], recipe.audio, '/local/inputs/hash.mp4');
  assert.equal(first.trimBefore / 24, 2);
  assert.equal(first.durationInFrames / 24, 5);
  assert.equal(second.trimBefore / 24, 12);
  assert.equal(recipe.segments[1].outputFromFrame, 120);
  assert.equal(second.trimBefore + second.durationInFrames, 408);
  assert.equal(Object.hasOwn(first, 'trimAfter'), false);
  assert.equal(first.playbackRate, 1);
  assert.equal(first.style.objectFit, 'contain');
});

test('accepts a final 30s cut from a five-minute source without shifting source or output time', () => {
  const input = fixture();
  input.sources[0].durationMs = 300_000;
  input.recipe.durationInFrames = 720;
  input.recipe.segments = [{
    segmentId: 'final-30s', sourceId: input.sources[0].sourceId,
    fromFrame: 6480, toFrame: 7200, outputFromFrame: 0, durationInFrames: 720,
  }];
  const before = JSON.stringify(input);
  const checked = validateClipProps(input);
  const video = segmentVideoProps(checked.recipe.segments[0], checked.recipe.audio, '/local');
  assert.equal(video.trimBefore / 24, 270);
  assert.equal(video.durationInFrames / 24, 30);
  assert.equal(checked.recipe.segments[0].outputFromFrame, 0);
  assert.equal(video.muted, false);
  assert.equal(video.style.objectFit, 'contain');
  assert.equal(JSON.stringify(input), before);
  assert.equal(FIXTURE_LIMITS.maxInputBytes, 64 * 1024);
  assert.equal(FIXTURE_LIMITS.maxOutputFrames, 720);

  input.sources[0].durationMs = 300_001;
  assert.throws(() => validateClipProps(input), /source duration limit/);
  input.sources[0].durationMs = 300_000;
  input.recipe.segments[0].toFrame = 7203;
  input.recipe.segments[0].fromFrame = 6483;
  assert.throws(() => validateClipProps(input), /exceeds inspected duration/);
});

test('input expansion accepts previously overlong sources but never expands output', () => {
  const input = fixture();
  input.sources[0].durationMs = 60_001;
  assert.deepEqual(validateClipProps(input), input);
  input.sources[0].durationMs = 300_000;
  input.recipe.durationInFrames = 723;
  input.recipe.segments = [{
    segmentId: 'too-long', sourceId: input.sources[0].sourceId,
    fromFrame: 0, toFrame: 723, outputFromFrame: 0, durationInFrames: 723,
  }];
  assert.throws(() => validateClipProps(input), /output limit/);
});

test('keeps HTML-looking captions literal in actual React markup', () => {
  const input = fixture();
  const text = '<script>alert("x")</script> <img src="https://evil.test" onerror="x"> & {value}';
  input.recipe.captions[0].text = text;
  const checked = validateClipProps(input);
  assert.equal(checked.recipe.captions[0].text, text);
  const html = renderToStaticMarkup(React.createElement(CaptionText, {text}));
  assert.ok(html.includes('&lt;script&gt;'));
  assert.ok(html.includes('&lt;img'));
  assert.ok(html.includes('&amp; {value}'));
  assert.ok(!html.includes('<script>') && !html.includes('<img'));
});

test('accepts adjacent captions, unicode code points and preserves array order', () => {
  const input = fixture();
  input.recipe.captions = [
    {captionId: 'second', fromFrame: 5, toFrame: 6, text: 'é'.repeat(200)},
    {captionId: 'first', fromFrame: 0, toFrame: 5, text: '  literal\ntext  '},
  ];
  assert.deepEqual(validateClipProps(input).recipe.captions, input.recipe.captions);
});

test('preserve requires exactly one inspected mixed audio stream; mute is explicit', () => {
  for (const streams of [0, 2]) {
    const input = fixture(); input.sources[0].audioStreams = streams;
    assert.throws(() => validateClipProps(input), /one inspected mixed audio stream/);
    input.recipe.audio = 'mute';
    const checked = validateClipProps(input);
    assert.equal(segmentVideoProps(checked.recipe.segments[0], checked.recipe.audio, '/local').muted, true);
  }
  assert.equal(segmentVideoProps(fixture().recipe.segments[0], 'preserve', '/local').muted, false);
});

test('rejects external URL/path tricks and missing, duplicate or undeclared source mappings', () => {
  for (const asset of ['https://example.com/x.mp4', 'http://127.0.0.1:8000/private', 'file:///etc/passwd', '/tmp/file.mp4', '../input.mp4', 'inputs/%2e%2e/x.mp4', 'inputs/' + 'a'.repeat(64) + '.mp4?token=x']) {
    const input = fixture(); input.sources[0].assetPath = asset;
    assert.throws(() => validateClipProps(input), /local source only/);
  }
  for (const mutate of [
    (x) => { x.sources = []; },
    (x) => { x.sources.push(x.sources[0]); },
    (x) => { x.recipe.segments[0].sourceId = 'other'; },
    (x) => { x.sources[0].contentSha256 = 'b'.repeat(64); },
    (x) => { x.sources[0].url = 'https://example.com'; },
  ]) {
    const input = fixture(); mutate(input);
    assert.throws(() => validateClipProps(input));
  }
});

test('rejects malformed profile, timing, counts and duplicate IDs before composition', () => {
  for (const mutate of [
    (x) => { x.recipe.width = 1080; },
    (x) => { x.recipe.fps = 30; },
    (x) => { x.recipe.schemaVersion = true; },
    (x) => { x.recipe.fit = 'cover'; },
    (x) => { x.recipe.audio = 'auto'; },
    (x) => { x.recipe.segments = []; },
    (x) => { x.recipe.segments[0].fromFrame = 49; },
    (x) => { x.recipe.segments[0].toFrame = 48; },
    (x) => { x.recipe.segments[0].durationInFrames = 119; },
    (x) => { x.recipe.segments[0].fromFrame = '48'; },
    (x) => { x.recipe.segments[0].fromFrame = NaN; },
    (x) => { x.recipe.segments[1].outputFromFrame = 119; },
    (x) => { x.recipe.segments[1].segmentId = 'first'; },
    (x) => { x.recipe.durationInFrames = 241; },
    (x) => { x.recipe.durationInFrames = 721; },
    (x) => { x.sources[0].durationMs = 16000; },
    (x) => { x.sources[0].durationMs = 300001; },
    (x) => { x.recipe.templateCode = 'arbitrary'; },
    (x) => { x.recipe.segments[0].filter = 'drawtext=...'; },
    (x) => { x.recipe.segments = new Array(9).fill(x.recipe.segments[0]); },
  ]) {
    const input = fixture(); mutate(input);
    assert.throws(() => validateClipProps(input));
  }
});

test('rejects out-of-bounds, overlapping, malformed and executable caption fields', () => {
  for (const mutate of [
    (x) => { x.recipe.captions[0].fromFrame = -1; },
    (x) => { x.recipe.captions[0].toFrame = 241; },
    (x) => { x.recipe.captions[0].toFrame = 24; },
    (x) => { x.recipe.captions[0].text = 'x'.repeat(201); },
    (x) => { x.recipe.captions[0].text = '\ud800'; },
    (x) => { x.recipe.captions[0].text = '\0'; },
    (x) => { x.recipe.captions[0].text = {html: 'x'}; },
    (x) => { x.recipe.captions[0].dangerouslySetInnerHTML = {__html: 'x'}; },
    (x) => { x.recipe.captions.push({...x.recipe.captions[0], captionId: 'other'}); },
    (x) => { x.recipe.captions.push({...x.recipe.captions[0], fromFrame: 96, toFrame: 100}); },
    (x) => { x.recipe.captions = new Array(17).fill(x.recipe.captions[0]); },
  ]) {
    const input = fixture(); mutate(input);
    assert.throws(() => validateClipProps(input));
  }
});

test('composition derives metadata only from validated input and has no fixture fallback', () => {
  const root = Root();
  assert.equal(root.props.id, 'jarvis-clip-v1');
  assert.equal(root.props.defaultProps, undefined);
  assert.equal(root.props.calculateMetadata({props: fixture()}).durationInFrames, 240);
  assert.equal(root.props.calculateMetadata({props: fixture('mr02-props-12s')}).durationInFrames, 288);
  assert.throws(() => root.props.calculateMetadata({props: {}}));
});
