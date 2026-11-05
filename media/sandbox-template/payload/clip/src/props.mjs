// Data only. This checks an internal rendering envelope; it issues no source grant.
export const TEMPLATE_ID = 'jarvis.clip.v1';
export const COMPOSITION_ID = 'jarvis-clip-v1'; // Remotion IDs cannot contain dots.
export const FPS = 24;
export const WIDTH = 720;
export const HEIGHT = 1280;
export const FIXTURE_LIMITS = Object.freeze({
  maxInputBytes: 64 * 1024,
  maxSourceBytes: 256 * 1024 * 1024,
  maxSourceDurationMs: 300_000,
  maxOutputFrames: 30 * FPS,
});

function requireThat(condition, message) {
  if (!condition) throw new Error(`Invalid clip input: ${message}`);
}

function record(value, keys, name) {
  requireThat(value !== null && typeof value === 'object' && !Array.isArray(value), name);
  requireThat(Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null, name);
  const actual = Reflect.ownKeys(value);
  requireThat(actual.length === keys.length && actual.every((key) => keys.includes(key)), `${name} fields`);
  requireThat(actual.every((key) => Object.hasOwn(Object.getOwnPropertyDescriptor(value, key), 'value')), `${name} must be data`);
}

function integer(value, name, minimum = 0) {
  requireThat(Number.isSafeInteger(value) && value >= minimum, name);
}

function identifier(value, name) {
  requireThat(typeof value === 'string' && /^[A-Za-z0-9_-]{1,80}$/.test(value), name);
}

function collection(value, min, max, name) {
  requireThat(Array.isArray(value) && value.length >= min && value.length <= max, name);
}

function unique(values, name) {
  requireThat(new Set(values).size === values.length, `duplicate ${name}`);
}

function literalText(value) {
  requireThat(typeof value === 'string' && [...value].length <= 200 && !value.includes('\0') && value.isWellFormed(), 'caption text');
}

/**
 * `recipe` is exactly ValidatedMediaRecipe.template_props(). `sources` is supplied
 * separately by trusted staging, never a model URL. These local observations do
 * not prove access rights or decoded media; Core must establish those upstream.
 */
export function validateClipProps(input) {
  record(input, ['recipe', 'sources'], 'envelope');
  const {recipe, sources} = input;
  record(recipe, ['schemaVersion', 'width', 'height', 'fps', 'durationInFrames', 'fit', 'audio', 'segments', 'captions'], 'recipe');
  requireThat(recipe.schemaVersion === 1 && recipe.width === WIDTH && recipe.height === HEIGHT && recipe.fps === FPS, 'fixed profile');
  requireThat(recipe.fit === 'contain' && ['preserve', 'mute'].includes(recipe.audio), 'fit/audio');
  integer(recipe.durationInFrames, 'durationInFrames', 1);
  requireThat(recipe.durationInFrames <= FIXTURE_LIMITS.maxOutputFrames, 'fixture output limit');
  collection(sources, 1, 1, 'one staged source');
  const source = sources[0];
  record(source, ['sourceId', 'assetPath', 'contentSha256', 'durationMs', 'audioStreams'], 'staged source');
  identifier(source.sourceId, 'sourceId');
  requireThat(typeof source.contentSha256 === 'string' && /^[0-9a-f]{64}$/.test(source.contentSha256), 'source hash');
  requireThat(source.assetPath === `inputs/${source.contentSha256}.mp4`, 'hash-named local source only');
  integer(source.durationMs, 'source duration', 1);
  requireThat(source.durationMs <= FIXTURE_LIMITS.maxSourceDurationMs, 'fixture source duration limit');
  integer(source.audioStreams, 'audioStreams');
  requireThat(recipe.audio === 'mute' || source.audioStreams === 1, 'preserve requires one inspected mixed audio stream');

  collection(recipe.segments, 1, 8, 'segments');
  let end = 0;
  for (const segment of recipe.segments) {
    record(segment, ['segmentId', 'sourceId', 'fromFrame', 'toFrame', 'outputFromFrame', 'durationInFrames'], 'segment');
    identifier(segment.segmentId, 'segmentId');
    requireThat(segment.sourceId === source.sourceId, 'segment source mapping');
    for (const field of ['fromFrame', 'toFrame', 'outputFromFrame', 'durationInFrames']) integer(segment[field], field);
    requireThat(segment.fromFrame % 3 === 0 && segment.toFrame % 3 === 0, 'integer-ms 24fps grid');
    requireThat(segment.toFrame > segment.fromFrame && segment.durationInFrames === segment.toFrame - segment.fromFrame, 'half-open segment');
    requireThat(segment.toFrame <= source.durationMs * FPS / 1000, 'segment exceeds inspected duration');
    requireThat(segment.outputFromFrame === end, 'contiguous output timeline');
    end += segment.durationInFrames;
  }
  unique(recipe.segments.map((segment) => segment.segmentId), 'segmentId');
  requireThat(end === recipe.durationInFrames, 'total duration');
  collection(recipe.captions, 0, 16, 'captions');
  for (const caption of recipe.captions) {
    record(caption, ['captionId', 'fromFrame', 'toFrame', 'text'], 'caption');
    identifier(caption.captionId, 'captionId');
    integer(caption.fromFrame, 'caption fromFrame');
    integer(caption.toFrame, 'caption toFrame');
    requireThat(caption.toFrame > caption.fromFrame && caption.toFrame <= end, 'half-open output caption');
    literalText(caption.text);
  }
  unique(recipe.captions.map((caption) => caption.captionId), 'captionId');
  const orderedCaptions = [...recipe.captions].sort((a, b) => a.fromFrame - b.fromFrame);
  for (let i = 1; i < orderedCaptions.length; i++) requireThat(orderedCaptions[i].fromFrame >= orderedCaptions[i - 1].toFrame, 'overlapping captions');
  requireThat(new TextEncoder().encode(JSON.stringify(input)).length <= FIXTURE_LIMITS.maxInputBytes, 'input byte limit');
  return structuredClone(input); // Never modify the accepted recipe or caller's mapping.
}

// All trim values are on the composition's 24fps clock. A 30fps encoded source
// still starts at 2 seconds for trimBefore=48, NOT at encoded frame 48/30.
export function segmentVideoProps(segment, audio, trustedLocalUrl) {
  return {
    src: trustedLocalUrl,
    trimBefore: segment.fromFrame,
    durationInFrames: segment.durationInFrames,
    playbackRate: 1,
    muted: audio === 'mute',
    audioStreamIndex: 0,
    volume: 1,
    style: {width: '100%', height: '100%', objectFit: 'contain'},
  };
}
