/** Isolated fixture harness, not a task worker, source resolver or publication API. */
import {constants, createReadStream} from 'node:fs';
import {access, copyFile, lstat, mkdir, readFile, realpath, stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {COMPOSITION_ID, FIXTURE_LIMITS, validateClipProps} from '../src/props.mjs';

const templateRoot = fileURLToPath(new URL('..', import.meta.url));

function absolute(value, name) {
  if (typeof value !== 'string' || !path.isAbsolute(value) || path.normalize(value) !== value) {
    throw new Error(`${name} must be an explicit normalized absolute path`);
  }
  return value;
}

async function absent(target) {
  try { await lstat(target); }
  catch (error) { if (error.code === 'ENOENT') return; throw error; }
  throw new Error('Use a new work directory and output for every attempt; never overwrite');
}

async function executable(target) {
  if (!(await stat(target)).isFile()) throw new Error('Provisioned executable must be a file');
  await access(target, constants.X_OK);
}

async function digestFile(target) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(target)) hash.update(chunk);
  return hash.digest('hex');
}

/** Read-only preflight. No browser, bundle, process, provider or download. */
export async function preflight(options) {
  const expectedKeys = ['inputFile', 'stagingDir', 'browserExecutable', 'binariesDirectory', 'workDir', 'outputPath'];
  if (Object.keys(options).length !== expectedKeys.length || expectedKeys.some((key) => !Object.hasOwn(options, key))) {
    throw new Error('Explicit input, staging, browser, binaries, work and output paths required');
  }
  for (const key of expectedKeys) absolute(options[key], key);
  if (options.outputPath !== path.join(options.workDir, 'output.mp4')) {
    throw new Error('Output must be output.mp4 inside the new isolated work directory');
  }
  await absent(options.workDir);
  if (!(await stat(path.dirname(options.workDir))).isDirectory()) throw new Error('Work directory parent must exist');
  await executable(options.browserExecutable);
  if (!(await stat(options.binariesDirectory)).isDirectory()) throw new Error('Provisioned binaries directory required');
  // This is Remotion's matching platform compositor bundle, not an arbitrary
  // system FFmpeg installation. Matching versions/libraries remain Core's duty.
  const suffix = process.platform === 'win32' ? '.exe' : '';
  for (const name of ['ffmpeg', 'ffprobe', 'remotion']) await executable(path.join(options.binariesDirectory, name + suffix));
  const inputStat = await stat(options.inputFile);
  if (!inputStat.isFile() || inputStat.size > FIXTURE_LIMITS.maxInputBytes) throw new Error('Input must be a JSON file of at most 64 KiB');
  const raw = await readFile(options.inputFile);
  if (raw.length > FIXTURE_LIMITS.maxInputBytes) throw new Error('Input grew past 64 KiB');
  const inputProps = validateClipProps(JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(raw)));
  const source = inputProps.sources[0];
  const stagingRoot = await realpath(options.stagingDir);
  const sourcePath = path.join(stagingRoot, source.assetPath);
  // Reject symlinks including the inputs directory, not only path traversal text.
  if (await realpath(sourcePath) !== sourcePath || !(await lstat(sourcePath)).isFile()) throw new Error('Staged source must be a regular local file without symlink redirection');
  const sourceStat = await stat(sourcePath);
  if (sourceStat.size < 1 || sourceStat.size > FIXTURE_LIMITS.maxSourceBytes) throw new Error('Source exceeds fixture byte allowance');
  if (await digestFile(sourcePath) !== source.contentSha256) throw new Error('Staged source hash mismatch');
  return {...options, inputProps, sourcePath};
}

export function refuseBrowserDownload() {
  throw new Error('Browser downloads are disabled; provision an explicit compatible executable');
}

export function renderOptions(prepared, serveUrl, composition) {
  return {
    serveUrl, composition, inputProps: prepared.inputProps,
    browserExecutable: prepared.browserExecutable,
    binariesDirectory: prepared.binariesDirectory,
    onBrowserDownload: refuseBrowserDownload,
    chromeMode: 'chrome-for-testing', // Full provisioned Chrome uses --headless=new.
    outputLocation: prepared.outputPath,
    codec: 'h264', audioCodec: 'aac', pixelFormat: 'yuv420p',
    colorSpace: 'bt709', imageFormat: 'png',
    crf: 18, audioBitrate: '192k', sampleRate: 48000,
    muted: prepared.inputProps.recipe.audio === 'mute',
    overwrite: false, concurrency: 1, offthreadVideoThreads: 1,
    chromiumOptions: {gl: 'swangle'},
    hardwareAcceleration: 'disable',
    timeoutInMilliseconds: 30_000, logLevel: 'warn',
  };
}

/**
 * Call only in a reserved render lane. Tests inject inert APIs; this module has
 * no top-level renderer import or default browser lookup. Return is a local
 * encoder attempt only, never a playable/retained/published success receipt.
 */
export async function renderFixture(options, loadApis = async () => {
  const [{bundle}, {selectComposition, renderMedia}] = await Promise.all([
    import('@remotion/bundler'), import('@remotion/renderer'),
  ]);
  return {bundle, selectComposition, renderMedia};
}) {
  const prepared = await preflight(options); // Fail malformed inputs before effects.
  await mkdir(prepared.workDir); // Exclusive creation closes the overwrite race.
  const publicDir = path.join(prepared.workDir, 'public');
  const stagedCopy = path.join(publicDir, prepared.inputProps.sources[0].assetPath);
  await mkdir(path.dirname(stagedCopy), {recursive: true});
  await copyFile(prepared.sourcePath, stagedCopy, constants.COPYFILE_EXCL);
  // Bind the bytes we will actually serve, including a source replacement race.
  const copiedStat = await stat(stagedCopy);
  if (copiedStat.size > FIXTURE_LIMITS.maxSourceBytes || await digestFile(stagedCopy) !== prepared.inputProps.sources[0].contentSha256) {
    throw new Error('Staged bytes changed while copying; no render started');
  }
  const {bundle, selectComposition, renderMedia} = await loadApis();
  const serveUrl = await bundle({
    entryPoint: path.join(templateRoot, 'src/index.mjs'), rootDir: templateRoot,
    outDir: path.join(prepared.workDir, 'bundle'), publicDir,
    enableCaching: false, symlinkPublicDir: false, gitSource: null,
    onSymlinkDetected: () => { throw new Error('Symlinks forbidden in bundle public assets'); },
  });
  if (serveUrl !== path.join(prepared.workDir, 'bundle')) throw new Error('Bundle must remain in the explicit local work directory');
  const composition = await selectComposition({
    serveUrl, id: COMPOSITION_ID, inputProps: prepared.inputProps,
    browserExecutable: prepared.browserExecutable, binariesDirectory: prepared.binariesDirectory,
    onBrowserDownload: refuseBrowserDownload, chromiumOptions: {gl: 'swangle'},
    chromeMode: 'chrome-for-testing',
    logLevel: 'warn',
  });
  const recipe = prepared.inputProps.recipe;
  for (const field of ['width', 'height', 'fps', 'durationInFrames']) {
    if (composition[field] !== recipe[field]) throw new Error('Composition metadata differs from validated recipe');
  }
  await renderMedia(renderOptions(prepared, serveUrl, composition));
  return {outputPath: prepared.outputPath};
}
