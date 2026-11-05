import {parseArgs} from 'node:util';
import {renderFixture} from './harness.mjs';

const names = ['input', 'staging-dir', 'browser', 'binaries-dir', 'work-dir', 'output'];
try {
  const {values} = parseArgs({options: Object.fromEntries(names.map((name) => [name, {type: 'string'}])), strict: true, allowPositionals: false});
  const result = await renderFixture({
    inputFile: values.input, stagingDir: values['staging-dir'],
    browserExecutable: values.browser, binariesDirectory: values['binaries-dir'],
    workDir: values['work-dir'], outputPath: values.output,
  });
  console.log(JSON.stringify({localOutputPath: result.outputPath, qualification: 'requires_decode_and_playback_inspection'}));
} catch (error) {
  console.error(`Clip fixture failed: ${error.message}`);
  process.exitCode = 1;
}
