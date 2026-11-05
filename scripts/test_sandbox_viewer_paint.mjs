import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const viewer = readFileSync(new URL('../backend/app/sandbox/view_assets/viewer.js', import.meta.url), 'utf8')
    .replace(/^import RFB[^\n]*\n/, '');
const probe = readFileSync(new URL('./retained_view_acceptance.py', import.meta.url), 'utf8');
const canvasStats = probe.match(/CANVASES = """([\s\S]*?)"""/)[1];

function fixture(width = 2, height = 1) {
    const pixels = new Uint8ClampedArray(width * height * 4);
    const canvas = { width, height,
        getContext: () => ({ getImageData: () => ({ data: pixels }) }),
        getBoundingClientRect: () => ({ width, height }),
    };
    const frames = [];
    const reports = [];
    let rfb;
    class RFB {
        constructor() { rfb = this; this._display = { flush: () => Promise.resolve() }; }
        _framebufferUpdate() { return true; }
        addEventListener() {}
    }
    const document = {
        getElementById: id => id === 'view-config'
            ? { textContent: JSON.stringify({ socket: '/stream', ticket: 'fixture' }) } : {},
        querySelector: () => canvas,
        querySelectorAll: () => [canvas],
    };
    vm.runInNewContext(viewer, { RFB, document,
        location: { protocol: 'http:', host: 'fixture' },
        window: { webkit: { messageHandlers: { sandboxView: { postMessage: event => reports.push(event.state) } } }, addEventListener() {} },
        requestAnimationFrame: callback => frames.push(callback),
    });
    return { pixels, reports, rfb,
        stats: () => vm.runInNewContext('(' + canvasStats + ')()', { document })[0],
        async paint() {
            await Promise.resolve();
            while (frames.length) frames.shift()();
        },
    };
}

test('metadata-only updates cannot report a frame; painted black can', async () => {
    const f = fixture();
    f.rfb._framebufferUpdate();
    await f.paint();
    assert.deepEqual(f.reports, []);
    assert.equal(f.stats().paintedPixels, 0);
    f.pixels[3] = 255;
    f.rfb._framebufferUpdate();
    await f.paint();
    assert.deepEqual(f.reports, ['frame']);
    assert.equal(f.stats().paintedPixels, 1);
    assert.equal(f.stats().variance, 0);
    assert.equal(f.stats().meanBrightness, 0);
});

test('overlapping flush callbacks report the first painted frame only once', async () => {
    const f = fixture();
    f.pixels[3] = 255;
    f.rfb._framebufferUpdate();
    f.rfb._framebufferUpdate();
    await f.paint();
    f.rfb._framebufferUpdate();
    await f.paint();
    assert.deepEqual(f.reports, ['frame']);
});

test('zero-sized canvas cannot report a frame', async () => {
    const f = fixture(0, 0);
    f.rfb._framebufferUpdate();
    await f.paint();
    assert.deepEqual(f.reports, []);
});

test('probe counts painted pixels even outside its color-statistics sample', () => {
    const f = fixture(1024, 768);
    f.pixels[7] = 255;
    assert.equal(f.stats().paintedPixels, 1);
    assert.equal(f.stats().nontransparent, 0);
});
