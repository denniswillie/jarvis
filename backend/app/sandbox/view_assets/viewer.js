import RFB from './novnc/core/rfb.js';

const config = JSON.parse(document.getElementById('view-config').textContent);
const report = state => window.webkit?.messageHandlers?.sandboxView?.postMessage({state});
const url = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}${config.socket}`;
const hasPaintedPixels = () => {
    const canvas = document.querySelector('#screen canvas');
    if (!canvas?.width || !canvas?.height) return false;
    const context = canvas.getContext('2d');
    if (!context) return false;
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let alpha = 3; alpha < pixels.length; alpha += 4) {
        if (pixels[alpha] !== 0) return true;
    }
    return false;
};
let rfb;
try {
    rfb = new RFB(document.getElementById('screen'), url, {
        wsProtocols: ['jarvis-view-v1', config.ticket],
    });
    rfb.viewOnly = true;
    rfb.scaleViewport = true;
    rfb.resizeSession = false;
    rfb.showDotCursor = false;
    rfb.qualityLevel = 6;
    rfb.compressionLevel = 6;
    // Pinned noVNC adapter: acknowledge only a completed framebuffer update,
    // flushed to canvas and through a browser paint, never the RFB handshake.
    const update = rfb._framebufferUpdate.bind(rfb);
    let shown = false;
    rfb._framebufferUpdate = function () {
        const complete = update();
        if (complete && !shown) {
            this._display.flush().then(() => requestAnimationFrame(() =>
                requestAnimationFrame(() => {
                    if (shown || !hasPaintedPixels()) return;
                    shown = true;
                    report('frame');
                })));
        }
        return complete;
    };
    rfb.addEventListener('connect', () => report('connected'));
    rfb.addEventListener('disconnect', () => report('disconnected'));
    rfb.addEventListener('securityfailure', () => report('failed'));
    rfb.addEventListener('credentialsrequired', () => { rfb.disconnect(); report('failed'); });
} catch {
    report('failed');
}
window.addEventListener('pagehide', () => rfb?.disconnect(), {once: true});
