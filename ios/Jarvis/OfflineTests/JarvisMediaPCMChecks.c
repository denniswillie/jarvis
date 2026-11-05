#include "../Jarvis/Services/JarvisMediaPCM.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static void render(JRDMediaPCM *media, uint32_t count, bool voice) {
    int16_t out[1920 * 2] = {0};
    while (count) {
        uint32_t frames = count > 1920 ? 1920 : count;
        JRDMediaRender(media, out, frames, 100, voice);
        count -= frames;
    }
}
static int16_t value(JRDMediaPCM *media, uint64_t generation) {
    int16_t sample[2] = {10000, 10000}, out[2] = {0};
    assert(JRDMediaEnqueue(media, sample, 1, generation, 1000));
    JRDMediaRender(media, out, 1, 100, false);
    return out[0];
}
int main(void) {
    JRDMediaPCM *media = calloc(1, sizeof(*media));
    assert(media);
    // A queue receipt, silence, muted output, expired data and an old source
    // must never be reported as first playback for the selected source.
    JRDMediaInitialize(media, 48000, 1);
    JRDMediaSelect(media, 100);
    atomic_store(&media->allowed, true); media->gain = 1;
    int16_t one = 10000, silence = 0, output = 0;
    assert(!JRDMediaHasRendered(media, 100));
    assert(JRDMediaEnqueue(media, &one, 1, 100, 1000));
    assert(!JRDMediaHasRendered(media, 100));
    assert(JRDMediaRender(media, &output, 1, 100, false));
    assert(JRDMediaHasRendered(media, 100));
    JRDMediaSelect(media, 101);
    assert(!JRDMediaHasRendered(media, 100) && !JRDMediaHasRendered(media, 101));
    assert(JRDMediaEnqueue(media, &silence, 1, 101, 1000));
    assert(!JRDMediaRender(media, &output, 1, 100, false));
    assert(!JRDMediaHasRendered(media, 101));
    atomic_store(&media->muted, true);
    assert(JRDMediaEnqueue(media, &one, 1, 101, 1000));
    assert(!JRDMediaRender(media, &output, 1, 100, false));
    assert(!JRDMediaHasRendered(media, 101));
    atomic_store(&media->muted, false);
    assert(JRDMediaEnqueue(media, &one, 1, 101, 50));
    assert(!JRDMediaRender(media, &output, 1, 100, false));
    assert(!JRDMediaHasRendered(media, 101));
    puts("PASS: first-playback receipt requires actual nonzero rendering of the current source");
    JRDMediaInitialize(media, 8000, 1);
    int16_t out[1920 * 2] = {123}, source[1920 * 2];
    for (unsigned i = 0; i < 1920 * 2; ++i) source[i] = 10000;
    assert(!JRDMediaRender(media, out, 1, 100, false) && out[0] == 123);
    assert(!JRDMediaEnqueue(media, source, 1, 1, 1000));
    JRDMediaSelect(media, 1); atomic_store(&media->allowed, true);
    render(media, 3000, false);
    assert(abs(value(media, 1) - 10000) <= 1);
    atomic_store(&media->userSpeaking, true); render(media, 800, false);
    assert(abs(value(media, 1) - 2000) <= 1);
    atomic_store(&media->userSpeaking, false); render(media, 800, true);
    assert(abs(value(media, 1) - 2000) <= 1); // Other speaker still active.
    atomic_store(&media->baseGainPermille, 500); render(media, 800, true);
    assert(abs(value(media, 1) - 1000) <= 1);
    render(media, 5000, false);
    assert(abs(value(media, 1) - 5000) <= 1); // Restore new, not original volume.
    atomic_store(&media->muted, true);
    assert(value(media, 1) == 0 && atomic_load(&media->queuedFrames) == 0);
    atomic_store(&media->muted, false); render(media, 3000, false);
    atomic_store(&media->allowed, false); assert(value(media, 1) == 0);
    atomic_store(&media->allowed, true); render(media, 3000, false);
    assert(JRDMediaEnqueue(media, source, 1, 1, 1000));
    JRDMediaSelect(media, 2);
    assert(!JRDMediaEnqueue(media, source, 1, 1, 1000));
    out[0] = 0; assert(!JRDMediaRender(media, out, 1, 100, false) && out[0] == 0);
    assert(JRDMediaEnqueue(media, source, 1, 2, 50));
    assert(!JRDMediaRender(media, out, 1, 100, false));
    assert(JRDMediaEnqueue(media, source, 1, 2, 1000));
    assert(!JRDMediaRender(media, out, 1, 0, false)); // Unknown clock is not fresh.
    assert(!JRDMediaEnqueue(media, source, 1921, 2, 1000));
    assert(JRDMediaEnqueue(media, source, 1920, 2, 1000));
    assert(JRDMediaEnqueue(media, source, 1920, 2, 1000));
    assert(!JRDMediaEnqueue(media, source, 1920, 2, 1000)); // <=500 ms.
    JRDMediaSelect(media, 0); render(media, 1, false);
    assert(atomic_load(&media->queuedFrames) == 0);
    for (uint32_t channels = 1; channels <= 2; channels++) {
        JRDMediaInitialize(media, 48000, channels);
        JRDMediaSelect(media, 1); atomic_store(&media->allowed, true);
        render(media, 16000, false);
        int16_t loud[2] = {32000, -32000}, mixed[2] = {30000, -30000};
        assert(JRDMediaEnqueue(media, loud, 1, 1, 1000));
        assert(JRDMediaRender(media, mixed, 1, 100, false));
        assert(mixed[0] == INT16_MAX);
        if (channels == 2) assert(mixed[1] == INT16_MIN);
    }
    JRDMediaConsumption *receipt = calloc(1, sizeof(*receipt));
    assert(receipt);
    JRDMediaInitialize(media, 16000, 1);
    JRDMediaSelect(media, 4); atomic_store(&media->allowed, true); media->gain = 1;
    assert(JRDMediaEnqueueMapped(media, source, 160, 4, 1000, 0));
    assert(JRDMediaEnqueueMapped(media, source, 160, 4, 1000, 480));
    memset(out, 0, sizeof(out));
    JRDMediaRenderConsumed(media, out, 80, 100, false, receipt);
    assert(receipt->count == 1 && receipt->outputFrames == 80);
    assert(receipt->spans[0].sourceFrame == 0 && receipt->spans[0].frames == 80);
    assert(receipt->gains[0] == 1 && receipt->gains[79] == 1);
    memset(receipt, 0, sizeof(*receipt));
    atomic_store(&media->muted, true);
    JRDMediaRenderConsumed(media, out, 240, 100, false, receipt);
    assert(receipt->count == 1 && receipt->spans[0].sourceFrame == 240);
    assert(receipt->spans[0].frames == 240 && receipt->gains[239] == 0);
    memset(receipt, 0, sizeof(*receipt));
    assert(JRDMediaEnqueueMapped(media, source, 1, 4, 50, 960));
    JRDMediaRenderConsumed(media, out, 1, 100, false, receipt);
    assert(receipt->outputFrames == 0); // Expiry is not consumption.
    assert(JRDMediaEnqueue(media, source, 1, 4, 1000));
    JRDMediaRenderConsumed(media, out, 1, 100, false, receipt);
    assert(receipt->unknownFrames == 1 && receipt->count == 0);
    memset(receipt, 0, sizeof(*receipt)); receipt->count = JRDMediaReceiptSpans;
    assert(JRDMediaEnqueueMapped(media, source, 1, 4, 1000, 1000));
    JRDMediaRenderConsumed(media, out, 1, 100, false, receipt);
    assert(receipt->droppedFrames == 1 && receipt->count == JRDMediaReceiptSpans);
    free(receipt);
    free(media);
    puts("PASS: partial consumption, cross-packet cumulative source phase, actual gain, expiry, unknown mapping and bounded sidecar overflow");
    puts("PASS: media queue, gain ramps, both speakers, mute, source replacement, expiry, capacity and clipping");
    return 0;
}
