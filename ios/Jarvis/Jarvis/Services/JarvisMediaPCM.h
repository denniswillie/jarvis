// Bounded, single-producer/single-consumer media input for the existing audio
// owner. Decode/resample before enqueueing. No allocation, locks, networking or
// source-state writes in the render callback. This is playback, not the master
// recording source: retain pre-hardware-conversion media for recording.
#ifndef JARVIS_MEDIA_PCM_H
#define JARVIS_MEDIA_PCM_H
#include <stdatomic.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

enum { JRDMediaSlots = 32, JRDMediaPacketFrames = 1920,
       JRDMediaReceiptSpans = 32, JRDMediaReceiptFrames = 8192 };
typedef struct {
    uint64_t generation;
    double sourceFrame; // Absolute 48k source position, including fractional phase.
    uint32_t outputOffset, frames;
} JRDMediaConsumedSpan;
typedef struct {
    uint32_t count, outputFrames, unknownFrames, droppedFrames;
    JRDMediaConsumedSpan spans[JRDMediaReceiptSpans];
    float gains[JRDMediaReceiptFrames]; // Actual gain for each output frame.
} JRDMediaConsumption;
typedef struct {
    uint64_t generation, expiresAt;
    double sourceFrame;
    uint32_t frames;
    int16_t samples[JRDMediaPacketFrames * 2];
} JRDMediaPacket;
typedef struct {
    _Atomic(uint64_t) written, read, generation, renderedGeneration;
    _Atomic(uint32_t) queuedFrames, baseGainPermille;
    _Atomic(bool) allowed, muted, userSpeaking;
    double rate;
    uint32_t channels, maximumQueuedFrames;
    // Consumer-owned interpolation and packet cursor.
    uint32_t offset, releaseHold;
    float gain;
    JRDMediaPacket packets[JRDMediaSlots];
} JRDMediaPCM;

static inline void JRDMediaInitialize(JRDMediaPCM *media, double rate, uint32_t channels) {
    memset(media, 0, sizeof(*media));
    atomic_init(&media->written, 0); atomic_init(&media->read, 0);
    atomic_init(&media->generation, 0); atomic_init(&media->queuedFrames, 0);
    atomic_init(&media->renderedGeneration, 0);
    atomic_init(&media->baseGainPermille, 1000);
    atomic_init(&media->allowed, false); atomic_init(&media->muted, false);
    atomic_init(&media->userSpeaking, false);
    media->rate = rate; media->channels = channels;
    media->maximumQueuedFrames = (uint32_t)(rate * .5);
}

// Serialized producer/control queue only. Zero detaches; nonzero generations
// must not be reused. In-flight output is not synchronously fenced by this.
static inline void JRDMediaSelect(JRDMediaPCM *media, uint64_t generation) {
    atomic_store_explicit(&media->generation, generation, memory_order_release);
}
static inline bool JRDMediaHasRendered(JRDMediaPCM *media, uint64_t generation) {
    return generation && generation == atomic_load_explicit(&media->generation, memory_order_acquire)
        && generation == atomic_load_explicit(&media->renderedGeneration, memory_order_acquire);
}
static inline bool JRDMediaEnqueueMapped(JRDMediaPCM *media, const int16_t *samples,
                                  uint32_t frames, uint64_t generation, uint64_t expiresAt,
                                  double sourceFrame) {
    if (!samples || !frames || frames > JRDMediaPacketFrames || !expiresAt ||
        !generation || generation != atomic_load_explicit(&media->generation, memory_order_acquire)) return false;
    uint64_t w = atomic_load_explicit(&media->written, memory_order_relaxed);
    uint64_t r = atomic_load_explicit(&media->read, memory_order_acquire);
    uint32_t queued = atomic_load_explicit(&media->queuedFrames, memory_order_acquire);
    if (w - r >= JRDMediaSlots || queued + frames > media->maximumQueuedFrames) return false;
    JRDMediaPacket *packet = &media->packets[w % JRDMediaSlots];
    packet->generation = generation; packet->expiresAt = expiresAt; packet->frames = frames;
    packet->sourceFrame = sourceFrame;
    memcpy(packet->samples, samples, frames * media->channels * sizeof(int16_t));
    atomic_fetch_add_explicit(&media->queuedFrames, frames, memory_order_relaxed);
    atomic_store_explicit(&media->written, w + 1, memory_order_release);
    return true;
}
static inline bool JRDMediaEnqueue(JRDMediaPCM *media, const int16_t *samples,
                                  uint32_t frames, uint64_t generation, uint64_t expiresAt) {
    return JRDMediaEnqueueMapped(media, samples, frames, generation, expiresAt, NAN);
}
static inline void JRDMediaAdvance(JRDMediaPCM *media, uint64_t r, const JRDMediaPacket *packet) {
    media->offset = 0;
    atomic_fetch_sub_explicit(&media->queuedFrames, packet->frames, memory_order_relaxed);
    atomic_store_explicit(&media->read, r + 1, memory_order_release);
}

// Adds media to a voice-only output buffer. Caller captures the voice contribution
// BEFORE this call. Media never enters the microphone callback. hostTime and
// packet expiry use the same monotonic clock; a missing clock drops media.
static inline bool JRDMediaRenderConsumed(JRDMediaPCM *media, int16_t *output, uint32_t frames,
                                  uint64_t hostTime, bool voiceAudible, JRDMediaConsumption *receipt) {
    uint64_t generation = atomic_load_explicit(&media->generation, memory_order_acquire);
    bool allowed = generation && atomic_load_explicit(&media->allowed, memory_order_relaxed)
        && !atomic_load_explicit(&media->muted, memory_order_relaxed);
    bool speaking = voiceAudible || atomic_load_explicit(&media->userSpeaking, memory_order_relaxed);
    float base = atomic_load_explicit(&media->baseGainPermille, memory_order_relaxed) / 1000.0f;
    if (speaking) media->releaseHold = (uint32_t)(media->rate * .25);
    bool audibleMedia = false;
    for (uint32_t f = 0; f < frames; ++f) {
        bool duck = speaking || media->releaseHold > 0;
        if (!speaking && media->releaseHold > 0) media->releaseHold--;
        float target = allowed ? base * (duck ? .2f : 1.f) : 0;
        if (!allowed) media->gain = 0; // Privacy/user mute is immediate, not a fade.
        else {
            float step = (float)(1.0 / (media->rate * (target < media->gain ? .08 : .3)));
            media->gain += fmaxf(-step, fminf(step, target - media->gain));
        }
        JRDMediaPacket *packet = NULL;
        uint64_t r = atomic_load_explicit(&media->read, memory_order_relaxed);
        uint64_t w = atomic_load_explicit(&media->written, memory_order_acquire);
        // At most the fixed queue size is discarded, even if a producer refills.
        for (uint32_t skipped = 0; r < w && skipped < JRDMediaSlots; ++skipped) {
            packet = &media->packets[r % JRDMediaSlots];
            if (generation && packet->generation == generation && hostTime && hostTime < packet->expiresAt) break;
            JRDMediaAdvance(media, r, packet); r++; packet = NULL;
        }
        if (!packet) continue;
        if (receipt) {
            receipt->outputFrames++;
            double source = packet->sourceFrame + media->offset * (48000.0 / media->rate);
            if (!isfinite(source)) receipt->unknownFrames++;
            else {
                JRDMediaConsumedSpan *span = receipt->count ? &receipt->spans[receipt->count - 1] : NULL;
                bool contiguous = span && span->generation == packet->generation &&
                    span->outputOffset + span->frames == f &&
                    fabs(span->sourceFrame + span->frames * (48000.0 / media->rate) - source) < .00001;
                if (f >= JRDMediaReceiptFrames || (!contiguous && receipt->count == JRDMediaReceiptSpans)) {
                    receipt->droppedFrames++;
                } else {
                    if (!contiguous) {
                        span = &receipt->spans[receipt->count++];
                        *span = (JRDMediaConsumedSpan){packet->generation, source, f, 0};
                    }
                    receipt->gains[f] = media->gain;
                    span->frames++;
                }
            }
        }
        for (uint32_t c = 0; c < media->channels; ++c) {
            int32_t addition = (int32_t)lrintf(packet->samples[media->offset * media->channels + c] * media->gain);
            int32_t mixed = output[f * media->channels + c] + addition;
            output[f * media->channels + c] = (int16_t)(mixed > INT16_MAX ? INT16_MAX : mixed < INT16_MIN ? INT16_MIN : mixed);
            audibleMedia |= addition != 0;
        }
        if (++media->offset == packet->frames) JRDMediaAdvance(media, r, packet);
    }
    if (audibleMedia && generation == atomic_load_explicit(&media->generation, memory_order_acquire))
        atomic_store_explicit(&media->renderedGeneration, generation, memory_order_release);
    return audibleMedia;
}
static inline bool JRDMediaRender(JRDMediaPCM *media, int16_t *output, uint32_t frames,
                                  uint64_t hostTime, bool voiceAudible) {
    return JRDMediaRenderConsumed(media, output, frames, hostTime, voiceAudible, NULL);
}
#endif
