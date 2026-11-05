#import "JarvisDuplexAudioIO.h"
#include "JarvisMediaPCM.h"
#import <TargetConditionals.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mach/mach_time.h>

// Two independent SPSC queues. Fixed PCM and consumption-sidecar storage. Each track
// has one AudioUnit producer and one capture-queue consumer. No heap work,
// dispatch, Objective-C message sends, or application locks in the callbacks;
// the only external calls are AudioUnitRender and cached WebRTC PCM blocks.
enum { JRDSlots = 32, JRDMaxFrames = 8192, JRDMaxChannels = 2,
       JRDExtCapacity = 65536, JRDExtMask = 65535 };
typedef struct {
    uint64_t token;
    AudioTimeStamp time;
    uint32_t frames;
    bool gate;
    int16_t samples[JRDMaxFrames * JRDMaxChannels];
    JRDMediaConsumption media;
    int16_t *diagnosticPostMix;
    uint64_t diagnosticCopyHost, diagnosticQueued, diagnosticReadFrame;
    uint64_t diagnosticFirstReceipt, diagnosticLastReceipt;
    bool diagnosticExternal;
} JRDSlot;
typedef struct {
    _Atomic(uint64_t) written, read, lostPackets, lostFrames;
    JRDSlot slots[JRDSlots];
} JRDRing;

static bool JRDWaitForZero(_Atomic(uint32_t) *value) {
    // Control/capture queue only. Never called by an audio callback.
    struct timespec delay = { .tv_sec = 0, .tv_nsec = 1000000 };
    for (unsigned i = 0; i < 100; ++i) {
        if (atomic_load_explicit(value, memory_order_acquire) == 0) return true;
        nanosleep(&delay, NULL);
    }
    return atomic_load_explicit(value, memory_order_acquire) == 0;
}

@interface JRDConsumedMediaSpan ()
@property(nonatomic, readwrite) uint64_t sourceGeneration;
@property(nonatomic, readwrite) double sourceFrame;
@property(nonatomic, readwrite) uint32_t outputOffset;
@property(nonatomic, readwrite) NSData *gains;
@end
@implementation JRDConsumedMediaSpan
@end

@interface JRDDuplexPacket ()
@property(nonatomic, readwrite, nullable) NSData *diagnosticPlaybackPCM16;
@property(nonatomic, readwrite, nullable) NSDictionary<NSString *, NSNumber *> *diagnosticTiming;
@property(nonatomic, readwrite) NSData *pcm16;
@property(nonatomic, readwrite) NSInteger track;
@property(nonatomic, readwrite) uint32_t frameCount;
@property(nonatomic, readwrite) uint64_t captureToken;
@property(nonatomic, readwrite) uint64_t hostTime;
@property(nonatomic, readwrite) double sampleTime;
@property(nonatomic, readwrite) uint32_t timestampFlags;
@property(nonatomic, readwrite) BOOL softwareGateOpen;
@property(nonatomic, readwrite) NSArray<JRDConsumedMediaSpan *> *mediaSpans;
@property(nonatomic, readwrite) uint32_t mediaOutputFrames;
@property(nonatomic, readwrite) uint32_t mediaUnknownFrames;
@property(nonatomic, readwrite) uint32_t mediaDroppedFrames;
@end
@implementation JRDDuplexPacket
@end

@interface JRDDuplexIO () {
@public
    AudioUnit _unit;
    RTCAudioDeviceDeliverRecordedDataBlock _deliver;
    RTCAudioDeviceGetPlayoutDataBlock _play;
    JRDRing *_rings[2];
    int16_t *_input;
    int16_t *_extSamples;
    int16_t *_diagnosticOutput;
    uint64_t *_diagnosticEnqueueTimes;
    JRDMediaPCM *_media;
    uint64_t _mediaMaxAgeTicks;
    _Atomic(uint64_t) _extWritten, _extRead, _externalInputFrames;
    _Atomic(bool) _extProducerLock, _extResetRequested;
    _Atomic(uint32_t) _playbackEchoHoldoffFrames;
    uint32_t _echoHoldoffPresetFrames;
    double _extResamplePos;
    int16_t _extPrevSample;
    bool _extHasPrevSample;
    uint32_t _inChannels, _outChannels;
    _Atomic(bool) _ioEnabled, _recordingWanted, _playoutWanted;
    _Atomic(bool) _microphoneGate, _outputGate;
    _Atomic(uint64_t) _captureToken;
    _Atomic(uint32_t) _callbacks, _captureReaders;
    _Atomic(uint64_t) _inputErrors, _outputErrors;
    _Atomic(int32_t) _lastInputStatus, _lastOutputStatus;
    _Atomic(uint64_t) _inputCallbackCount, _renderedInputFrames, _inputDeliveryCount;
    _Atomic(int32_t) _lastInputDeliveryStatus;
    _Atomic(uint32_t) _inputPeakBeforeGate;
    _Atomic(uint64_t) _outputCallbackCount;
    _Atomic(uint32_t) _outputPeakAfterGate;
    bool _configured, _started, _holdsSelf, _closed, _closeRequired;
    bool _configurationAttempted;
    bool _rawAudio;
}
@end

static void JRDSilence(AudioBufferList *data) {
    if (!data) return;
    for (UInt32 i = 0; i < data->mNumberBuffers; ++i) {
        if (data->mBuffers[i].mData)
            memset(data->mBuffers[i].mData, 0, data->mBuffers[i].mDataByteSize);
    }
}

static JRDSlot *JRDBeginCapture(__unsafe_unretained JRDDuplexIO *io, unsigned track,
                       const AudioTimeStamp *time, uint32_t frames,
                       const int16_t *samples, bool gate) {
    atomic_fetch_add_explicit(&io->_captureReaders, 1, memory_order_acq_rel);
    uint64_t token = atomic_load_explicit(&io->_captureToken, memory_order_acquire);
    JRDSlot *slot = NULL;
    if (token != 0) {
        JRDRing *ring = io->_rings[track];
        uint64_t w = atomic_load_explicit(&ring->written, memory_order_relaxed);
        uint64_t r = atomic_load_explicit(&ring->read, memory_order_acquire);
        if (!samples || !time || frames > JRDMaxFrames || w - r >= JRDSlots) {
            atomic_fetch_add_explicit(&ring->lostPackets, 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&ring->lostFrames, frames, memory_order_relaxed);
        } else {
            slot = &ring->slots[w % JRDSlots];
            slot->token = token; slot->time = *time; slot->frames = frames; slot->gate = gate;
            slot->diagnosticCopyHost = io->_diagnosticEnqueueTimes ? mach_absolute_time() : 0;
            slot->diagnosticQueued = slot->diagnosticReadFrame = 0;
            slot->diagnosticFirstReceipt = slot->diagnosticLastReceipt = 0;
            slot->diagnosticExternal = false;
            slot->media.count = slot->media.outputFrames = 0;
            slot->media.unknownFrames = slot->media.droppedFrames = 0;
            uint32_t channels = track == 0 ? io->_inChannels : io->_outChannels;
            memcpy(slot->samples, samples, (size_t)frames * channels * sizeof(int16_t));
        }
    }
    return slot;
}
static void JRDEndCapture(__unsafe_unretained JRDDuplexIO *io, unsigned track, JRDSlot *slot) {
    if (slot) atomic_fetch_add_explicit(&io->_rings[track]->written, 1, memory_order_release);
    atomic_fetch_sub_explicit(&io->_captureReaders, 1, memory_order_release);
}
static void JRDCapture(__unsafe_unretained JRDDuplexIO *io, unsigned track,
                       const AudioTimeStamp *time, uint32_t frames,
                       const int16_t *samples, bool gate) {
    JRDSlot *slot = JRDBeginCapture(io, track, time, frames, samples, gate);
    JRDEndCapture(io, track, slot);
}

static OSStatus JRDInput(void *refcon, AudioUnitRenderActionFlags *flags,
                         const AudioTimeStamp *time, UInt32 bus,
                         UInt32 frames, AudioBufferList *unused) {
    __unsafe_unretained JRDDuplexIO *io = (__bridge JRDDuplexIO *)refcon;
    atomic_fetch_add_explicit(&io->_callbacks, 1, memory_order_acq_rel);
    atomic_fetch_add_explicit(&io->_inputCallbackCount, 1, memory_order_relaxed);
    OSStatus status = noErr;
    if (!atomic_load_explicit(&io->_ioEnabled, memory_order_acquire) || frames == 0) goto done;
    if (!time || frames > JRDMaxFrames) { status = kAudioUnitErr_TooManyFramesToProcess; goto failed; }
    AudioBufferList list = { .mNumberBuffers = 1, .mBuffers = {{
        .mNumberChannels = io->_inChannels,
        .mDataByteSize = frames * io->_inChannels * (UInt32)sizeof(int16_t),
        .mData = io->_input
    }} };
    status = AudioUnitRender(io->_unit, flags, time, 1, frames, &list);
    if (status != noErr) goto failed;
    atomic_fetch_add_explicit(&io->_renderedInputFrames, frames, memory_order_relaxed);
    uint32_t echoHoldoff = atomic_load_explicit(&io->_playbackEchoHoldoffFrames, memory_order_relaxed);
    if (echoHoldoff > 0) {
        uint32_t nextHoldoff = echoHoldoff > frames ? (echoHoldoff - frames) : 0;
        atomic_store_explicit(&io->_playbackEchoHoldoffFrames, nextHoldoff, memory_order_relaxed);
    }
        bool usedExternalMic = false;
        uint64_t diagnosticFirstReceipt = 0, diagnosticLastReceipt = 0;
    uint64_t extW = atomic_load_explicit(&io->_extWritten, memory_order_acquire);
    uint64_t extR = atomic_load_explicit(&io->_extRead, memory_order_relaxed);
    if (atomic_exchange_explicit(&io->_extResetRequested, false, memory_order_acq_rel)) {
        extR = extW;
        atomic_store_explicit(&io->_extRead, extR, memory_order_release);
    }
    if (extW > extR && (extW - extR) > (JRDExtCapacity / 2)) {
        // Bound wireless glasses audio jitter accumulation without dropping active speech.
        extR = extW - (frames * 4 < (JRDExtCapacity / 2) ? frames * 4 : frames);
    }
    if (io->_extSamples && extW >= extR && (extW - extR) >= frames) {
        for (uint32_t f = 0; f < frames; ++f) {
            int16_t s = io->_extSamples[(extR + f) & JRDExtMask];
            for (uint32_t c = 0; c < io->_inChannels; ++c) {
                io->_input[f * io->_inChannels + c] = s;
            }
        }
            if (io->_diagnosticEnqueueTimes) {
                diagnosticFirstReceipt = io->_diagnosticEnqueueTimes[extR & JRDExtMask];
                diagnosticLastReceipt = io->_diagnosticEnqueueTimes[(extR + frames - 1) & JRDExtMask];
            }
            atomic_store_explicit(&io->_extRead, extR + frames, memory_order_release);
        atomic_fetch_add_explicit(&io->_externalInputFrames, frames, memory_order_relaxed);
        usedExternalMic = true;
    }
    // The input callback is the sole producer, as for the microphone ring.
    // Widen before negation so INT16_MIN has the valid magnitude 32768.
    uint32_t peak = atomic_load_explicit(&io->_inputPeakBeforeGate, memory_order_relaxed);
    uint32_t slicePeak = 0;
    uint32_t sampleCount = frames * io->_inChannels;
    for (uint32_t i = 0; i < sampleCount; ++i) {
        int32_t sample = io->_input[i];
        uint32_t magnitude = (uint32_t)(sample < 0 ? -sample : sample);
        if (magnitude > slicePeak) slicePeak = magnitude;
        if (magnitude > peak) peak = magnitude;
    }
    atomic_store_explicit(&io->_inputPeakBeforeGate, peak, memory_order_relaxed);
    bool gate = atomic_load_explicit(&io->_microphoneGate, memory_order_acquire);
    // Recording is independent of the network transmit gate and keeps un-ducked
    // ambient + wearer glasses audio for natural POV video playback.
        JRDSlot *capture = JRDBeginCapture(io, 0, time, frames, io->_input, gate);
        if (capture) {
            capture->diagnosticExternal = usedExternalMic;
            capture->diagnosticQueued = extW >= extR ? extW - extR : 0;
            capture->diagnosticReadFrame = extR;
            capture->diagnosticFirstReceipt = diagnosticFirstReceipt;
            capture->diagnosticLastReceipt = diagnosticLastReceipt;
        }
        JRDEndCapture(io, 0, capture);
    if (atomic_load_explicit(&io->_recordingWanted, memory_order_acquire)) {
        if (!gate) {
            memset(io->_input, 0, list.mBuffers[0].mDataByteSize);
        } else if (usedExternalMic && echoHoldoff > 0 && slicePeak < 5500) {
            // Un-beamformed DAT glasses mic sits next to open-ear A2DP speakers.
            // Duck low-level speaker bleed during active playback + A2DP tail while
            // letting normal wearer barge-in (>= 5500 PCM16) punch right through.
            for (uint32_t i = 0; i < sampleCount; ++i) {
                io->_input[i] = (int16_t)(io->_input[i] >> 3);
            }
        }
        atomic_fetch_add_explicit(&io->_inputDeliveryCount, 1, memory_order_relaxed);
        status = io->_deliver(flags, time, 1, frames, &list, NULL, nil);
        atomic_store_explicit(&io->_lastInputDeliveryStatus, status, memory_order_relaxed);
        if (status != noErr) {
            atomic_fetch_add_explicit(&io->_inputErrors, 1, memory_order_relaxed);
            atomic_store_explicit(&io->_lastInputStatus, status, memory_order_relaxed);
        }
    }
    goto done;
failed:
    atomic_fetch_add_explicit(&io->_inputErrors, 1, memory_order_relaxed);
    atomic_store_explicit(&io->_lastInputStatus, status, memory_order_relaxed);
    JRDCapture(io, 0, time, frames, NULL, false);
done:
    atomic_fetch_sub_explicit(&io->_callbacks, 1, memory_order_release);
    return status;
}

static OSStatus JRDOutput(void *refcon, AudioUnitRenderActionFlags *flags,
                          const AudioTimeStamp *time, UInt32 bus,
                          UInt32 frames, AudioBufferList *data) {
    __unsafe_unretained JRDDuplexIO *io = (__bridge JRDDuplexIO *)refcon;
    atomic_fetch_add_explicit(&io->_callbacks, 1, memory_order_acq_rel);
    atomic_fetch_add_explicit(&io->_outputCallbackCount, 1, memory_order_relaxed);
    OSStatus status = noErr;
    if (!atomic_load_explicit(&io->_ioEnabled, memory_order_acquire) || frames == 0) {
        JRDSilence(data); goto done;
    }
    if (!time || frames > JRDMaxFrames || !data || data->mNumberBuffers != 1 ||
        !data->mBuffers[0].mData || data->mBuffers[0].mNumberChannels != io->_outChannels ||
        data->mBuffers[0].mDataByteSize < frames * io->_outChannels * sizeof(int16_t)) {
        status = kAudioUnitErr_FormatNotSupported; JRDSilence(data);
        JRDCapture(io, 1, time, frames, NULL, false); goto failed;
    }
    bool playing = atomic_load_explicit(&io->_playoutWanted, memory_order_acquire);
    if (playing) status = io->_play(flags, time, 0, frames, data);
    bool gate = playing && atomic_load_explicit(&io->_outputGate, memory_order_acquire);
    bool voiceAudible = false;
    if (!gate || status != noErr) JRDSilence(data);
    else {
        const int16_t *outPCM = (const int16_t *)data->mBuffers[0].mData;
        uint32_t outCount = frames * io->_outChannels;
        uint32_t outPeak = 0;
        for (uint32_t i = 0; i < outCount; ++i) {
            int32_t s = outPCM[i];
            uint32_t mag = (uint32_t)(s < 0 ? -s : s);
            if (mag > outPeak) outPeak = mag;
        }
        uint32_t previousPeak = atomic_load_explicit(&io->_outputPeakAfterGate, memory_order_relaxed);
        if (outPeak > previousPeak) {
            atomic_store_explicit(&io->_outputPeakAfterGate, outPeak, memory_order_relaxed);
        }
        if (outPeak > 350) {
            voiceAudible = true;
            atomic_store_explicit(&io->_playbackEchoHoldoffFrames, io->_echoHoldoffPresetFrames, memory_order_relaxed);
        }
    }
    // Preserve the voice-only recording contribution BEFORE adding music.
    // Legacy recordings must not accidentally start capturing selected media.
    JRDSlot *capture = JRDBeginCapture(io, 1, time, frames, data->mBuffers[0].mData, gate && status == noErr);
    if (playing && status == noErr && io->_media) {
        uint64_t host = (time->mFlags & kAudioTimeStampHostTimeValid) ? time->mHostTime : 0;
        if (JRDMediaRenderConsumed(io->_media, data->mBuffers[0].mData, frames, host, voiceAudible,
                                   capture ? &capture->media : NULL))
            *flags &= ~kAudioUnitRenderAction_OutputIsSilence;
    }
    if (capture && capture->diagnosticPostMix) {
        memcpy(capture->diagnosticPostMix, data->mBuffers[0].mData,
               frames * io->_outChannels * sizeof(int16_t));
    }
    JRDEndCapture(io, 1, capture);
    if (status != noErr) goto failed;
    goto done;
failed:
    atomic_fetch_add_explicit(&io->_outputErrors, 1, memory_order_relaxed);
    atomic_store_explicit(&io->_lastOutputStatus, status, memory_order_relaxed);
done:
    atomic_fetch_sub_explicit(&io->_callbacks, 1, memory_order_release);
    return status;
}

static AudioStreamBasicDescription JRDFormat(double rate, uint32_t channels) {
    AudioStreamBasicDescription f = {0};
    f.mSampleRate = rate; f.mFormatID = kAudioFormatLinearPCM;
    f.mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked | kAudioFormatFlagsNativeEndian;
    f.mBytesPerPacket = f.mBytesPerFrame = channels * sizeof(int16_t);
    f.mFramesPerPacket = 1; f.mChannelsPerFrame = channels; f.mBitsPerChannel = 16;
    return f;
}

@implementation JRDDuplexIO
- (instancetype)initWithDelegate:(id<RTCAudioDeviceDelegate>)delegate sampleRate:(double)rate
                   inputChannels:(NSInteger)input outputChannels:(NSInteger)output
                  routeGeneration:(uint64_t)generation {
    if (!(self = [super init])) return nil;
    if (!isfinite(rate) || rate < 8000 || rate > 192000 || input < 1 || input > 2 || output < 1 || output > 2) return nil;
    _sampleRate = rate; _inChannels = (uint32_t)input; _outChannels = (uint32_t)output;
    _echoHoldoffPresetFrames = (uint32_t)(rate * 0.32);
    _inputChannels = input; _outputChannels = output; _routeGeneration = generation;
    _deliver = [delegate.deliverRecordedData copy]; _play = [delegate.getPlayoutData copy];
    _input = calloc(JRDMaxFrames * JRDMaxChannels, sizeof(int16_t));
    _extSamples = calloc(JRDExtCapacity, sizeof(int16_t));
    _media = calloc(1, sizeof(JRDMediaPCM));
    if (!_media) return nil;
    JRDMediaInitialize(_media, rate, (uint32_t)output);
    mach_timebase_info_data_t clock;
    mach_timebase_info(&clock);
    _mediaMaxAgeTicks = 500000000ULL * clock.denom / clock.numer;
    for (unsigned i = 0; i < 2; ++i) {
        _rings[i] = calloc(1, sizeof(JRDRing));
        if (!_rings[i]) return nil;
        atomic_init(&_rings[i]->written, 0); atomic_init(&_rings[i]->read, 0);
        atomic_init(&_rings[i]->lostPackets, 0); atomic_init(&_rings[i]->lostFrames, 0);
    }
    atomic_init(&_ioEnabled, false); atomic_init(&_recordingWanted, false); atomic_init(&_playoutWanted, false);
    atomic_init(&_microphoneGate, false); atomic_init(&_outputGate, false);
    atomic_init(&_captureToken, 0); atomic_init(&_callbacks, 0); atomic_init(&_captureReaders, 0);
    atomic_init(&_inputErrors, 0); atomic_init(&_outputErrors, 0);
    atomic_init(&_lastInputStatus, 0); atomic_init(&_lastOutputStatus, 0);
    atomic_init(&_inputCallbackCount, 0); atomic_init(&_renderedInputFrames, 0);
    atomic_init(&_inputDeliveryCount, 0); atomic_init(&_lastInputDeliveryStatus, 0);
    atomic_init(&_inputPeakBeforeGate, 0);
    atomic_init(&_outputCallbackCount, 0); atomic_init(&_outputPeakAfterGate, 0);
    atomic_init(&_extWritten, 0); atomic_init(&_extRead, 0); atomic_init(&_externalInputFrames, 0);
    atomic_init(&_extProducerLock, false); atomic_init(&_extResetRequested, false);
    atomic_init(&_playbackEchoHoldoffFrames, 0);
    _extResamplePos = 0.0; _extPrevSample = 0; _extHasPrevSample = false;
    if (!_input || !_extSamples || !_deliver || !_play || !atomic_is_lock_free(&_captureToken) ||
        !atomic_is_lock_free(&_callbacks) || !atomic_is_lock_free(&_ioEnabled) ||
        !atomic_is_lock_free(&_lastInputStatus)) return nil;
#if DEBUG
    const char *diagnostic = getenv("JARVIS_RECORDING_IO_DIAGNOSTICS");
    if (diagnostic && strcmp(diagnostic, "1") == 0) {
        _diagnosticOutput = calloc(JRDSlots * JRDMaxFrames * JRDMaxChannels, sizeof(int16_t));
        _diagnosticEnqueueTimes = calloc(JRDExtCapacity, sizeof(uint64_t));
        if (_diagnosticOutput && _diagnosticEnqueueTimes) {
            for (unsigned i = 0; i < JRDSlots; ++i)
                _rings[1]->slots[i].diagnosticPostMix = _diagnosticOutput + i * JRDMaxFrames * JRDMaxChannels;
        } else {
            free(_diagnosticOutput); free(_diagnosticEnqueueTimes);
            _diagnosticOutput = NULL; _diagnosticEnqueueTimes = NULL;
        }
    }
#endif
    return self;
}
- (void)dealloc {
    // An allocated unit holds self until close successfully disposes it.
    NSCAssert(_unit == NULL, @"Close the duplex unit before releasing its callback owner");
    free(_input); free(_extSamples); free(_media); free(_rings[0]); free(_rings[1]);
    free(_diagnosticOutput); free(_diagnosticEnqueueTimes);
}
- (OSStatus)prepare {
    if (_closed || _closeRequired) return kAudioUnitErr_CannotDoInCurrentContext;
    if (_unit) return _holdsSelf ? noErr : kAudioUnitErr_CannotDoInCurrentContext;
    AudioComponentDescription description = { .componentType = kAudioUnitType_Output,
        .componentSubType = kAudioUnitSubType_VoiceProcessingIO, .componentManufacturer = kAudioUnitManufacturer_Apple };
#if TARGET_OS_IPHONE
    // DAT supplies microphone PCM. Do not bind it to the voice-processing
    // unit's microphone path; use the negotiated call-audio output instead.
    _rawAudio = true;
    description.componentSubType = kAudioUnitSubType_RemoteIO;
#endif
    AudioComponent component = AudioComponentFindNext(NULL, &description);
    if (!component) return kAudioUnitErr_FailedInitialization;
    OSStatus status = AudioComponentInstanceNew(component, &_unit);
    if (status != noErr) return status;
    CFRetain((__bridge CFTypeRef)self); _holdsSelf = true;
    return noErr;
}
- (OSStatus)configureWithSampleRate:(double)rate inputChannels:(NSInteger)input
                    outputChannels:(NSInteger)output {
    if (_closed || _closeRequired || _configurationAttempted) return kAudioUnitErr_CannotDoInCurrentContext;
    if (!isfinite(rate) || rate < 8000 || rate > 192000 || input < 1 || input > 2 || output < 1 || output > 2)
        return kAudioUnitErr_FormatNotSupported;
    // Buffers are preallocated for JRDMaxChannels. Only format metadata changes
    // here, before initialization/callbacks; the allocated unit is retained.
    _sampleRate = rate; _inChannels = (uint32_t)input; _outChannels = (uint32_t)output;
    _echoHoldoffPresetFrames = (uint32_t)(rate * 0.32);
    _inputChannels = input; _outputChannels = output;
    JRDMediaInitialize(_media, rate, (uint32_t)output);
    return [self configure];
}
- (OSStatus)configure {
    if (_closed || _closeRequired || _configurationAttempted) return kAudioUnitErr_CannotDoInCurrentContext;
    OSStatus status = [self prepare];
    if (status != noErr) return status;
    _configurationAttempted = true;
    UInt32 one = 1, zero = 0, maximum = JRDMaxFrames;
    AudioStreamBasicDescription input = JRDFormat(_sampleRate, _inChannels);
    AudioStreamBasicDescription output = JRDFormat(_sampleRate, _outChannels);
    AURenderCallbackStruct inputCallback = { .inputProc = JRDInput, .inputProcRefCon = (__bridge void *)self };
    AURenderCallbackStruct outputCallback = { .inputProc = JRDOutput, .inputProcRefCon = (__bridge void *)self };
#define JRD_SET(property, scope, element, value) \
    do { status = AudioUnitSetProperty(_unit, property, scope, element, &(value), sizeof(value)); \
         if (status != noErr) goto failed; } while (0)
    JRD_SET(kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, one);
    JRD_SET(kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, one);
    JRD_SET(kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, input);
    JRD_SET(kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, output);
    JRD_SET(kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0, maximum);
    JRD_SET(kAudioUnitProperty_ShouldAllocateBuffer, kAudioUnitScope_Output, 1, zero);
    // Match native iOS voice-processing intent: do not bypass AEC/processing.
    if (!_rawAudio) {
        JRD_SET(kAUVoiceIOProperty_BypassVoiceProcessing, kAudioUnitScope_Global, 0, zero);
    }
    JRD_SET(kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 1, inputCallback);
    JRD_SET(kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, outputCallback);
#undef JRD_SET
    status = AudioUnitInitialize(_unit);
    if (status != noErr) goto failed;
    _configured = true;
    // Match the pinned WebRTC AGC check: initialized unit, microphone bus 1.
    if (!_rawAudio) {
        UInt32 agc = 0, agcSize = sizeof(agc);
        status = AudioUnitGetProperty(_unit, kAUVoiceIOProperty_VoiceProcessingEnableAGC,
                                     kAudioUnitScope_Global, 1, &agc, &agcSize);
        NSLog(@"[JarvisDuplex] agc_initial bus=1 status=%d value=%u", (int)status, (unsigned)agc);
        if (status != noErr) goto failed;
        if (!agc) {
            status = AudioUnitSetProperty(_unit, kAUVoiceIOProperty_VoiceProcessingEnableAGC,
                                         kAudioUnitScope_Global, 1, &one, sizeof(one));
            if (status != noErr) {
                NSLog(@"[JarvisDuplex] agc_enable bus=1 status=%d", (int)status);
                goto failed;
            }
            agcSize = sizeof(agc);
            status = AudioUnitGetProperty(_unit, kAUVoiceIOProperty_VoiceProcessingEnableAGC,
                                         kAudioUnitScope_Global, 1, &agc, &agcSize);
            NSLog(@"[JarvisDuplex] agc_readback bus=1 status=%d value=%u", (int)status, (unsigned)agc);
            if (status != noErr) goto failed;
            if (!agc) { status = kAudioUnitErr_FailedInitialization; goto failed; }
        }
    }
    // Accept only the exact PCM contract configured above. Never label an
    // unexpected layout/rate as though it were the requested format.
    for (unsigned i = 0; i < 2; ++i) {
        AudioStreamBasicDescription actual = {0}; UInt32 size = sizeof(actual);
        status = AudioUnitGetProperty(_unit, kAudioUnitProperty_StreamFormat,
            i == 0 ? kAudioUnitScope_Output : kAudioUnitScope_Input, i == 0 ? 1 : 0, &actual, &size);
        AudioStreamBasicDescription expected = i == 0 ? input : output;
        if (status != noErr) goto failed;
        if (actual.mSampleRate != expected.mSampleRate || actual.mFormatID != expected.mFormatID ||
            actual.mFormatFlags != expected.mFormatFlags || actual.mChannelsPerFrame != expected.mChannelsPerFrame ||
            actual.mBitsPerChannel != 16 || actual.mBytesPerFrame != expected.mBytesPerFrame ||
            actual.mFramesPerPacket != 1 || actual.mBytesPerPacket != expected.mBytesPerPacket) {
            status = kAudioUnitErr_FormatNotSupported; goto failed;
        }
    }
    UInt32 size = sizeof(maximum);
    status = AudioUnitGetProperty(_unit, kAudioUnitProperty_MaximumFramesPerSlice,
                                 kAudioUnitScope_Global, 0, &maximum, &size);
    if (status != noErr) goto failed;
    if (maximum == 0 || maximum > JRDMaxFrames) { status = kAudioUnitErr_TooManyFramesToProcess; goto failed; }
    return noErr;
failed:
    // On failed disposal the self-held object remains available for close retry.
    return status;
}
- (OSStatus)start {
    if (_closeRequired) return kAudioUnitErr_CannotDoInCurrentContext;
    if (_closed || !_configured) return kAudioUnitErr_Uninitialized;
    if (_started) return noErr;
    atomic_store_explicit(&_ioEnabled, true, memory_order_release);
    OSStatus status = AudioOutputUnitStart(_unit);
    if (status == noErr) _started = true;
    else atomic_store_explicit(&_ioEnabled, false, memory_order_release);
    return status;
}
- (BOOL)stopAndFence {
    atomic_store_explicit(&_ioEnabled, false, memory_order_release);
    if (_started) {
        if (AudioOutputUnitStop(_unit) != noErr) { _closeRequired = true; return NO; }
        _started = false;
    }
    BOOL fenced = JRDWaitForZero(&_callbacks);
    if (!fenced) _closeRequired = true;
    return fenced;
}
- (BOOL)close {
    if (_closed) return YES;
    _closeRequired = true;
    if (![self stopAndFence] || ![self endCapture]) return NO;
    if (_configured) {
        if (AudioUnitUninitialize(_unit) != noErr) return NO;
        _configured = false;
    }
    if (_unit) {
        if (AudioComponentInstanceDispose(_unit) != noErr) return NO;
        _unit = NULL;
    }
    _closed = true;
    bool held = _holdsSelf; _holdsSelf = false;
    if (held) CFRelease((__bridge CFTypeRef)self);
    return YES;
}
- (void)setRequestsRecording:(BOOL)recording playout:(BOOL)playout {
    atomic_store_explicit(&_recordingWanted, recording, memory_order_release);
    atomic_store_explicit(&_playoutWanted, playout, memory_order_release);
}
- (void)setGatesMicrophone:(BOOL)microphone output:(BOOL)output {
    atomic_store_explicit(&_microphoneGate, microphone, memory_order_release);
    atomic_store_explicit(&_outputGate, output, memory_order_release);
}
- (void)pushExternalMicrophoneSamples:(const int16_t *)samples
                                count:(uint32_t)count
                           sampleRate:(double)sampleRate {
    if (!samples || count == 0 || !isfinite(sampleRate) || sampleRate < 8000 || sampleRate > 192000) return;
    if (_closed || _closeRequired || !_extSamples) return;
    uint64_t receiptHost = _diagnosticEnqueueTimes ? mach_absolute_time() : 0;
    while (atomic_exchange_explicit(&_extProducerLock, true, memory_order_acquire)) {
        struct timespec spin = { .tv_sec = 0, .tv_nsec = 50000 };
        nanosleep(&spin, NULL);
    }
    double dstRate = _sampleRate > 0 ? _sampleRate : 48000.0;
    double step = sampleRate / dstRate;
    if (!_extHasPrevSample) {
        _extPrevSample = samples[0];
        _extHasPrevSample = true;
        _extResamplePos = 0.0;
    }
    uint64_t r = atomic_load_explicit(&_extRead, memory_order_acquire);
    uint64_t w = atomic_load_explicit(&_extWritten, memory_order_relaxed);
    while (_extResamplePos < (double)count) {
        uint32_t idx = (uint32_t)_extResamplePos;
        double frac = _extResamplePos - (double)idx;
        double s0 = (idx == 0) ? (double)_extPrevSample : (double)samples[idx - 1];
        double s1 = (double)samples[idx];
        double interp = s0 + (s1 - s0) * frac;
        if (interp > 32767.0) interp = 32767.0;
        else if (interp < -32768.0) interp = -32768.0;
        if (w - r < JRDExtCapacity) {
            _extSamples[w & JRDExtMask] = (int16_t)lrint(interp);
            if (_diagnosticEnqueueTimes) _diagnosticEnqueueTimes[w & JRDExtMask] = receiptHost;
            w += 1;
        }
        _extResamplePos += step;
    }
    _extResamplePos -= (double)count;
    if (_extResamplePos < 0.0) _extResamplePos = 0.0;
    _extPrevSample = samples[count - 1];
    atomic_store_explicit(&_extWritten, w, memory_order_release);
    atomic_store_explicit(&_extProducerLock, false, memory_order_release);
}
- (void)clearExternalMicrophoneSamples {
    while (atomic_exchange_explicit(&_extProducerLock, true, memory_order_acquire)) {
        struct timespec spin = { .tv_sec = 0, .tv_nsec = 50000 };
        nanosleep(&spin, NULL);
    }
    _extHasPrevSample = false;
    _extResamplePos = 0.0;
    atomic_store_explicit(&_extResetRequested, true, memory_order_release);
    atomic_store_explicit(&_extProducerLock, false, memory_order_release);
}
- (void)selectMediaSource:(uint64_t)generation {
    JRDMediaSelect(_media, generation);
}
- (BOOL)hasRenderedMediaSource:(uint64_t)generation {
    return JRDMediaHasRendered(_media, generation);
}
- (BOOL)pushMediaSamples:(const int16_t *)samples frames:(uint32_t)frames
             generation:(uint64_t)generation {
    return [self pushMediaSamples:samples frames:frames generation:generation sourceFrame:NAN];
}
- (BOOL)pushMediaSamples:(const int16_t *)samples frames:(uint32_t)frames
             generation:(uint64_t)generation sourceFrame:(double)sourceFrame {
    return JRDMediaEnqueueMapped(_media, samples, frames, generation,
                           mach_absolute_time() + _mediaMaxAgeTicks, sourceFrame);
}
- (void)setMediaAllowed:(BOOL)allowed muted:(BOOL)muted gain:(float)gain {
    uint32_t level = isfinite(gain) ? (uint32_t)lrintf(fmaxf(0, fminf(1, gain)) * 1000) : 0;
    atomic_store_explicit(&_media->baseGainPermille, level, memory_order_relaxed);
    atomic_store_explicit(&_media->muted, muted, memory_order_relaxed);
    atomic_store_explicit(&_media->allowed, allowed, memory_order_release);
}
- (void)setMediaUserSpeaking:(BOOL)speaking {
    atomic_store_explicit(&_media->userSpeaking, speaking, memory_order_release);
}
- (BOOL)beginCapture:(uint64_t)token {
    if (token == 0) return NO;
    uint64_t empty = 0;
    return atomic_compare_exchange_strong_explicit(&_captureToken, &empty, token,
                                                   memory_order_acq_rel, memory_order_acquire);
}
- (BOOL)endCapture {
    atomic_store_explicit(&_captureToken, 0, memory_order_release);
    return JRDWaitForZero(&_captureReaders);
}
- (JRDDuplexPacket *)takePacket:(NSInteger)track {
    if (track < 0 || track > 1) return nil;
    JRDRing *ring = _rings[track];
    uint64_t r = atomic_load_explicit(&ring->read, memory_order_relaxed);
    if (r == atomic_load_explicit(&ring->written, memory_order_acquire)) return nil;
    JRDSlot *slot = &ring->slots[r % JRDSlots];
    JRDDuplexPacket *packet = [JRDDuplexPacket new];
    packet.pcm16 = [NSData dataWithBytes:slot->samples length:(NSUInteger)slot->frames *
                   (track == 0 ? _inChannels : _outChannels) * sizeof(int16_t)];
    packet.track = track; packet.frameCount = slot->frames; packet.captureToken = slot->token;
    packet.hostTime = slot->time.mHostTime; packet.sampleTime = slot->time.mSampleTime;
    packet.timestampFlags = slot->time.mFlags; packet.softwareGateOpen = slot->gate;
    if (_diagnosticEnqueueTimes) {
        if (slot->diagnosticPostMix)
            packet.diagnosticPlaybackPCM16 = [NSData dataWithBytes:slot->diagnosticPostMix
                length:(NSUInteger)slot->frames * _outChannels * sizeof(int16_t)];
        packet.diagnosticTiming = @{
            @"capture_copy_host_time": @(slot->diagnosticCopyHost),
            @"external_microphone": @(slot->diagnosticExternal),
            @"external_queued_frames_before_read": @(slot->diagnosticQueued),
            @"external_read_frame": @(slot->diagnosticReadFrame),
            @"external_first_submit_host_time": @(slot->diagnosticFirstReceipt),
            @"external_last_submit_host_time": @(slot->diagnosticLastReceipt)
        };
    }
    NSMutableArray<JRDConsumedMediaSpan *> *spans = [NSMutableArray arrayWithCapacity:slot->media.count];
    for (uint32_t i = 0; i < slot->media.count; ++i) {
        JRDMediaConsumedSpan value = slot->media.spans[i];
        JRDConsumedMediaSpan *span = [JRDConsumedMediaSpan new];
        span.sourceGeneration = value.generation; span.sourceFrame = value.sourceFrame;
        span.outputOffset = value.outputOffset;
        span.gains = [NSData dataWithBytes:slot->media.gains + value.outputOffset length:value.frames * sizeof(float)];
        [spans addObject:span];
    }
    packet.mediaSpans = spans;
    packet.mediaOutputFrames = slot->media.outputFrames;
    packet.mediaUnknownFrames = slot->media.unknownFrames;
    packet.mediaDroppedFrames = slot->media.droppedFrames;
    atomic_store_explicit(&ring->read, r + 1, memory_order_release);
    return packet;
}
- (JRDDuplexCounters)counters {
    return (JRDDuplexCounters) {
        .microphoneDroppedPackets = atomic_load_explicit(&_rings[0]->lostPackets, memory_order_relaxed),
        .microphoneDroppedFrames = atomic_load_explicit(&_rings[0]->lostFrames, memory_order_relaxed),
        .outputDroppedPackets = atomic_load_explicit(&_rings[1]->lostPackets, memory_order_relaxed),
        .outputDroppedFrames = atomic_load_explicit(&_rings[1]->lostFrames, memory_order_relaxed),
        .microphoneErrors = atomic_load_explicit(&_inputErrors, memory_order_relaxed),
        .outputErrors = atomic_load_explicit(&_outputErrors, memory_order_relaxed),
        .lastMicrophoneStatus = atomic_load_explicit(&_lastInputStatus, memory_order_relaxed),
        .lastOutputStatus = atomic_load_explicit(&_lastOutputStatus, memory_order_relaxed),
        .inputCallbackCount = atomic_load_explicit(&_inputCallbackCount, memory_order_relaxed),
        .renderedInputFrames = atomic_load_explicit(&_renderedInputFrames, memory_order_relaxed),
        .inputDeliveryCount = atomic_load_explicit(&_inputDeliveryCount, memory_order_relaxed),
        .lastInputDeliveryStatus = atomic_load_explicit(&_lastInputDeliveryStatus, memory_order_relaxed),
        .inputPeakBeforeGate = atomic_load_explicit(&_inputPeakBeforeGate, memory_order_relaxed),
        .externalInputFrames = atomic_load_explicit(&_externalInputFrames, memory_order_relaxed),
        .outputCallbackCount = atomic_load_explicit(&_outputCallbackCount, memory_order_relaxed),
        .outputPeakAfterGate = atomic_load_explicit(&_outputPeakAfterGate, memory_order_relaxed),
        .recordingRequested = atomic_load_explicit(&_recordingWanted, memory_order_relaxed),
        .microphoneGateOpen = atomic_load_explicit(&_microphoneGate, memory_order_relaxed),
        .playoutRequested = atomic_load_explicit(&_playoutWanted, memory_order_relaxed),
        .outputGateOpen = atomic_load_explicit(&_outputGate, memory_order_relaxed)
    };
}
- (NSDictionary<NSString *, NSNumber *> *)controlDiagnosticSnapshot {
    if (!_unit || !_configured || _closed) return @{ @"configured": @NO };
    NSMutableDictionary<NSString *, NSNumber *> *result = [@{
        @"configured": @YES, @"started": @(_started), @"raw_audio_unit": @(_rawAudio)
    } mutableCopy];
    const struct {
        __unsafe_unretained NSString *name;
        AudioUnitPropertyID property;
        AudioUnitScope scope;
    } properties[] = {
        { @"agc", kAUVoiceIOProperty_VoiceProcessingEnableAGC, kAudioUnitScope_Global },
        { @"microphone_muted", kAUVoiceIOProperty_MuteOutput, kAudioUnitScope_Global },
        { @"processing_bypassed", kAUVoiceIOProperty_BypassVoiceProcessing, kAudioUnitScope_Global },
        { @"input_enabled", kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input }
    };
    for (unsigned i = _rawAudio ? 3 : 0; i < sizeof(properties) / sizeof(properties[0]); ++i) {
        UInt32 value = 0, size = sizeof(value);
        OSStatus status = AudioUnitGetProperty(_unit, properties[i].property,
                                              properties[i].scope, 1, &value, &size);
        result[[properties[i].name stringByAppendingString:@"_status"]] = @(status);
        if (status == noErr) result[properties[i].name] = @(value);
    }
    AudioStreamBasicDescription format = {0};
    UInt32 size = sizeof(format);
    OSStatus status = AudioUnitGetProperty(_unit, kAudioUnitProperty_StreamFormat,
                                          kAudioUnitScope_Output, 1, &format, &size);
    result[@"input_format_status"] = @(status);
    if (status == noErr) {
        result[@"input_sample_rate"] = @(format.mSampleRate);
        result[@"input_channels"] = @(format.mChannelsPerFrame);
    }
    return result;
}
@end
