// Test-only AudioUnit substitution. Real production callback/ring/lifecycle
// code is compiled below; no microphone, speaker, or actual AudioUnit is opened.
#import "JarvisDuplexTestSupport.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    AudioStreamBasicDescription input, output;
    AURenderCallbackStruct inputCallback, outputCallback;
    BOOL initialized;
    UInt32 agc;
} TestUnit;
static TestUnit *current;
static int failure, starts, allocations, disposals;
static int agcGets, agcSets;
static UInt32 initialAGC = 1;
static BOOL inputSilent, outputSilent, nativeMuted;

static AudioComponent TestFind(AudioComponent ignored, const AudioComponentDescription *description) {
    return (AudioComponent)(uintptr_t)1;
}
static OSStatus TestNew(AudioComponent component, AudioComponentInstance *result) {
    current = calloc(1, sizeof(TestUnit));
    current->agc = initialAGC;
    *result = (AudioComponentInstance)current; allocations++; return noErr;
}
static OSStatus TestSet(AudioUnit value, AudioUnitPropertyID property, AudioUnitScope scope,
                        AudioUnitElement element, const void *data, UInt32 size) {
    TestUnit *unit = (TestUnit *)value;
    if (property == kAudioUnitProperty_StreamFormat) {
        if (scope == kAudioUnitScope_Output) unit->input = *(const AudioStreamBasicDescription *)data;
        else unit->output = *(const AudioStreamBasicDescription *)data;
    } else if (property == kAudioOutputUnitProperty_SetInputCallback) {
        unit->inputCallback = *(const AURenderCallbackStruct *)data;
    } else if (property == kAudioUnitProperty_SetRenderCallback) {
        unit->outputCallback = *(const AURenderCallbackStruct *)data;
    } else if (property == kAUVoiceIOProperty_VoiceProcessingEnableAGC) {
        assert(unit->initialized && scope == kAudioUnitScope_Global && element == 1);
        assert(size == sizeof(UInt32));
        agcSets++;
        if (failure == 5) return -50;
        if (failure != 7) unit->agc = *(const UInt32 *)data;
    }
    return noErr;
}
static OSStatus TestGet(AudioUnit value, AudioUnitPropertyID property, AudioUnitScope scope,
                        AudioUnitElement element, void *data, UInt32 *size) {
    TestUnit *unit = (TestUnit *)value;
    if (property == kAudioUnitProperty_StreamFormat) {
        *(AudioStreamBasicDescription *)data = scope == kAudioUnitScope_Output ? unit->input : unit->output;
    } else if (property == kAudioUnitProperty_MaximumFramesPerSlice) {
        *(UInt32 *)data = 8192;
    } else if (property == kAUVoiceIOProperty_VoiceProcessingEnableAGC) {
        assert(unit->initialized && scope == kAudioUnitScope_Global && element == 1);
        assert(*size == sizeof(UInt32));
        agcGets++;
        if (failure == 4 || (failure == 6 && agcGets == 2)) return -50;
        *(UInt32 *)data = unit->agc;
        *size = sizeof(UInt32);
    } else if (property == kAUVoiceIOProperty_MuteOutput ||
               property == kAUVoiceIOProperty_BypassVoiceProcessing ||
               property == kAudioOutputUnitProperty_EnableIO) {
        assert(element == 1 && *size == sizeof(UInt32));
        if (property == kAUVoiceIOProperty_MuteOutput && failure == 8) return -50;
        *(UInt32 *)data = property == kAudioOutputUnitProperty_EnableIO ? 1 :
            (property == kAUVoiceIOProperty_MuteOutput ? nativeMuted : 0);
    } else { return kAudioUnitErr_InvalidProperty; }
    return noErr;
}
static OSStatus TestInitialize(AudioUnit value) {
    ((TestUnit *)value)->initialized = YES; return noErr;
}
static OSStatus TestStart(AudioUnit value) { starts++; return noErr; }
static OSStatus TestStop(AudioUnit value) { return failure == 1 ? -50 : noErr; }
static OSStatus TestUninitialize(AudioUnit value) {
    if (failure == 2) return -50;
    ((TestUnit *)value)->initialized = NO; return noErr;
}
static OSStatus TestDispose(AudioComponentInstance value) {
    if (failure == 3) return -50;
    if ((void *)current == (void *)value) current = NULL;
    free((void *)value); disposals++; return noErr;
}
static OSStatus TestRender(AudioUnit value, AudioUnitRenderActionFlags *flags,
                           const AudioTimeStamp *time, UInt32 bus, UInt32 frames, AudioBufferList *data) {
    int16_t *pcm = data->mBuffers[0].mData;
    for (UInt32 i = 0; i < frames * data->mBuffers[0].mNumberChannels; ++i) pcm[i] = 123;
    return noErr;
}
#define AudioComponentFindNext TestFind
#define AudioComponentInstanceNew TestNew
#define AudioUnitSetProperty TestSet
#define AudioUnitGetProperty TestGet
#define AudioUnitInitialize TestInitialize
#define AudioOutputUnitStart TestStart
#define AudioOutputUnitStop TestStop
#define AudioUnitUninitialize TestUninitialize
#define AudioComponentInstanceDispose TestDispose
#define AudioUnitRender TestRender
#import "../Jarvis/Services/JarvisDuplexAudioIO.m"

@implementation JRDTestDelegate {
    NSInteger _dispatchDepth;
}
- (double)preferredInputSampleRate { return 48000; }
- (double)preferredOutputSampleRate { return 48000; }
- (NSTimeInterval)preferredInputIOBufferDuration { return .01; }
- (NSTimeInterval)preferredOutputIOBufferDuration { return .01; }
- (RTCAudioDeviceDeliverRecordedDataBlock)deliverRecordedData {
    return ^OSStatus(AudioUnitRenderActionFlags *flags, const AudioTimeStamp *time, NSInteger bus,
                     UInt32 frames, const AudioBufferList *data, void *context,
                     RTCAudioDeviceRenderRecordedDataBlock render) {
        inputSilent = YES;
        const int16_t *pcm = data->mBuffers[0].mData;
        for (UInt32 i = 0; i < frames * data->mBuffers[0].mNumberChannels; ++i) {
            if (pcm[i] != 0) inputSilent = NO;
        }
        return noErr;
    };
}
- (RTCAudioDeviceGetPlayoutDataBlock)getPlayoutData {
    return ^OSStatus(AudioUnitRenderActionFlags *flags, const AudioTimeStamp *time, NSInteger bus,
                     UInt32 frames, AudioBufferList *data) {
        int16_t *pcm = data->mBuffers[0].mData;
        for (UInt32 i = 0; i < frames * data->mBuffers[0].mNumberChannels; ++i) pcm[i] = 456;
        return noErr;
    };
}
- (void)notifyAudioInputParametersChange {
    _parameterNotifications++; if (_dispatchDepth == 0) _notificationViolations++;
}
- (void)notifyAudioOutputParametersChange {
    _parameterNotifications++; if (_dispatchDepth == 0) _notificationViolations++;
}
- (void)notifyAudioInputInterrupted {
    _interruptionNotifications++; if (_dispatchDepth == 0) _notificationViolations++;
}
- (void)notifyAudioOutputInterrupted {
    _interruptionNotifications++; if (_dispatchDepth == 0) _notificationViolations++;
}
- (void)dispatchSync:(dispatch_block_t)block { _dispatchDepth++; block(); _dispatchDepth--; }
- (void)dispatchAsync:(dispatch_block_t)block { [self dispatchSync:block]; }
@end

void JRDTestReset(void) {
    assert(current == NULL);
    failure = starts = allocations = disposals = 0;
    agcGets = agcSets = 0; initialAGC = 1;
    inputSilent = outputSilent = nativeMuted = NO;
}
void JRDTestSetFailure(int stage) { failure = stage; }
int JRDTestStarts(void) { return starts; }
int JRDTestAllocations(void) { return allocations; }
int JRDTestDisposals(void) { return disposals; }
BOOL JRDTestLastInputWasSilent(void) { return inputSilent; }
BOOL JRDTestLastOutputWasSilent(void) { return outputSilent; }
void JRDTestPump(UInt32 frames, UInt64 hostTime) {
    assert(current && frames <= 8192);
    AudioTimeStamp time = {.mSampleTime = 64, .mHostTime = hostTime,
        .mFlags = kAudioTimeStampHostTimeValid | kAudioTimeStampSampleTimeValid};
    AudioUnitRenderActionFlags flags = 0;
    assert(current->inputCallback.inputProc(current->inputCallback.inputProcRefCon,
        &flags, &time, 1, frames, NULL) == noErr);
    int16_t pcm[8192 * 2];
    memset(pcm, 0x7f, sizeof(pcm));
    AudioBufferList list = {.mNumberBuffers = 1, .mBuffers = {{
        .mNumberChannels = current->output.mChannelsPerFrame,
        .mDataByteSize = frames * current->output.mBytesPerFrame, .mData = pcm}}};
    assert(current->outputCallback.inputProc(current->outputCallback.inputProcRefCon,
        &flags, &time, 0, frames, &list) == noErr);
    outputSilent = YES;
    for (UInt32 i = 0; i < frames * list.mBuffers[0].mNumberChannels; ++i) {
        if (pcm[i] != 0) outputSilent = NO;
    }
}
int JRDRunLowLevelChecks(void) {
    int checks = 0;
    // AGC must be checked after initialization on microphone element 1.
    // Disabled AGC is enabled and verified; enabled AGC needs no write.
    for (UInt32 enabled = 0; enabled <= 1; ++enabled) {
        JRDTestReset(); initialAGC = enabled;
        JRDDuplexIO *io = [[JRDDuplexIO alloc] initWithDelegate:[JRDTestDelegate new]
            sampleRate:48000 inputChannels:1 outputChannels:1 routeGeneration:1];
        assert([io configure] == noErr);
        assert(current->initialized && current->agc == 1);
        assert(agcGets == (enabled ? 1 : 2) && agcSets == (enabled ? 0 : 1));
        assert([io start] == noErr && [io close] && disposals == 1);
        checks++;
    }
    // Initial get, enable, and readback errors must propagate. A successful
    // set whose readback stays disabled must not report successful setup.
    for (int stage = 4; stage <= 7; ++stage) {
        JRDTestReset(); initialAGC = 0; JRDTestSetFailure(stage);
        JRDDuplexIO *io = [[JRDDuplexIO alloc] initWithDelegate:[JRDTestDelegate new]
            sampleRate:48000 inputChannels:1 outputChannels:1 routeGeneration:1];
        OSStatus expected = stage == 7 ? kAudioUnitErr_FailedInitialization : -50;
        assert([io configure] == expected && starts == 0);
        assert(agcGets == (stage <= 5 ? 1 : 2) && agcSets == (stage == 4 ? 0 : 1));
        assert([io close] && disposals == 1);
        checks++;
    }
    for (int stage = 1; stage <= 3; stage++) {
        JRDTestReset();
        JRDDuplexIO *io = [[JRDDuplexIO alloc] initWithDelegate:[JRDTestDelegate new]
            sampleRate:48000 inputChannels:1 outputChannels:1 routeGeneration:1];
        assert([io configure] == noErr && [io start] == noErr);
        JRDTestSetFailure(stage);
        assert(![io close]);
        assert([io start] == kAudioUnitErr_CannotDoInCurrentContext && starts == 1);
        JRDTestSetFailure(0);
        assert([io close] && [io close] && disposals == 1);
        assert([io start] != noErr); checks++;
    }
    JRDTestReset();
    JRDDuplexIO *io = [[JRDDuplexIO alloc] initWithDelegate:[JRDTestDelegate new]
        sampleRate:48000 inputChannels:1 outputChannels:1 routeGeneration:2];
    assert([io configure] == noErr && [io start] == noErr && [io beginCapture:7]);
    [io setRequestsRecording:YES playout:YES];
    [io setGatesMicrophone:NO output:NO];
    for (int i = 0; i < 40; i++) JRDTestPump(64, 999);
    assert(inputSilent && outputSilent);
    JRDDuplexCounters lost = [io counters];
    assert(lost.microphoneDroppedPackets == 8 && lost.outputDroppedPackets == 8);
    assert(lost.microphoneDroppedFrames == 512 && lost.outputDroppedFrames == 512); checks++;
    for (NSInteger track = 0; track < 2; track++) {
        for (int i = 0; i < 32; i++) {
            JRDDuplexPacket *packet = [io takePacket:track];
            assert(packet && packet.hostTime == 999 && packet.sampleTime == 64);
            assert(packet.captureToken == 7 && !packet.softwareGateOpen && packet.frameCount == 64);
            const int16_t *pcm = packet.pcm16.bytes;
            assert(pcm[0] == (track == 0 ? 123 : 0));
        }
        assert([io takePacket:track] == nil);
    }
    checks++;
    assert([io endCapture] && [io beginCapture:8]);
    [io setGatesMicrophone:YES output:YES]; JRDTestPump(64, 1001);
    assert(!inputSilent && !outputSilent);
    JRDDuplexPacket *packet = [io takePacket:1];
    assert(packet.captureToken == 8 && packet.softwareGateOpen);
    assert(((const int16_t *)packet.pcm16.bytes)[0] == 456); checks++;
    // Diagnose native mute independently of the application's open gate.
    nativeMuted = YES;
    int previousStarts = starts, previousSets = agcSets;
    NSDictionary *diagnostic = [io controlDiagnosticSnapshot];
    assert([diagnostic[@"configured"] boolValue] && [diagnostic[@"started"] boolValue]);
    assert([diagnostic[@"microphone_muted"] unsignedIntValue] == 1);
    assert([diagnostic[@"agc"] unsignedIntValue] == 1);
    assert([diagnostic[@"input_sample_rate"] doubleValue] == 48000);
    assert([diagnostic[@"input_channels"] unsignedIntValue] == 1);
    JRDTestSetFailure(8);
    diagnostic = [io controlDiagnosticSnapshot];
    assert([diagnostic[@"microphone_muted_status"] intValue] == -50);
    assert(diagnostic[@"microphone_muted"] == nil);
    assert(starts == previousStarts && agcSets == previousSets);
    JRDTestSetFailure(0);
    assert([io close]);
    assert(![[io controlDiagnosticSnapshot][@"configured"] boolValue]);
    checks++; JRDTestReset();

    // External DAT glasses microphone resampling (16 kHz -> 48 kHz), un-ducked POV
    // recording capture, open-ear speaker echo ducking, and builtInMic fallback.
    JRDDuplexIO *extIO = [[JRDDuplexIO alloc] initWithDelegate:[JRDTestDelegate new]
        sampleRate:48000 inputChannels:1 outputChannels:1 routeGeneration:3];
    assert([extIO configure] == noErr && [extIO start] == noErr && [extIO beginCapture:9]);
    [extIO setRequestsRecording:YES playout:YES];
    [extIO setGatesMicrophone:YES output:YES];
    int16_t dat16k[160];
    for (int i = 0; i < 160; ++i) dat16k[i] = 2400;
    [extIO pushExternalMicrophoneSamples:dat16k count:160 sampleRate:16000.0];
    JRDTestPump(240, 2001); // First pump also renders output=456 (>350), arming echo holdoff.
    JRDDuplexPacket *cap1 = [extIO takePacket:0];
    assert(cap1 && ((const int16_t *)cap1.pcm16.bytes)[10] == 2400);
    JRDTestPump(240, 2002); // Consumes remaining 240 resampled frames under echo holdoff.
    JRDDuplexPacket *cap2 = [extIO takePacket:0];
    assert(cap2 && ((const int16_t *)cap2.pcm16.bytes)[10] == 2400); // Recording stays un-ducked!
    assert([extIO counters].externalInputFrames == 480);
    // Ring is now empty -> seamless fallback to AudioUnitRender (123).
    JRDTestPump(240, 2003);
    JRDDuplexPacket *capFallback = [extIO takePacket:0];
    assert(capFallback && ((const int16_t *)capFallback.pcm16.bytes)[0] == 123);
    assert([extIO close]);
    checks++; JRDTestReset();
    // Verify real render-buffer delivery, not just the diagnostic peak. Cover
    // the negotiated stereo A2DP shape as well as mono, with guard samples.
    for (UInt32 channels = 1; channels <= 2; ++channels) {
        JRDTestReset();
        JRDDuplexIO *outputIO = [[JRDDuplexIO alloc] initWithDelegate:[JRDTestDelegate new]
            sampleRate:48000 inputChannels:1 outputChannels:channels routeGeneration:4];
        assert([outputIO configure] == noErr && [outputIO start] == noErr);
        [outputIO setRequestsRecording:YES playout:YES];
        assert(current->output.mSampleRate == 48000);
        assert(current->output.mFormatID == kAudioFormatLinearPCM);
        assert(current->output.mBitsPerChannel == 16);
        assert(current->output.mBytesPerFrame == channels * sizeof(int16_t));
        assert(current->output.mFramesPerPacket == 1);
        assert(!(current->output.mFormatFlags & kAudioFormatFlagIsNonInterleaved));
        for (int open = 0; open <= 1; ++open) {
            [outputIO setGatesMicrophone:YES output:open];
            int16_t guarded[2 + 480 * 2];
            for (size_t i = 0; i < sizeof(guarded) / sizeof(guarded[0]); ++i) guarded[i] = 12345;
            AudioBufferList output = {.mNumberBuffers = 1, .mBuffers = {{
                .mNumberChannels = channels, .mDataByteSize = 480 * channels * sizeof(int16_t),
                .mData = &guarded[1]}}};
            AudioTimeStamp time = {.mSampleTime = 0, .mHostTime = 3000,
                .mFlags = kAudioTimeStampHostTimeValid | kAudioTimeStampSampleTimeValid};
            AudioUnitRenderActionFlags flags = 0;
            assert(current->outputCallback.inputProc(current->outputCallback.inputProcRefCon,
                &flags, &time, 0, 480, &output) == noErr);
            assert(output.mBuffers[0].mData == &guarded[1]);
            assert(output.mBuffers[0].mDataByteSize == 480 * channels * sizeof(int16_t));
            for (UInt32 i = 0; i < 480 * channels; ++i) assert(guarded[i + 1] == (open ? 456 : 0));
            assert(guarded[0] == 12345 && guarded[1 + 480 * channels] == 12345);
        }
        assert([outputIO close]);
        checks++;
    }
    JRDTestReset();
    // Real callback composition: media remains distinct from voice capture and
    // cannot arm voice echo holdoff or leak into microphone delivery.
    JRDDuplexIO *mediaIO = [[JRDDuplexIO alloc] initWithDelegate:[JRDTestDelegate new]
        sampleRate:48000 inputChannels:1 outputChannels:1 routeGeneration:5];
    assert([mediaIO configure] == noErr && [mediaIO start] == noErr);
    [mediaIO setRequestsRecording:YES playout:YES];
    [mediaIO setGatesMicrophone:YES output:NO];
    [mediaIO selectMediaSource:10];
    [mediaIO setMediaAllowed:YES muted:NO gain:1];
    for (int i = 0; i < 32; ++i) JRDTestPump(480, mach_absolute_time());
    assert([mediaIO beginCapture:77]);
    int16_t tone[480], rendered[480];
    for (int i = 0; i < 480; ++i) tone[i] = 10000;
    assert([mediaIO pushMediaSamples:tone frames:480 generation:10 sourceFrame:0]);
    AudioBufferList mediaOutput = {.mNumberBuffers = 1, .mBuffers = {{
        .mNumberChannels = 1, .mDataByteSize = sizeof(rendered), .mData = rendered}}};
    AudioTimeStamp mediaTime = {.mHostTime = mach_absolute_time(), .mFlags = kAudioTimeStampHostTimeValid};
    AudioUnitRenderActionFlags mediaFlags = kAudioUnitRenderAction_OutputIsSilence;
    assert(current->outputCallback.inputProc(current->outputCallback.inputProcRefCon,
        &mediaFlags, &mediaTime, 0, 480, &mediaOutput) == noErr);
    assert(rendered[0] == 10000 && rendered[479] == 10000);
    assert(!(mediaFlags & kAudioUnitRenderAction_OutputIsSilence));
    assert(atomic_load(&mediaIO->_playbackEchoHoldoffFrames) == 0);
    JRDDuplexPacket *voiceOnly = [mediaIO takePacket:1];
    assert(voiceOnly && !voiceOnly.softwareGateOpen);
    assert(voiceOnly.mediaSpans.count == 1 && voiceOnly.mediaOutputFrames == 480);
    assert(voiceOnly.mediaSpans[0].sourceGeneration == 10 && voiceOnly.mediaSpans[0].sourceFrame == 0);
    assert(voiceOnly.mediaSpans[0].gains.length == 480 * sizeof(float));
    assert(((const float *)voiceOnly.mediaSpans[0].gains.bytes)[0] == 1);
    for (int i = 0; i < 480; ++i) assert(((const int16_t *)voiceOnly.pcm16.bytes)[i] == 0);
    JRDTestPump(480, mach_absolute_time());
    JRDDuplexPacket *microphoneOnly = [mediaIO takePacket:0];
    assert(microphoneOnly && ((const int16_t *)microphoneOnly.pcm16.bytes)[0] == 123);
    assert(microphoneOnly.mediaSpans.count == 0 && microphoneOnly.mediaOutputFrames == 0);
    [mediaIO setMediaAllowed:YES muted:YES gain:1];
    assert([mediaIO pushMediaSamples:tone frames:480 generation:10]);
    JRDTestPump(480, mach_absolute_time());
    assert(outputSilent && atomic_load(&mediaIO->_media->queuedFrames) == 0);
    assert([mediaIO close]); checks++;
    JRDTestReset();
    // Sidecars observe final mixed output and external FIFO receipt timing,
    // while the ordinary recorder still gets its unmodified voice-only track.
    for (int enabled = 0; enabled <= 1; ++enabled) {
        setenv("JARVIS_RECORDING_IO_DIAGNOSTICS", enabled ? "1" : "0", 1);
        JRDDuplexIO *probe = [[JRDDuplexIO alloc] initWithDelegate:[JRDTestDelegate new]
            sampleRate:48000 inputChannels:1 outputChannels:1 routeGeneration:6];
        assert([probe configure] == noErr && [probe start] == noErr && [probe beginCapture:80]);
        [probe setRequestsRecording:YES playout:YES]; [probe setGatesMicrophone:YES output:YES];
        [probe selectMediaSource:20]; [probe setMediaAllowed:YES muted:NO gain:1];
        assert([probe pushMediaSamples:tone frames:480 generation:20]);
        AudioTimeStamp timestamp = {.mHostTime = mach_absolute_time(), .mFlags = kAudioTimeStampHostTimeValid};
        AudioUnitRenderActionFlags flags = 0;
        assert(current->outputCallback.inputProc(current->outputCallback.inputProcRefCon,
            &flags, &timestamp, 0, 480, &mediaOutput) == noErr);
        JRDDuplexPacket *out = [probe takePacket:1];
        for (int i = 0; i < 480; ++i) assert(((const int16_t *)out.pcm16.bytes)[i] == 456);
        if (enabled) {
            assert(out.diagnosticPlaybackPCM16.length == sizeof(rendered));
            assert(memcmp(out.diagnosticPlaybackPCM16.bytes, rendered, sizeof(rendered)) == 0);
            assert([out.diagnosticTiming[@"capture_copy_host_time"] unsignedLongLongValue] > 0);
        } else {
            assert(out.diagnosticPlaybackPCM16 == nil && out.diagnosticTiming == nil);
            assert(probe->_diagnosticEnqueueTimes == NULL && probe->_diagnosticOutput == NULL);
        }
        [probe pushExternalMicrophoneSamples:dat16k count:160 sampleRate:16000];
        JRDTestPump(240, mach_absolute_time());
        JRDDuplexPacket *input = [probe takePacket:0];
        if (enabled) {
            assert([input.diagnosticTiming[@"external_microphone"] boolValue]);
            assert([input.diagnosticTiming[@"external_queued_frames_before_read"] unsignedLongLongValue] == 480);
            assert([input.diagnosticTiming[@"external_first_submit_host_time"] unsignedLongLongValue] > 0);
            assert([input.diagnosticTiming[@"external_first_submit_host_time"] isEqual:
                    input.diagnosticTiming[@"external_last_submit_host_time"]]);
        }
        JRDTestPump(240, mach_absolute_time()); (void)[probe takePacket:0];
        JRDTestPump(240, mach_absolute_time()); input = [probe takePacket:0];
        if (enabled) assert(![input.diagnosticTiming[@"external_microphone"] boolValue]);
        assert([probe close]); JRDTestReset(); checks++;
    }
    unsetenv("JARVIS_RECORDING_IO_DIAGNOSTICS");
    return checks;
}
