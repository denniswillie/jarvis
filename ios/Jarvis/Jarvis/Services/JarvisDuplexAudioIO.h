// Opt-in hardware adapter. Compile .m with ARC and import
// this header through the app bridging header. Requires pinned WebRTC 156.
#import <Foundation/Foundation.h>
#import <AudioToolbox/AudioToolbox.h>
#import <WebRTC/RTCAudioDevice.h>

NS_ASSUME_NONNULL_BEGIN

// Cumulative per hardware generation, never an assertion of contiguous loss.
// Core records a baseline when attaching and differences thereafter. Missing
// timestamp flags must remain missing; counters do not manufacture intervals.
typedef struct {
    uint64_t microphoneDroppedPackets;
    uint64_t microphoneDroppedFrames;
    uint64_t outputDroppedPackets;
    uint64_t outputDroppedFrames;
    uint64_t microphoneErrors;
    uint64_t outputErrors;
    int32_t lastMicrophoneStatus;
    int32_t lastOutputStatus;
    // Per-generation diagnostics, read independently with atomic loads.
    // Callback entries include disabled/empty callbacks; frames count only
    // successful input renders. No additional PCM is retained.
    uint64_t inputCallbackCount;
    uint64_t renderedInputFrames;
    uint64_t inputDeliveryCount;
    int32_t lastInputDeliveryStatus;
    uint32_t inputPeakBeforeGate; // Maximum absolute PCM16 sample: 0...32768.
    uint64_t externalInputFrames;
    uint64_t outputCallbackCount;
    uint32_t outputPeakAfterGate; // Software render submission, not audible playback.
    BOOL recordingRequested;
    BOOL microphoneGateOpen;
    BOOL playoutRequested;
    BOOL outputGateOpen;
} JRDDuplexCounters;

@interface JRDConsumedMediaSpan : NSObject
@property(nonatomic, readonly) uint64_t sourceGeneration;
@property(nonatomic, readonly) double sourceFrame;
@property(nonatomic, readonly) uint32_t outputOffset;
@property(nonatomic, readonly) NSData *gains; // Native Float32, one per consumed route frame.
@end

@interface JRDDuplexPacket : NSObject
// Debug-only sidecars; nil unless explicitly enabled. PCM is the final
// post-mix render submission, not proof of acoustic playback.
@property(nonatomic, readonly, nullable) NSData *diagnosticPlaybackPCM16;
@property(nonatomic, readonly, nullable) NSDictionary<NSString *, NSNumber *> *diagnosticTiming;
@property(nonatomic, readonly) NSData *pcm16;
@property(nonatomic, readonly) NSInteger track; // 0 = microphone, 1 = rendered output
@property(nonatomic, readonly) uint32_t frameCount;
@property(nonatomic, readonly) uint64_t captureToken;
@property(nonatomic, readonly) uint64_t hostTime;
@property(nonatomic, readonly) double sampleTime;
@property(nonatomic, readonly) uint32_t timestampFlags;
@property(nonatomic, readonly) BOOL softwareGateOpen;
@property(nonatomic, readonly) NSArray<JRDConsumedMediaSpan *> *mediaSpans;
@property(nonatomic, readonly) uint32_t mediaOutputFrames;
@property(nonatomic, readonly) uint32_t mediaUnknownFrames;
@property(nonatomic, readonly) uint32_t mediaDroppedFrames;
@end

// Lifecycle/configuration: exclusively WebRTC's ADM thread. Capture methods:
// exclusively ONE off-I/O consumer queue. Gates may be set from any thread.
// No AVAudioSession setters here. prepare may precede session activation;
// Core's session lease must be active before configure/start.
// An instance is one immutable hardware-format/route generation, not reusable.
@interface JRDDuplexIO : NSObject
- (nullable instancetype)initWithDelegate:(id<RTCAudioDeviceDelegate>)delegate
                              sampleRate:(double)sampleRate
                           inputChannels:(NSInteger)inputChannels
                          outputChannels:(NSInteger)outputChannels
                         routeGeneration:(uint64_t)routeGeneration;
- (instancetype)init NS_UNAVAILABLE;
@property(nonatomic, readonly) double sampleRate;
@property(nonatomic, readonly) NSInteger inputChannels;
@property(nonatomic, readonly) NSInteger outputChannels;
@property(nonatomic, readonly) uint64_t routeGeneration;
// Idempotent allocation only. Call close even if configure is never called.
- (OSStatus)prepare;
- (OSStatus)configure;
- (OSStatus)configureWithSampleRate:(double)sampleRate
                     inputChannels:(NSInteger)inputChannels
                    outputChannels:(NSInteger)outputChannels
    NS_SWIFT_NAME(configure(sampleRate:inputChannels:outputChannels:));
- (OSStatus)start;
// Failed stop/fence/close is terminal until successfully retried. Do not create
// another unit alongside it. The refcon deliberately stays alive on failure.
- (BOOL)stopAndFence;
- (BOOL)close;
- (void)setRequestsRecording:(BOOL)recording playout:(BOOL)playout
    NS_SWIFT_NAME(setRequests(recording:playout:));
- (void)setGatesMicrophone:(BOOL)microphone output:(BOOL)output
    NS_SWIFT_NAME(setGates(microphone:output:));
// Thread-safe DAT in-stream glasses microphone feed (mono signed PCM16).
// Falls back seamlessly to AudioUnitRender whenever the external ring is empty.
- (void)pushExternalMicrophoneSamples:(const int16_t *)samples
                                count:(uint32_t)count
                           sampleRate:(double)sampleRate
    NS_SWIFT_NAME(pushExternalMicrophoneSamples(_:count:sampleRate:));
- (void)clearExternalMicrophoneSamples;
// One serialized producer, outside I/O. PCM must already match this route's
// sampleRate/outputChannels, <=1920 frames per packet. Never use this playback
// conversion as the original high-rate recording source. Rejected packets
// must be dropped/reported, not accumulated in an unbounded upstream queue.
- (void)selectMediaSource:(uint64_t)generation;
// True only after nonzero samples from this source reached an output callback.
- (BOOL)hasRenderedMediaSource:(uint64_t)generation;
- (BOOL)pushMediaSamples:(const int16_t *)samples frames:(uint32_t)frames
             generation:(uint64_t)generation
    NS_SWIFT_NAME(pushMediaSamples(_:frames:generation:));
- (BOOL)pushMediaSamples:(const int16_t *)samples frames:(uint32_t)frames
             generation:(uint64_t)generation sourceFrame:(double)sourceFrame
    NS_SWIFT_NAME(pushMediaSamples(_:frames:generation:sourceFrame:));
- (void)setMediaAllowed:(BOOL)allowed muted:(BOOL)muted gain:(float)gain
    NS_SWIFT_NAME(setMedia(allowed:muted:gain:));
- (void)setMediaUserSpeaking:(BOOL)speaking;
- (BOOL)beginCapture:(uint64_t)token;
// Does not stop audio. Waits off I/O, bounded to 100 ms, for recorder copies.
- (BOOL)endCapture;
- (nullable JRDDuplexPacket *)takePacket:(NSInteger)track;
- (JRDDuplexCounters)counters;
// Read-only control-thread diagnostics. Never call from an audio callback.
- (NSDictionary<NSString *, NSNumber *> *)controlDiagnosticSnapshot;
@end

NS_ASSUME_NONNULL_END
