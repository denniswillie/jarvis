#import <Foundation/Foundation.h>
#import <CoreAudio/CoreAudio.h>
#import <CoreAudio/CATapDescription.h>
#import <CoreAudio/AudioHardwareTapping.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach/mach_time.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { MaxFrames = 960, QueueSlots = 64, MaxWireBytes = 2 + 12 + MaxFrames * 4 };
enum { BadBuffers = 1, FormatChanged = 2 };
_Static_assert(ATOMIC_INT_LOCK_FREE == 2, "Capture requires lock-free unsigned atomics");

typedef struct {
    uint64_t capturedAt;
    size_t length;
    uint8_t bytes[MaxWireBytes];
} Packet;

typedef struct {
    Packet packets[QueueSlots];
    _Atomic unsigned head;
    _Atomic unsigned tail;
    _Atomic unsigned fault;
    uint64_t staleTicks;
    uint64_t droppedFull;  // Producer-owned; read only after capture stops.
    uint64_t droppedStale; // Consumer-owned.
    uint32_t timestamp;
    uint32_t ssrc;
    uint16_t sequence;
    bool firstPacket;
    bool planar;
} Capture;

static Capture capture;
static volatile sig_atomic_t stopping;

static void StopSignal(int signalNumber) {
    (void)signalNumber;
    stopping = 1;
}

static void Put16(uint8_t *p, uint16_t value) {
    p[0] = (uint8_t)(value >> 8);
    p[1] = (uint8_t)value;
}

static void Put32(uint8_t *p, uint32_t value) {
    p[0] = (uint8_t)(value >> 24);
    p[1] = (uint8_t)(value >> 16);
    p[2] = (uint8_t)(value >> 8);
    p[3] = (uint8_t)value;
}

static int16_t L16(float value) {
    if (!isfinite(value)) return 0;
    if (value >= 1.0f) return INT16_MAX;
    if (value <= -1.0f) return INT16_MIN;
    return (int16_t)(value * 32768.0f);
}

static void Initialize(Capture *c, bool synthetic) {
    // Allocate/touch all queue storage before the real-time callback starts.
    memset(c, 0, sizeof(*c));
    atomic_init(&c->head, 0);
    atomic_init(&c->tail, 0);
    atomic_init(&c->fault, 0);
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    c->staleTicks = (UINT64_C(500000000) * timebase.denom) / timebase.numer;
    c->firstPacket = true;
    if (synthetic) {
        c->ssrc = UINT32_C(0x534f4c4f);
    } else {
        uint32_t random[3];
        arc4random_buf(random, sizeof(random));
        c->ssrc = random[0];
        c->timestamp = random[1];
        c->sequence = (uint16_t)random[2];
    }
}

// Single producer. No allocation, locks, I/O, Objective-C, or dispatch here.
static void Packetize(Capture *c, const float *left, const float *right,
                      unsigned stride, uint32_t frames, uint64_t capturedAt) {
    for (uint32_t offset = 0; offset < frames;) {
        uint32_t count = frames - offset;
        if (count > MaxFrames) count = MaxFrames;
        uint16_t sequence = c->sequence++;
        uint32_t timestamp = c->timestamp;
        c->timestamp += count; // Includes every dropped capture frame.
        unsigned head = atomic_load_explicit(&c->head, memory_order_relaxed);
        unsigned tail = atomic_load_explicit(&c->tail, memory_order_acquire);
        if (head - tail >= QueueSlots) {
            c->droppedFull++;
        } else {
            Packet *p = &c->packets[head % QueueSlots];
            p->capturedAt = capturedAt;
            p->length = 2 + 12 + count * 4;
            Put16(p->bytes, (uint16_t)(p->length - 2));
            uint8_t *rtp = p->bytes + 2;
            rtp[0] = 0x80; // V=2, P=0, X=0, CC=0.
            rtp[1] = 96 | (c->firstPacket ? 0x80 : 0);
            Put16(rtp + 2, sequence);
            Put32(rtp + 4, timestamp);
            Put32(rtp + 8, c->ssrc);
            for (uint32_t i = 0; i < count; i++) {
                size_t sample = (size_t)(offset + i) * stride;
                Put16(rtp + 12 + i * 4, (uint16_t)L16(left[sample]));
                Put16(rtp + 14 + i * 4, (uint16_t)L16(right[sample]));
            }
            c->firstPacket = false;
            atomic_store_explicit(&c->head, head + 1, memory_order_release);
        }
        offset += count;
    }
}

static bool SupportedFormat(AudioStreamBasicDescription f) {
    bool planar = (f.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0;
    UInt32 bytes = planar ? 4 : 8;
    UInt32 flags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked |
                   kAudioFormatFlagsNativeEndian |
                   (planar ? kAudioFormatFlagIsNonInterleaved : 0);
    return f.mFormatID == kAudioFormatLinearPCM && f.mSampleRate == 48000.0 &&
           f.mChannelsPerFrame == 2 && f.mBitsPerChannel == 32 &&
           f.mFramesPerPacket == 1 && f.mBytesPerFrame == bytes &&
           f.mBytesPerPacket == bytes &&
           (f.mFormatFlags & ~kAudioFormatFlagIsNonMixable) == flags;
}

static OSStatus CaptureIO(AudioObjectID device, const AudioTimeStamp *now,
                         const AudioBufferList *input, const AudioTimeStamp *inputTime,
                         AudioBufferList *output, const AudioTimeStamp *outputTime,
                         void *context) {
    (void)device; (void)now; (void)output; (void)outputTime;
    Capture *c = context;
    if (atomic_load_explicit(&c->fault, memory_order_relaxed)) return noErr;
    if (!input || !input->mNumberBuffers) return noErr;
    const float *left = NULL;
    const float *right = NULL;
    uint32_t frames = 0;
    if (c->planar && input->mNumberBuffers == 2 &&
        input->mBuffers[0].mNumberChannels == 1 && input->mBuffers[1].mNumberChannels == 1 &&
        input->mBuffers[0].mDataByteSize == input->mBuffers[1].mDataByteSize &&
        input->mBuffers[0].mDataByteSize % 4 == 0) {
        frames = input->mBuffers[0].mDataByteSize / 4;
        left = input->mBuffers[0].mData;
        right = input->mBuffers[1].mData;
    } else if (!c->planar && input->mNumberBuffers == 1 &&
               input->mBuffers[0].mNumberChannels == 2 &&
               input->mBuffers[0].mDataByteSize % 8 == 0) {
        frames = input->mBuffers[0].mDataByteSize / 8;
        left = input->mBuffers[0].mData;
        right = left ? left + 1 : NULL;
    } else {
        atomic_store_explicit(&c->fault, BadBuffers, memory_order_relaxed);
        return noErr;
    }
    if (!frames) return noErr;
    if (!left || !right) {
        atomic_store_explicit(&c->fault, BadBuffers, memory_order_relaxed);
        return noErr;
    }
    uint64_t capturedAt = inputTime && (inputTime->mFlags & kAudioTimeStampHostTimeValid)
                             ? inputTime->mHostTime : mach_absolute_time();
    Packetize(c, left, right, c->planar ? 1 : 2, frames, capturedAt);
    return noErr;
}

static OSStatus ChangedFormat(AudioObjectID object, UInt32 count,
                              const AudioObjectPropertyAddress *addresses, void *context) {
    (void)object; (void)count; (void)addresses;
    Capture *c = context;
    atomic_store_explicit(&c->fault, FormatChanged, memory_order_relaxed);
    return noErr;
}

// Single consumer on the main thread. A slot stays owned until fully written/dropped.
static int WritePackets(Capture *c, bool finite) {
    size_t offset = 0;
    bool firstWritten = true;
    while (!stopping) {
        unsigned fault = atomic_load_explicit(&c->fault, memory_order_relaxed);
        if (fault) {
            fprintf(stderr, "capture: %s; stopping without conversion\n",
                    fault == FormatChanged ? "audio format changed" : "unsupported audio buffers");
            return 1;
        }
        unsigned tail = atomic_load_explicit(&c->tail, memory_order_relaxed);
        unsigned head = atomic_load_explicit(&c->head, memory_order_acquire);
        if (head == tail) {
            if (finite) return 0;
            poll(NULL, 0, 5);
            continue;
        }
        Packet *p = &c->packets[tail % QueueSlots];
        uint64_t now = mach_absolute_time();
        if (now > p->capturedAt && now - p->capturedAt > c->staleTicks) {
            if (offset) {
                // Skipping a partial RFC4571 frame would desynchronize the entire stream.
                fprintf(stderr, "capture: partially written packet expired; closing stream\n");
                return 1;
            }
            c->droppedStale++;
            atomic_store_explicit(&c->tail, tail + 1, memory_order_release);
            continue;
        }
        if (!offset && firstWritten) p->bytes[3] |= 0x80;
        ssize_t written = write(STDOUT_FILENO, p->bytes + offset, p->length - offset);
        if (written > 0) {
            offset += (size_t)written;
            if (offset == p->length) {
                firstWritten = false;
                offset = 0;
                atomic_store_explicit(&c->tail, tail + 1, memory_order_release);
            }
            continue;
        }
        if (written < 0 && errno == EINTR) continue;
        if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            struct pollfd fd = { .fd = STDOUT_FILENO, .events = POLLOUT };
            int ready = poll(&fd, 1, 5);
            if (ready >= 0 && !(fd.revents & (POLLERR | POLLHUP | POLLNVAL))) continue;
            if (ready < 0 && errno == EINTR) continue;
            fprintf(stderr, "capture: stdout pipe unavailable\n");
            return 1;
        }
        fprintf(stderr, "capture: stdout write failed (errno %d)\n", written < 0 ? errno : EIO);
        return 1;
    }
    return 0;
}

static void Check(OSStatus status, NSString *operation) {
    if (status != noErr) {
        @throw [NSException exceptionWithName:@"CaptureError"
            reason:[NSString stringWithFormat:@"%@: OSStatus %d", operation, (int)status]
            userInfo:nil];
    }
}

static AudioStreamBasicDescription ReadFormat(AudioObjectID object,
                                              AudioObjectPropertyAddress address) {
    AudioStreamBasicDescription format = {0};
    UInt32 size = sizeof(format);
    Check(AudioObjectGetPropertyData(object, &address, 0, NULL, &size, &format), @"read audio format");
    if (size != sizeof(format) || !SupportedFormat(format)) {
        fprintf(stderr, "capture: unsupported format: rate=%.0f channels=%u bits=%u flags=0x%x bytes/frame=%u\n",
                format.mSampleRate, (unsigned)format.mChannelsPerFrame,
                (unsigned)format.mBitsPerChannel, (unsigned)format.mFormatFlags,
                (unsigned)format.mBytesPerFrame);
        @throw [NSException exceptionWithName:@"CaptureError"
            reason:@"requires actual native packed Float32 stereo at 48000 Hz; no resampler"
            userInfo:nil];
    }
    return format;
}

static void CleanupStatus(OSStatus status, const char *operation, int *result) {
    if (status != noErr) {
        fprintf(stderr, "capture: %s: OSStatus %d\n", operation, (int)status);
        *result = 1;
    }
}

static int RunCapture(pid_t pid, bool muteLocal) {
    AudioObjectID tap = kAudioObjectUnknown, aggregate = kAudioObjectUnknown;
    AudioStreamID stream = kAudioObjectUnknown;
    AudioDeviceIOProcID proc = NULL;
    bool started = false, tapListener = false, streamListener = false;
    int result = 1;
    AudioObjectPropertyAddress tapFormat = {
        kAudioTapPropertyFormat, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain
    };
    AudioObjectPropertyAddress streamFormat = {
        kAudioStreamPropertyVirtualFormat, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain
    };
    @try {
        AudioObjectID process = kAudioObjectUnknown;
        UInt32 size = sizeof(process);
        AudioObjectPropertyAddress address = {
            kAudioHardwarePropertyTranslatePIDToProcessObject,
            kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain
        };
        // Soloist's control socket can be ready before its first audio device.
        // Wait briefly for this exact process; never fall back to system audio.
        OSStatus processStatus = noErr;
        for (unsigned attempt = 0; attempt < 40; attempt++) {
            size = sizeof(process);
            processStatus = AudioObjectGetPropertyData(kAudioObjectSystemObject, &address,
                                                       sizeof(pid), &pid, &size, &process);
            if (processStatus == noErr && size == sizeof(process) && process != kAudioObjectUnknown) break;
            if (kill(pid, 0) != 0 || attempt == 39) break;
            usleep(50000);
        }
        Check(processStatus, @"resolve exact PID");
        if (size != sizeof(process) || process == kAudioObjectUnknown) {
            @throw [NSException exceptionWithName:@"CaptureError"
                reason:@"exact PID has no Core Audio process object" userInfo:nil];
        }
        CATapDescription *description = [[CATapDescription alloc]
            initStereoMixdownOfProcesses:@[@(process)]];
        description.name = @"Jarvis Soloist exact-process capture";
        description.privateTap = YES;
        description.muteBehavior = muteLocal ? CATapMutedWhenTapped : CATapUnmuted;
        Check(AudioHardwareCreateProcessTap(description, &tap), @"create process tap");
        Check(AudioObjectAddPropertyListener(tap, &tapFormat, ChangedFormat, &capture), @"watch tap format");
        tapListener = true;
        (void)ReadFormat(tap, tapFormat);
        NSDictionary *config = @{
            @kAudioAggregateDeviceNameKey: @"Jarvis Soloist private tap",
            @kAudioAggregateDeviceUIDKey: NSUUID.UUID.UUIDString,
            @kAudioAggregateDeviceIsPrivateKey: @YES,
            @kAudioAggregateDeviceTapAutoStartKey: @YES,
            @kAudioAggregateDeviceTapListKey: @[@{
                @kAudioSubTapUIDKey: description.UUID.UUIDString,
                @kAudioSubTapDriftCompensationKey: @YES
            }]
        };
        Check(AudioHardwareCreateAggregateDevice((__bridge CFDictionaryRef)config, &aggregate),
              @"create private tap-only aggregate");
        address = (AudioObjectPropertyAddress){
            kAudioDevicePropertyStreams, kAudioObjectPropertyScopeInput, kAudioObjectPropertyElementMain
        };
        Check(AudioObjectGetPropertyDataSize(aggregate, &address, 0, NULL, &size), @"get input stream count");
        if (size != sizeof(stream)) {
            @throw [NSException exceptionWithName:@"CaptureError"
                reason:@"requires one tap-only stereo input stream" userInfo:nil];
        }
        Check(AudioObjectGetPropertyData(aggregate, &address, 0, NULL, &size, &stream), @"get input stream");
        Check(AudioObjectAddPropertyListener(stream, &streamFormat, ChangedFormat, &capture), @"watch input format");
        streamListener = true;
        AudioStreamBasicDescription format = ReadFormat(stream, streamFormat);
        capture.planar = (format.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0;
        Check(AudioDeviceCreateIOProcID(aggregate, CaptureIO, &capture, &proc), @"create capture callback");
        if (!stopping && !atomic_load_explicit(&capture.fault, memory_order_relaxed)) {
            Check(AudioDeviceStart(aggregate, proc), @"start capture");
            started = true;
        }
        result = WritePackets(&capture, false);
    } @catch (NSException *error) {
        fprintf(stderr, "capture: %s\n", error.reason.UTF8String ?: "unknown error");
    } @finally {
        if (started) CleanupStatus(AudioDeviceStop(aggregate, proc), "stop capture", &result);
        if (proc) CleanupStatus(AudioDeviceDestroyIOProcID(aggregate, proc), "destroy callback", &result);
        if (streamListener) CleanupStatus(AudioObjectRemovePropertyListener(stream, &streamFormat, ChangedFormat, &capture),
                                          "remove input listener", &result);
        if (tapListener) CleanupStatus(AudioObjectRemovePropertyListener(tap, &tapFormat, ChangedFormat, &capture),
                                       "remove tap listener", &result);
        if (aggregate != kAudioObjectUnknown)
            CleanupStatus(AudioHardwareDestroyAggregateDevice(aggregate), "destroy aggregate", &result);
        if (tap != kAudioObjectUnknown)
            CleanupStatus(AudioHardwareDestroyProcessTap(tap), "destroy tap", &result);
    }
    return result;
}

static int Usage(int status) {
    fprintf(stderr, "Usage: JarvisSoloistCapture --pid PID [--mute-local]\n"
                    "       JarvisSoloistCapture --self-test\n"
                    "stdout is binary RFC4571-framed RTP L16/48000/2, PT96.\n");
    return status;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        bool synthetic = argc == 2 && strcmp(argv[1], "--self-test") == 0;
        bool muteLocal = false;
        pid_t pid = 0;
        if (argc == 2 && strcmp(argv[1], "--help") == 0) return Usage(0);
        if (!synthetic) {
            for (int i = 1; i < argc; i++) {
                if (strcmp(argv[i], "--mute-local") == 0 && !muteLocal) {
                    muteLocal = true;
                } else if (strcmp(argv[i], "--pid") == 0 && !pid && i + 1 < argc) {
                    const char *value = argv[++i];
                    if (!*value || strspn(value, "0123456789") != strlen(value)) return Usage(2);
                    errno = 0;
                    long parsed = strtol(value, NULL, 10);
                    if (errno || parsed <= 0 || parsed > INT_MAX) return Usage(2);
                    pid = (pid_t)parsed;
                } else {
                    return Usage(2);
                }
            }
            if (!pid) return Usage(2);
        }
        struct stat output;
        if (fstat(STDOUT_FILENO, &output) != 0 ||
            !(S_ISFIFO(output.st_mode) || S_ISSOCK(output.st_mode) || S_ISREG(output.st_mode))) {
            fprintf(stderr, "capture: stdout must be a pipe, socket, or file, not a terminal\n");
            return 2;
        }
        struct sigaction action = {0};
        sigemptyset(&action.sa_mask);
        action.sa_handler = StopSignal;
        if (sigaction(SIGINT, &action, NULL) || sigaction(SIGTERM, &action, NULL)) {
            perror("capture: install stop handlers");
            return 1;
        }
        action.sa_handler = SIG_IGN;
        if (sigaction(SIGPIPE, &action, NULL)) {
            perror("capture: ignore SIGPIPE");
            return 1;
        }
        int flags = fcntl(STDOUT_FILENO, F_GETFL);
        if (flags < 0 || fcntl(STDOUT_FILENO, F_SETFL, flags | O_NONBLOCK) < 0) {
            perror("capture: set nonblocking stdout");
            return 1;
        }
        Initialize(&capture, synthetic);
        int result;
        if (synthetic) {
            float stereo[48 * 2];
            for (unsigned i = 0; i < 48; i++) {
                stereo[i * 2] = 0.25f;
                stereo[i * 2 + 1] = -0.25f;
            }
            Packetize(&capture, stereo, stereo + 1, 2, 48, mach_absolute_time());
            result = WritePackets(&capture, true);
        } else {
            result = RunCapture(pid, muteLocal);
        }
        if (fcntl(STDOUT_FILENO, F_SETFL, flags) < 0) {
            perror("capture: restore stdout flags");
            result = 1;
        }
        return result;
    }
}
