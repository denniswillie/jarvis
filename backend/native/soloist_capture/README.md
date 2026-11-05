# Soloist exact-process capture helper

Small macOS 14.2+ command-line app bundle. Reuses the successful private process
tap/private aggregate design in `.jarvis-dev/soloist/AudioProbe.m`, replacing its
statistics callback with a fixed SPSC packet queue. No process discovery, launches,
microphone, global tap, networking, provider calls, retries, or process manager.
Only the caller-supplied Core Audio process object is included. The caller owns
selecting the real Soloist audio PID; helper subprocesses are not auto-included.

## Build and CLI

```sh
bash /Users/denniswillie/jarvis/backend/native/soloist_capture/build.sh
HELPER=/Users/denniswillie/jarvis/.jarvis-dev/soloist/JarvisSoloistCapture.app/Contents/MacOS/JarvisSoloistCapture
"$HELPER" --self-test > /tmp/soloist-synthetic.rfc4571
# Parent-owned live test ONLY, after capture permission is authorized:
"$HELPER" --pid "$SOLOIST_PID" | your_transport_consumer
# Optional: --pid "$SOLOIST_PID" --mute-local
```

Builds the current host architecture with clang, Foundation and CoreAudio; signs
the bundle ad hoc (not notarized). `NSAudioCaptureUsageDescription` is in its plist.
Launch the executable directly, not `open`, to retain the binary stdout pipe.
Normal live startup may invoke macOS audio-capture permission; self-test never
creates a tap or aggregate and needs no audio permission. Default is
`CATapUnmuted`; `--mute-local` uses `CATapMutedWhenTapped`, only during tap reading.
No microphone permission key or hardware input device is requested.

## Output contract

- Stdout contains only RFC4571 frames: two-byte unsigned network-order RTP packet
  length (excludes the prefix), followed immediately by that RTP packet.
- RTP v2, fixed 12-byte header, no padding/extension/CSRC, dynamic payload type 96.
  Configure the receiver out of band as `a=rtpmap:96 L16/48000/2`; there is no SDP,
  RTCP, handshake, or format preamble in stdout.
- Signed 16-bit big-endian PCM in original tap channel order: left, right, repeat.
  Finite float samples multiply by 32768 and truncate toward zero; clip at
  -32768/+32767. NaN and infinity become zero. No dithering or gain adjustment.
- Both the tap format and actual aggregate input stream must already be packed,
  native-endian Float32, stereo, exactly 48000 Hz. Interleaved and two-plane
  noninterleaved layouts are supported and validated per callback. Other formats
  fail; no sample-rate/channel conversion or relabeling. A format-change notice
  stops capture rather than silently renegotiating.
- At most 960 frames (20 ms) per packet, splitting larger callbacks and preserving
  a short final packet. Maximum RTP length 3852 bytes, framed length 3854 bytes.
  This stream framing does not imply a safe UDP datagram/MTU size.
- Sequence increments for every packetization attempt, including queue-full
  drops. Timestamp increments by stereo capture frames, not individual samples,
  including dropped packets. Unsigned wrap is intentional. Writer drops preserve
  the original sequence/timestamps, exposing gaps to the receiver.
- Live SSRC, initial sequence and timestamp are randomized once per process start.
  The first packet actually written has M=1, even if earlier queued packets expired;
  subsequent packets have M=0. No additional marker is added after ordinary gaps.
- Self-test is an explicit deterministic exception: SSRC `0x534f4c4f`, sequence 0,
  timestamp 0, M=1, 48 frames of (+0.25, -0.25), yielding (+8192, -8192).
  One 204-byte RTP packet plus its two-byte prefix, 206 bytes total. It uses the
  same packetizer, queue and writer as live capture, without touching Core Audio.

## Backpressure and stopping

The Core Audio C callback only validates buffer layout, converts samples and
publishes packets through lock-free atomics. It does not allocate, block, invoke
Objective-C, dispatch work, write, or network. Queue storage is 64 fixed packet
slots, touched before capture; at most 245760 queued PCM bytes plus fixed metadata
and headers. Full queue drops the new packet, but still advances the RTP timeline.

The main thread is the sole stdout writer. It polls nonblocking stdout in 5 ms
steps, dropping unsent packets older than 500 ms by Core Audio host time (or
monotonic callback time when unavailable). Bytes already accepted into the OS pipe
cannot be retracted; the downstream transport owns its own latency bound. If a
partially written frame expires, the helper fails/closes instead of skipping its
tail and corrupting all subsequent framing. Consumers must discard an incomplete
final frame at EOF, including interruption or a write failure.

SIGINT/SIGTERM request a clean stop without draining stale music; stop/destroy the
IOProc, remove listeners, destroy aggregate and tap, then restore stdout flags.
Pipe closure/error also takes this cleanup path; SIGPIPE is ignored. Diagnostics
and usage go only to stderr, never raw samples or secrets. Exit 0 is a normal stop
or finished self-test, 1 is capture/output/cleanup failure, 2 is invalid usage or
stdout type. A terminal stdout is rejected. Regular-file output is supported for
synthetic fixtures, but filesystem writes have no nonblocking latency guarantee;
use a pipe/socket for live transport. No shutdown guarantee is claimed for SIGKILL
or an OS permission dialog/API that has not returned. A silent/exited target is
not restarted or replaced; parent owns process lifetime and cancellation.

## Offline checks and evidence

Private evidence/scripts for this implementation live at:
`/Users/denniswillie/jarvis/.jarvis-dev/soloist/capture-helper-evidence-20261104/`.
Build/signing, sanitized packetizer tests, protocol parser, malformed CLI,
stalled-pipe stale drops, signal interruption and broken-pipe checks are recorded
there. Tests use synthetic samples only. Build or synthetic success does **not**
establish real Soloist capture, permission behavior, live mute restoration,
hardware-format acceptance, backend transport, phone playback or live teardown.
Those remain parent-owned live acceptance, not performed here.

Minimal independent parser check (does not print the sample payload):

```sh
HELPER=/Users/denniswillie/jarvis/.jarvis-dev/soloist/JarvisSoloistCapture.app/Contents/MacOS/JarvisSoloistCapture
python3 - "$HELPER" <<'PY'
import struct, subprocess, sys
p = subprocess.run([sys.argv[1], '--self-test'], capture_output=True, check=True)
assert not p.stderr
assert len(p.stdout) == 206
length, flags, pt, seq, timestamp, ssrc = struct.unpack('!HBBHII', p.stdout[:14])
assert (length, flags, pt, seq, timestamp, ssrc) == (204, 0x80, 0xe0, 0, 0, 0x534f4c4f)
assert struct.unpack('!96h', p.stdout[14:]) == (8192, -8192) * 48
print('PASS: one RFC4571/RTP PT96 L16/48000/2 packet, 48 stereo frames')
PY
```

Platform reference: [Apple: Capturing system audio with Core Audio taps](https://developer.apple.com/documentation/CoreAudio/capturing-system-audio-with-core-audio-taps).
