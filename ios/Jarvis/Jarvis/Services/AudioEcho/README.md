# Recording-only echo reduction

`JarvisEchoDSP.c` compiles the echo canceller and FFT from SpeexDSP 1.2.1.
The files in `vendor/` are unmodified upstream sources, with their BSD-style
license in `vendor/COPYING`. Upstream archive:
https://downloads.xiph.org/releases/speex/speexdsp-1.2.1.tar.gz

The recording worker processes aligned mono 48 kHz audio in 10 ms blocks,
with a 500 ms adaptive filter. It cleans only microphone samples, then adds
the selected clean digital Voice/music tracks. It does not mute user speech,
own hardware, change live transport, or process audio on the realtime callback.
Partial blocks keep their original timestamps and length. Silence without a
recent playback reference bypasses filtering. Discontinuities reset adaptation.

The current reference uses timestamped rendered Voice and consumed music with
actual gains, resampled by the existing recorder. Bluetooth transport delay and
speaker response are learned by the filter; physical performance is not implied
by synthetic tests. This is echo reduction, not a guarantee of perfect removal.

## Opt-in diagnostic

Debug launches with `JARVIS_RECORDING_ECHO_DIAGNOSTICS=1` save the first recording's
first 20 seconds to private `Documents/JarvisEchoDiagnostic/capture.wav`, with a
JSON description alongside it. No flag means no diagnostic file. An existing file
is never overwritten. Channels are raw filter microphone input, the current
filter reference, cleaned microphone and selected digital mix; the second channel
is **not** an independent hardware-output tap. The normal recording is unchanged.

After retrieval, `scripts/analyze_recording_echo.py capture.wav` reports candidate
delay and cancellation measurements (NumPy required). `--self-test` uses synthetic
signals only. Speech/periodic music may confuse correlation; inspect evidence
rather than using these metrics as an automatic acceptance test. Diagnostic audio
is private user data and must not be added to source control or uploaded elsewhere.
