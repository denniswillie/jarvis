#ifndef JARVIS_ECHO_DSP_H
#define JARVIS_ECHO_DSP_H

// Recording worker only. Each call consumes 480 mono Float32 samples at 48 kHz.
// The caller aligns microphone and audible playback on the recording timeline.
void *JarvisEchoCreate(void);
void JarvisEchoDestroy(void *state);
void JarvisEchoReset(void *state);
void JarvisEchoProcess(void *state, const float *microphone,
                       const float *playback, float *clean);
#endif
