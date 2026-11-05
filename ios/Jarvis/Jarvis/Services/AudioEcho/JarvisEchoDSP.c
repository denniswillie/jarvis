// Only the echo canceller and its FFT are compiled from unmodified SpeexDSP.
// See vendor/COPYING. No audio-session, device or playback ownership here.
#define FLOATING_POINT
#define USE_SMALLFT
#define EXPORT
#include "vendor/mdf.c"
#include "vendor/fftwrap.c"
#include "vendor/smallft.c"
#include "JarvisEchoDSP.h"

typedef struct {
    SpeexEchoState *echo;
    int referenceTail;
} JarvisEcho;

void *JarvisEchoCreate(void) {
    JarvisEcho *state = calloc(1, sizeof(*state));
    if (!state) return NULL;
    state->echo = speex_echo_state_init(480, 24000);
    if (!state->echo) { free(state); return NULL; }
    int rate = 48000;
    speex_echo_ctl(state->echo, SPEEX_ECHO_SET_SAMPLING_RATE, &rate);
    return state;
}

void JarvisEchoDestroy(void *pointer) {
    JarvisEcho *state = pointer;
    if (!state) return;
    speex_echo_state_destroy(state->echo);
    free(state);
}

void JarvisEchoReset(void *pointer) {
    JarvisEcho *state = pointer;
    if (!state) return;
    speex_echo_state_reset(state->echo);
    state->referenceTail = 0;
}

static spx_int16_t jarvisPCM(float sample) {
    if (!isfinite(sample)) return 0;
    return (spx_int16_t)lrintf(fmaxf(-32768.f, fminf(32767.f, sample * 32768.f)));
}

void JarvisEchoProcess(void *pointer, const float *microphone,
                       const float *playback, float *clean) {
    JarvisEcho *state = pointer;
    if (!state) { memcpy(clean, microphone, 480 * sizeof(float)); return; }
    spx_int16_t near[480], far[480], output[480];
    int audible = 0;
    for (int i = 0; i < 480; ++i) {
        near[i] = jarvisPCM(microphone[i]);
        far[i] = jarvisPCM(playback[i]);
        if (abs(far[i]) > 3) audible = 1;
    }
    if (audible) state->referenceTail = 50;
    else if (state->referenceTail > 0) --state->referenceTail;
    speex_echo_cancellation(state->echo, near, far, output);
    for (int i = 0; i < 480; ++i)
        clean[i] = state->referenceTail > 0 ? output[i] / 32768.f : microphone[i];
}
