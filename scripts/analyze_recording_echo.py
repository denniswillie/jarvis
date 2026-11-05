"""Read-only timing analysis of Jarvis's opt-in four-channel diagnostic WAV.

Reports correlations, not an automatic perceptual-quality verdict. The reference
is the current filter input, not an independently recorded hardware output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np


def lag_measure(mic: np.ndarray, reference: np.ndarray, cleaned: np.ndarray, rate: int) -> dict:
    # Positive lag means the microphone contains a later copy of the reference.
    n = len(mic)
    size = 1 << (2 * n - 1).bit_length()
    cross = np.fft.irfft(np.fft.rfft(mic, size) * np.fft.rfft(reference[::-1], size), size)[:2 * n - 1]
    lags = np.arange(-min(n // 2, rate), min(n // 2, rate) + 1)
    m0, r0 = np.maximum(lags, 0), np.maximum(-lags, 0)
    count = n - np.abs(lags)
    me = np.concatenate(([0.0], np.cumsum(mic * mic)))
    re = np.concatenate(([0.0], np.cumsum(reference * reference)))
    norm = np.sqrt((me[m0 + count] - me[m0]) * (re[r0 + count] - re[r0]))
    correlation = cross[n - 1 + lags] / np.maximum(norm, 1e-12)
    best = int(np.argmax(np.abs(correlation)))
    lag = int(lags[best])
    m, r, length = int(m0[best]), int(r0[best]), int(count[best])
    ref = reference[r:r + length]
    before = float(np.dot(mic[m:m + length], ref))
    after = float(np.dot(cleaned[m:m + length], ref))
    return {
        "microphone_lag_ms": round(lag * 1000 / rate, 2),
        "reference_correlation": round(float(correlation[best]), 4),
        "correlated_component_reduction_db": round(20 * np.log10(max(abs(before), 1e-12) / max(abs(after), 1e-12)), 2),
        "weak_evidence": bool(abs(correlation[best]) < 0.25),
    }


def analyze(samples: np.ndarray) -> dict:
    rate = 3000
    # Equal block-average decimation on all channels. Diagnostic only, not DSP
    # used by the app. Difference removes DC/slow gain offsets before matching.
    count = len(samples) // 16 * 16
    reduced = samples[:count].reshape(-1, 16, 4).mean(axis=1).astype(np.float64)
    reduced = np.diff(reduced, axis=0)
    windows = []
    for start in range(0, len(reduced) - 4 * rate + 1, 2 * rate):
        block = reduced[start:start + 4 * rate]
        windows.append({"start_seconds": start / rate,
                        **lag_measure(block[:, 0], block[:, 1], block[:, 2], rate)})
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2, axis=0)) if len(samples) else np.zeros(4)
    return {"seconds": len(samples) / 48000,
            "channel_rms": dict(zip(["microphone", "filter_reference", "cleaned_microphone", "digital_mix"], map(float, rms))),
            "windows": windows,
            "caveat": "Speech and periodic music can confuse correlation. Weak matches or reductions are not proof of echo removal."}


def read_probe(path: Path) -> np.ndarray:
    if not 44 <= path.stat().st_size <= 44 + 20 * 48000 * 16:
        raise ValueError("Not a bounded Jarvis diagnostic file")
    data = path.read_bytes()
    if (data[:4], data[8:16], data[36:40]) != (b"RIFF", b"WAVEfmt ", b"data"):
        raise ValueError("Expected the Jarvis diagnostic WAV header")
    if struct.unpack_from("<HHIIHH", data, 20) != (3, 4, 48000, 768000, 16, 32):
        raise ValueError("Expected four-channel Float32 at 48 kHz")
    if struct.unpack_from("<I", data, 40)[0] != len(data) - 44 or (len(data) - 44) % 16:
        raise ValueError("Incomplete diagnostic; wait until recording has stopped")
    samples = np.frombuffer(data, dtype="<f4", offset=44).reshape(-1, 4)
    if not np.isfinite(samples).all():
        raise ValueError("Diagnostic contains non-finite samples")
    return samples


def self_test() -> None:
    rng = np.random.default_rng(47)
    reference = rng.normal(0, 0.1, 4 * 3000)
    for delay in [-450, 0, 660]:
        mic = np.zeros_like(reference)
        if delay >= 0:
            mic[delay:] = 0.4 * reference[:len(reference) - delay]
        else:
            mic[:delay] = 0.4 * reference[-delay:]
        result = lag_measure(mic, reference, mic * 0.1, 3000)
        assert abs(result["microphone_lag_ms"] - delay / 3) < 0.01, result
        assert result["reference_correlation"] > 0.99, result
        assert abs(result["correlated_component_reduction_db"] - 20) < 0.01, result
    print("PASS: known positive/negative/zero delay and attenuation; no physical-audio conclusion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.wav:
        print(json.dumps(analyze(read_probe(args.wav)), indent=2))
    else:
        parser.error("provide a diagnostic WAV or --self-test")
