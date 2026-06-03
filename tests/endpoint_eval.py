"""Offline segmentation / noise evaluation harness.

The roadmap (docs/roadmap.md, Phase 1) names this a **hard predecessor** to any
endpointer tuning: "a few WAVs (monologue, two-person turn-taking, AC-only)
replayed through the trigger logic with pre/post counts is the regression guard
for every tuning item." None existed (`test_vad.py` only tests the VAD unit).

This harness replays audio through the REAL VAD path — per-frame Silero
probabilities and `SileroVAD.get_speech_segments()` (the clause-aware
segmenter) — and reports counts. It changes no behaviour; it only measures, so
later tuning can show before/after numbers against a fixed corpus.

Drop real WAVs into ``tests/seg_fixtures/`` (see its README); deterministic
synthetic clips (silence / tone / band-limited noise) are generated in-process
so the harness has something to measure in CI.

Run:  python tests/endpoint_eval.py [fixtures_dir]
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

# Allow `python tests/endpoint_eval.py` (script) as well as import from tests.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SAMPLE_RATE  # noqa: E402
from audio.vad import SileroVAD  # noqa: E402

FRAME = 512  # Silero frame at 16 kHz = 32 ms


# ── audio sources ──────────────────────────────────────────────────────────

def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def tone(seconds: float, hz: float = 440.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def noise(seconds: float, level: float = 0.03, seed: int = 0) -> np.ndarray:
    """Low-level broadband noise — a stand-in for AC/fan hum (deterministic)."""
    rng = np.random.default_rng(seed)
    return (level * rng.standard_normal(int(seconds * SAMPLE_RATE))).astype(np.float32)


def load_wav(path: str) -> np.ndarray:
    """Read a WAV as float32 mono at 16 kHz (resampling/mixing as needed)."""
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    data = np.asarray(data)
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    elif data.dtype.kind == "u":
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:  # stereo -> mono
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(int(sr), SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return data


# ── analysis ────────────────────────────────────────────────────────────────

def analyze(audio: np.ndarray, vad: SileroVAD | None = None) -> dict:
    """Replay `audio` through the VAD and return measurement metrics.

    Metrics:
      duration_s, frames, speech_frames, speech_fraction, max_prob, mean_prob,
      segments (count via get_speech_segments), segment_seconds (total),
      vad_loaded.
    """
    vad = vad or SileroVAD()
    audio = np.asarray(audio, dtype=np.float32)

    vad.reset()
    probs: list[float] = []
    for off in range(0, len(audio) - FRAME + 1, FRAME):
        probs.append(vad.speech_probability(audio[off:off + FRAME]))
    probs_arr = np.asarray(probs, dtype=np.float32)

    speech = int((probs_arr >= vad.threshold).sum()) if probs else 0
    segs = vad.get_speech_segments(audio)
    seg_samples = sum(e - s for s, e in segs)

    return {
        "duration_s": round(len(audio) / SAMPLE_RATE, 2),
        "frames": len(probs),
        "speech_frames": speech,
        "speech_fraction": round(speech / len(probs), 3) if probs else 0.0,
        "max_prob": round(float(probs_arr.max()), 3) if probs else 0.0,
        "mean_prob": round(float(probs_arr.mean()), 3) if probs else 0.0,
        "segments": len(segs),
        "segment_seconds": round(seg_samples / SAMPLE_RATE, 2),
        "vad_loaded": vad.is_loaded,
    }


def baseline_corpus() -> list[tuple[str, np.ndarray]]:
    """Deterministic synthetic clips so the harness always has inputs."""
    return [
        ("synth:silence/2s", silence(2.0)),
        ("synth:tone-440hz/2s", tone(2.0)),
        ("synth:noise-low/3s", noise(3.0)),
    ]


def run_corpus(fixtures_dir: str | None = None) -> dict[str, dict]:
    vad = SileroVAD()
    clips = list(baseline_corpus())
    if fixtures_dir and os.path.isdir(fixtures_dir):
        for wav in sorted(glob.glob(os.path.join(fixtures_dir, "*.wav"))):
            try:
                clips.append((os.path.basename(wav), load_wav(wav)))
            except Exception as exc:  # noqa: BLE001 - report, don't crash the run
                print(f"  !! skip {wav}: {exc}")
    results = {name: analyze(audio, vad) for name, audio in clips}
    return results


def _print_table(results: dict[str, dict]) -> None:
    cols = ("duration_s", "frames", "speech_frames", "speech_fraction",
            "max_prob", "segments", "segment_seconds")
    print(f"{'clip':28s} " + " ".join(f"{c:>14s}" for c in cols))
    for name, m in results.items():
        print(f"{name:28s} " + " ".join(f"{str(m[c]):>14s}" for c in cols))


def main() -> None:
    fixtures = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "seg_fixtures"
    )
    results = run_corpus(fixtures)
    if not next(iter(results.values()))["vad_loaded"]:
        print("WARNING: Silero VAD not loaded — metrics are fallbacks, not real.")
    _print_table(results)


if __name__ == "__main__":
    main()
