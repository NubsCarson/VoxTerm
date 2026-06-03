"""Tests for the offline segmentation/noise eval harness (endpoint_eval.py).

These guard the harness itself (it measures and returns sane metrics) and the
one property that must hold regardless of tuning: pure silence yields zero
speech frames. Real-WAV corpus metrics are reported by the harness CLI, not
asserted here (the corpus is the maintainer's to populate).
"""

import glob
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import sibling harness
import endpoint_eval as ee  # noqa: E402
from audio.vad import SileroVAD  # noqa: E402

_VAD = SileroVAD()
_needs_vad = pytest.mark.skipif(not _VAD.is_loaded, reason="Silero VAD model not available")


def test_analyze_returns_expected_metrics():
    m = ee.analyze(ee.silence(1.0), _VAD)
    for key in ("duration_s", "frames", "speech_frames", "speech_fraction",
                "max_prob", "mean_prob", "segments", "segment_seconds", "vad_loaded"):
        assert key in m
    assert m["frames"] > 0
    assert 0.0 <= m["speech_fraction"] <= 1.0


@_needs_vad
def test_silence_has_no_speech_frames():
    # The invariant that must survive any future endpointer tuning.
    m = ee.analyze(ee.silence(2.0), _VAD)
    assert m["speech_frames"] == 0
    assert m["max_prob"] < _VAD.threshold


@_needs_vad
def test_noise_and_tone_are_measurable():
    # No hard threshold (steady-state noise tripping the VAD is exactly the
    # Phase-1 problem the harness exists to quantify) — just assert it measures.
    for clip in (ee.noise(2.0), ee.tone(2.0)):
        m = ee.analyze(clip, _VAD)
        assert m["frames"] > 0 and 0.0 <= m["speech_fraction"] <= 1.0


def test_run_corpus_covers_synthetic_baseline():
    results = ee.run_corpus(fixtures_dir=None)
    assert {"synth:silence/2s", "synth:tone-440hz/2s", "synth:noise-low/3s"} <= set(results)


def test_real_fixtures_if_present():
    fdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seg_fixtures")
    wavs = glob.glob(os.path.join(fdir, "*.wav"))
    if not wavs:
        pytest.skip("no real WAV fixtures dropped in tests/seg_fixtures/ yet")
    m = ee.analyze(ee.load_wav(wavs[0]), _VAD)
    assert m["frames"] > 0
