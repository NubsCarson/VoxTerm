# Segmentation eval fixtures

Drop short **16 kHz mono WAV** clips here; `tests/endpoint_eval.py` replays each
through the real VAD path and reports per-clip metrics (speech frames, speech
fraction, segment count/duration). They are reported, not asserted — this is the
regression corpus for Phase-1 endpointer tuning (see `docs/roadmap.md`).

Recommended clips (the roadmap's named cases):

| file | what it is | why |
|---|---|---|
| `monologue.wav` | ~30 s of one person speaking | mid-clause / mid-word split counts |
| `two_person.wav` | ~30 s of two-person turn-taking | turn boundaries |
| `ac_only.wav` | ~30 s of AC/fan noise, no speech | should yield ~0 speech (the "AC noise wrecks them" case) |

Anything not 16 kHz mono is resampled/downmixed on load. Files here are not
committed (kept out of git); the harness also generates deterministic synthetic
silence/tone/noise so it always has something to measure.

```sh
python tests/endpoint_eval.py            # synthetic baseline + any WAVs here
python tests/endpoint_eval.py /path/dir  # a different fixtures dir
```
