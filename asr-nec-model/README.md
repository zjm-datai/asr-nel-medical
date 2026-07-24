# ASR NEC Model

Project scaffold for the ASR named entity correction model.

## Layout

- `src/asr_nec_model/models`: model modules
- `src/asr_nec_model/data`: datasets and batch helpers
- `src/asr_nec_model/training`: training loops
- `src/asr_nec_model/inference`: retrieval and correction pipeline
- `src/asr_nec_model/audio`: feature extraction
- `src/asr_nec_model/utils`: shared helpers

Generated artifacts are intentionally outside this code repository. From this
repository, scripts resolve `../data`, `../weights`, `../runs`, and `../tmp` as
workspace-level directories.

For detached GPU training in Docker, see `DOCKER_TRAINING.md` and start the
Compose service with `docker compose up -d ss-train`.

For interactive GPU development on a server, open this repository in VS Code
and use `Dev Containers: Reopen in Container`. The configuration mounts the
workspace-level `data`, `weights`, `runs`, and `tmp` directories alongside the
code at `/workspace/asr-nec-model`.

## Quick start

```bash
uv sync
uv run python -m asr_nec_model
```

## TCM seed data

Build the expert-reviewable core lexicon and deterministic outpatient scripts:

```bash
uv run python scripts/build_tcm_entity_lexicon.py
uv run python scripts/build_tcm_scenarios.py
uv run python scripts/build_multidomain_ss_corpus.py
```

The generated text is cold-start material for recording or TTS. It is not an
ASR transcript and must not be labeled as a real ASR error. All entities and
clinical combinations remain `pending_expert_review` until reviewed by a TCM
professional.

Generate the complete TTS corpus with resumable per-file output. Compatible
pilot files are hard-linked or copied into the separate full-corpus directory:

```bash
python scripts/generate_tts_full.py --plan-only
python scripts/generate_tts_full.py --workers 3
```

The complete corpus is written under `../data/speech_searcher/audio_full` and
contains 3,600 utterance WAV files plus 690 entity-reference WAV files.
