# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VDA (Verified Data Attribution) — research implementation for identifying root-cause errors in multi-agent task execution. Uses LLM-based discriminators and optimization algorithms, based on an implementation note (Sections 6–7).

**Current status:** Stage 1a (discriminator probability generation) is implemented. Stages 2+ (OMD calibration, Gauss-Seidel blame assignment) are defined in config but not yet coded.

## Commands

```bash
# Install
pip install -r requirements.txt

# Tests (26 tests across 4 files)
pytest tests/
pytest tests/test_discriminator.py          # single file
pytest tests/test_discriminator.py::test_extract_probability_a_basic -v  # single test

# Generate Stage 1a reports (requires API credentials)
python scripts/generate_reports.py --subset Hand-Crafted --limit 5
python scripts/generate_reports.py --subset Algorithm-Generated --resume

# Inspect generated .npz reports
python scripts/inspect_reports.py --subset Hand-Crafted
```

## Architecture

### Pipeline (Stage-based)

1. **Dataset** (`vda_datasets/who_and_when.py`) — loads HuggingFace Kevin355/Who_and_When traces, filters to action-only steps (WebSurfer, Assistant, FileSurfer, ComputerTerminal), converts to `TraceStep` objects
2. **Prompt** (`vda/prompt.py`) — builds binary classification prompt: "Is step t the single root-cause mistake? (A) Yes / (B) No"
3. **Discriminator** (`vda/discriminator.py`) — sends prompt to LLM, extracts P('A') from logprobs. Backends: `OpenAIDiscriminator` (Chat Completions API) and `VertexAIDiscriminator` (Gemini via Vertex AI). `build_ensemble()` creates K discriminators from config.
4. **Reports** (`scripts/generate_reports.py`) — runs K×T queries per trace, saves `theta_hat` matrix as `.npz` to `data/reports/<subset>/`
5. **Analysis** (`explore_reports.ipynb`) — heatmaps, accuracy metrics, lift analysis

### Key design decisions

- **Logprob extraction** (`extract_probability_a`): normalizes token variants (" A", "(A)", "a" → "A"), falls back to P=0.5 if neither A nor B found. `max_tokens=1` forces single-token response.
- **Action-only filtering**: `WhoAndWhenTrace` stores both full history and action-only steps; discriminators see full context but theta_hat is indexed by action steps only. Ground truth `mistake_step` is remapped to the action-only index.
- **Multi-model ensemble**: `VDAConfig.discriminators` list supports mixed providers/models/temperatures. Empty list falls back to legacy single-OpenAI-model mode with K temperatures.
- **Config** (`config.py`): single `VDAConfig` dataclass with all hyperparameters matching Section 7 of the implementation note. OMD, Gauss-Seidel, and gradient params are defined here for future stages.

### Data flow

`Who&When HF dataset → WhoAndWhenTrace → [TraceStep] → discriminator prompt → logprobs → theta_hat (K×T_action .npz) → (future: OMD calibration → blame assignment)`

### Output format

Reports saved as `.npz` with keys: `theta_hat`, `model_ids`, `fallback_counts`, `mistake_agent`, `mistake_step`. A `manifest.json` tracks metadata per trace.

## Dependencies

Vertex AI packages (`google-cloud-aiplatform`, `vertexai`) are commented out in requirements.txt — install manually on GCP VMs only.
