# Context: GUI Grounding Benchmark on Ableton Screenshots

For future-me picking this back up after the user runs `gui_grounding_benchmark.py`
in Google Colab Pro (L4 GPU) and comes back with results.

## What this is

User has Colab Pro credit expiring soon and wants a quick, hands-on look at how
open-source GUI-grounding VLMs currently perform, testing on their own Ableton
Live 12 screenshots (browser panel, Edit menu, etc. — see
`Ableton_screenshots/` tree the user pasted earlier: 40 files, including
menu screenshots). Source doc: `GUI_Grounding_Updated.md`, an executive
summary of the GUI-grounding research landscape (as of "August 2026" — note
some named models/benchmarks in that doc, e.g. Claude Opus 4.8 / GPT-5.4 /
Gemini 3.1 Pro / OSWorld-Verified numbers, could not be independently verified
against my training data and should be treated as the user's reference
material, not confirmed fact).

This is a **hands-on thought experiment**, not a rigorous benchmark: no
labeled ground-truth coordinates exist for the test screenshots. "Correct"
is judged visually (does the drawn crosshair land on the right UI element?),
not against a scored dataset.

## Models under test (all chosen to fit a single 24GB L4 in bf16)

| Model | Repo ID | Why included |
|---|---|---|
| Qwen2.5-VL-7B-Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` | General-purpose VLM baseline, NOT grounding-specialized — expect this to be the weakest, useful as a control |
| GTA1-7B | `HelloKKMe/GTA1-7B` | Salesforce, GRPO-trained grounding specialist; per its model card, currently the strongest fully-open 7B on ScreenSpot-Pro (~55.5%) |
| UI-TARS-1.5-7B | `ByteDance-Seed/UI-TARS-1.5-7B` | ByteDance's native GUI/computer-use agent (61.6% ScreenSpot-Pro per its own card — highest of the three) |

Deliberately excluded: `xlangai/OpenCUA-7B` (custom RoPE/tokenizer, more
fragile to load generically) and anything 32B+ (won't reliably fit 24GB).
I verified all three repo IDs and their loading code against the live HF
model cards before writing the script — GTA1's predict function is copied
almost verbatim from its model card's official inference snippet, including
the smart-resize coordinate rescaling.

## Where the outputs land (after a successful Colab run)

```
gui_grounding_results/
├── report.md          <- markdown table: model | image | instruction | predicted (x,y) | latency_sec | raw output | annotated image path
├── raw_results.json    <- one dict per (model, task), full raw_text + xy + latency_sec + any error
└── annotated/*.png     <- each screenshot with a red crosshair at the predicted click point
```

`raw_results.json` schema per row:
```json
{
  "model": "GTA1-7B",
  "image": "02_edit-menu.png",
  "instruction": "Click 'Group' in the Edit menu",
  "raw_text": "(412, 483)",
  "xy": [412.3, 483.1],
  "latency_sec": 1.8,
  "annotated_path": "gui_grounding_results/annotated/GTA1-7B__task3__02_edit-menu.png"
}
```
Rows can instead have just `"error": "..."` if that model/task failed —
check these first when debugging.

## When the user comes back to evaluate quality + speed

1. **Speed**: `latency_sec` is per-task generation time only (excludes model
   load time, which is printed separately during the run but not saved to
   JSON — ask the user to paste terminal output if load-time comparison
   matters). Compare `latency_sec` across models for the same task; note
   MAX_NEW_TOKENS was capped at 64, so latency differences mostly reflect
   raw decode speed, not reasoning-token overhead — none of these three
   models emit long chain-of-thought before the coordinate in the default
   prompt used here.
2. **Quality**: open the annotated PNGs, not just the numbers. Judge whether
   the crosshair is actually on the named element vs. numerically-plausible
   but visually wrong (e.g., right height, wrong menu column). Ask the user
   which points looked right/wrong per task if it's not obvious from the
   image alone — I can't re-view the screenshot pixel-diff myself without
   them re-uploading the annotated images.
3. Aggregate impression should note: Qwen2.5-VL-7B-Instruct is the control
   (expect weaker/less consistent grounding, possibly more "not parsed"
   rows since it wasn't trained to emit a strict `(x,y)` format); GTA1-7B
   and UI-TARS-1.5-7B are the actual grounding specialists to compare
   against each other.

## Common failure modes to check first if something broke

- **"not parsed" xy / regex misses**: each model phrases coordinates
  differently. Check `raw_text` in the JSON and extend `COORD_PATTERNS` in
  the script if a model's real output format wasn't one of the 4 patterns
  tried.
- **OOM on load or generate**: L4 has 24GB; each bf16 7B model is
  ~15-17GB. Should fit one at a time (script frees VRAM between models via
  `del model; gc.collect(); torch.cuda.empty_cache()`), but if the Colab
  instance actually handed out a T4 (16GB) or shared/lower-tier GPU instead
  of an L4, that will OOM — check the printed `print_gpu_info()` line at
  the top of the run.
- **flash-attn errors on GTA1**: the script already try/excepts this and
  falls back to default attention if `flash_attention_2` isn't installed —
  shouldn't be fatal, just slightly slower.
- **Missing `qwen_vl_utils`**: only needed for GTA1's predict function
  (`process_vision_info`, `smart_resize`). If not installed, GTA1 will fail
  to load/predict specifically — reinstall with
  `pip install qwen-vl-utils` and re-run just that model if needed.
- **HF download stalls/rate-limited**: check whether `setup_hf_token()`
  actually picked up a token (it prints a confirmation line either way at
  the top of the run).

## Files referenced in this project

- `gui_grounding_benchmark.py` — the script itself (source of truth for
  exact prompts/parsing logic used per model)
- `GUI_Grounding_Updated.md` — user's source executive summary
- `screenshots/02_browser-sounds-tab.png`, `screenshots/02_edit-menu.png` —
  the two test images currently wired into `TASKS`
