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
- `select_diverse_screenshots.py` — added in session 2, see below. Picks a
  visually-diverse shortlist out of the user's full `Ableton_screenshots/`
  folder (40 files) for the next benchmark round.

---

## Session 2 findings (first full run, both bugs fixed)

The disk-full and GTA1 `min_pixels` crash from session 1 were both fixed
(see git history / the script itself — `delete_model_from_disk()` clears
each model's HF cache after use, and `_get_min_max_pixels()` handles the
newer `transformers` API where min/max pixels moved off the image
processor's direct attributes and into `.size["shortest_edge"/"longest_edge"]`).
Rerun completed cleanly: 18/18 (3 models × 6 tasks), no errors, no "not
parsed" rows for GTA1 or UI-TARS.

**Visually inspected all 18 annotated crosshairs against the source
screenshots** (not just the raw coordinates — see `context.md`'s original
"Quality" guidance). Result:

- **GTA1-7B: 6/6 visually correct.**
- **UI-TARS-1.5-7B: 6/6 visually correct**, and its coordinates land within
  ~10-15px of GTA1's on nearly every task — the two specialists agree with
  each other closely, which is a reassuring cross-check given there's no
  labeled ground truth.
- **Qwen2.5-VL-7B-Instruct (the control): 0/6 visually correct.** Not just
  "close but off" — three distinct failure modes: (1) returned a bounding
  box instead of a point on task 0, so the parser correctly logged "not
  parsed"; (2) missed the target element entirely on tasks 1-3 (e.g.
  predicted a point ~270px away from the actual Solo button, landing in an
  unrelated part of the mixer); (3) most notably, on tasks 4 and 5 ("Freeze
  Track" vs "Consolidate," two different Edit-menu items ~100px apart) it
  returned nearly *identical* coordinates for both — it doesn't appear to be
  reading the specific instruction on those two, just returning "somewhere
  in the lower Edit menu."

This is consistent with the hypothesis going in (Qwen is the non-specialized
baseline, expected to be weakest) but the *degree* of the gap — a clean 0/6
vs two 6/6s on this small sample — was more stark than expected. Caveat
repeated from session 1: six tasks on two screenshots is a vibe check, not
a statistically meaningful benchmark. Treat GTA1/UI-TARS's 6/6 as "looked
solid on this sample," not "solved."

The full breakdown (per-task verdicts + reasoning) is in
`gui_grounding_results/report.md` under "Visual quality assessment."

## Next session plan: expand testing, GTA1 vs UI-TARS only

User wants to drop the Qwen control (its weakness is now well-established on
this sample) and instead throw more/harder screenshots at **GTA1-7B and
UI-TARS-1.5-7B only**, since those two are the ones actually worth
differentiating between. User has ~32 more Ableton screenshots on disk
(`Ableton_screenshots/`, tree pasted into an earlier message — includes a
`menus/` subfolder with 8 more menu screenshots, so 40 files total) but
many are near-duplicates of each other (user's own estimate: "sometimes only
90% visually similar" — e.g. same panel with a different item highlighted).

To avoid burning Colab time on redundant layouts, `select_diverse_screenshots.py`
was written to shortlist a diverse subset automatically:

- Computes a 64-bit perceptual hash (dHash — gradient-based, sensitive to
  layout/content changes rather than overall brightness/color, so it's a
  reasonable fit for "different panel/tab/menu" vs "same panel, cursor
  moved") for every image, no GPU/ML needed, just Pillow.
- Clusters near-duplicates together (default threshold: >=90% similar,
  matching the user's own estimate of what counts as "basically the same")
  and keeps only the most "typical" representative of each cluster (the one
  with the smallest total distance to its cluster-mates, not just an
  arbitrary/alphabetical pick).
- If more distinct images survive deduping than the target shortlist size,
  runs greedy farthest-point sampling on the survivors to maximize visual
  spread across the final pick, rather than clustering in one area.
- Copies the shortlist into an output folder and writes a full CSV report
  (every image, its cluster, its similarity to the cluster representative,
  whether it made the final cut) so the user can eyeball/override the
  automatic choices before committing to a Colab run.
- **Tested** against a synthetic near-duplicate set built from this
  session's own annotated images (3 models × crosshair variants of the same
  2 base screenshots = 23 near-identical files): correctly collapsed all 23
  down to exactly the 2 true underlying screenshots. Dedup logic confirmed
  working; not yet run against the user's real 40-file `Ableton_screenshots/`
  folder (that folder lives on the user's machine, not in this environment).

**When the user comes back with a shortlist + new Colab run:**
1. Ask them to run `select_diverse_screenshots.py` locally against
   `Ableton_screenshots/` first (defaults: target 15 images, 90% dup
   threshold — both overridable via flags) and review `screenshot_diversity_report.csv`
   before uploading the shortlist to Colab, in case the automatic clustering
   grouped anything they'd actually want to keep separate (dHash is decent
   but not perfect — two visually-different-but-structurally-similar UI
   states, e.g. two different device racks with the same panel chrome,
   could conceivably land in the same cluster).
2. Update `gui_grounding_benchmark.py`'s `SCREENSHOTS_DIR`/`TASKS` to point
   at the new shortlist + a richer set of instructions per image (the
   current `TASKS` list only has 6 entries across 2 images — will need
   expanding to actually exercise the new screenshots).
3. Consider trimming `MODELS` down to just GTA1-7B and UI-TARS-1.5-7B per
   the user's request, which also frees up disk/time for more
   screenshots/instructions per Colab session instead of spending it on a
   control that's already shown to be weak.
4. Same evaluation approach as this session: open annotated PNGs, don't
   trust raw coordinates alone.

---

## Session 3 (cut short — hit token limit before finishing)

Session 3 started the handoff above but ran out of context budget partway
through. Status as of the cutoff:

- **`gui_grounding_benchmark.py` was already updated**: `MODELS` is trimmed
  to just `GTA1-7B` and `UI-TARS-1.5-7B` (Qwen control dropped, per plan).
  `TASKS` was deliberately **left unchanged** (still the old proven
  2-image/6-task set) rather than guessed from filenames — see the
  `# TODO(session 3)` comment right above `TASKS` in the script for the
  full list of 15 filenames still needing real instructions.
- **The user ran `select_diverse_screenshots.py` locally** against their
  real `Ableton_screenshots/` folder (not the synthetic test set) and got a
  15-image shortlist:
  ```
  01_browser-and-device-view-collapsed.png
  02_browser-sounds-tab.png
  04_browser-plugins-tab.png
  05_device-view-empty-midi-track.png
  05_view-menu.png
  06_navigate-menu.png
  07_device-view-instrument-rack-chains.png
  07_options-menu.png
  08_help-menu.png
  12_automation-lane-breakpoint-envelope.png
  14_locators-timeline-markers.png
  23_device-chain-3-stacked-devices.png
  26_return-track-send-knob.png
  29_settings-audio-tab.png
  32_save-copy-dialog.png
  ```
  This is a genuinely diverse-looking set (browser tabs, device views, 3
  more menus, automation, locators, a dialog) — the dedup step appears to
  have worked as intended on the real folder, though this wasn't
  independently re-verified (no direct visual check was done in this
  session, only the CSV report the user would have generated locally).
- **The user has pushed these 15 screenshots to GitHub** (the same repo
  used earlier in this project — `akbargherbal/gui_grounding_benchmark`)
  but explicitly asked NOT to start working on them yet in this session,
  just to record the handoff. **I have not cloned/viewed them.**
- The plan (unchanged from the "Next session plan" section above) was to
  view each of these 15 images directly before writing instructions —
  guessing from filenames risks wasting a Colab run on an instruction that
  doesn't match what's actually on screen. That viewing step is what got
  cut off.

### First things to do next session

1. `git clone` (or `git pull` if the repo's already local) the
   `akbargherbal/gui_grounding_benchmark` repo to pick up the newly-pushed
   screenshots — check whether the user put them in `screenshots/`,
   `shortlisted_screenshots/`, or somewhere else; the exact path wasn't
   confirmed before the session ended.
2. View all 15 images directly (not just filenames) and write 1-2 grounded
   instructions per image for the `TASKS` list, the same way the original
   2-image/6-task set was written — favor exact visible text
   ("Click 'X' in the Y menu") over vague/guessed element descriptions.
3. Fill in `gui_grounding_benchmark.py`'s `TASKS` (currently still the old
   placeholder set) and remove the `# TODO(session 3)` comment once done.
4. Hand back the finished script for the user to run in Colab, then repeat
   the same evaluation approach used in session 2: open every annotated
   PNG and visually verify GTA1 vs UI-TARS rather than trusting raw
   coordinates, and update `report.md` + this file with the findings.

---

## Session 4 (cut short again — hit an image-view limit, not a token limit)

Confirmed the 15 screenshots from the session-3 handoff are live in the repo
at `shortlisted_screenshots/` (not `screenshots/` — that older folder still
only has the original 2 test images). One of the 15,
`02_browser-sounds-tab.png`, is byte-identical (md5-checked) to the original
`screenshots/02_browser-sounds-tab.png`, confirming it's the same file
carried through, not a new capture.

**Discrepancy found and resolved:** the `gui_grounding_benchmark.py` pushed
to GitHub at the start of this session was missing the `# TODO(session 3)`
comment and still had `Qwen2.5-VL-7B-Instruct` in `MODELS` (i.e. it was an
older/incomplete copy). The user then uploaded the actual latest script
directly, which *did* have `MODELS` correctly trimmed to just `GTA1-7B` and
`UI-TARS-1.5-7B` and *did* have the TODO comment. That uploaded version was
used as the base for this session's edits and is what should be on GitHub
now (the user said they'd push it after this session). **If a future
session finds `MODELS` still includes Qwen or the TODO comment references
"session 3" instead of "session 4," the wrong/stale copy got pushed —
double check before trusting the repo's `gui_grounding_benchmark.py`.**

**Progress this session:**
- `SCREENSHOTS_DIR` updated from `./screenshots` to `./shortlisted_screenshots`.
- Directly viewed 10 of the 15 shortlisted images and wrote 2 grounded
  instructions each (20 `TASKS` entries total) for: `01_browser-and-device-view-collapsed`,
  `02_browser-sounds-tab`, `04_browser-plugins-tab`, `05_device-view-empty-midi-track`,
  `05_view-menu`, `06_navigate-menu`, `07_device-view-instrument-rack-chains`,
  `07_options-menu`, `08_help-menu`, `12_automation-lane-breakpoint-envelope`.
  Instructions favor exact visible text per the session-3 plan (e.g. "Click
  'Mixer' in the View menu", "Click the Solo (S) button on track '5 MIDI'").
- **Hit a hard per-conversation image-view limit (10 successful `view` calls)
  partway through.** The remaining 5 images —
  `14_locators-timeline-markers.png`, `23_device-chain-3-stacked-devices.png`,
  `26_return-track-send-knob.png`, `29_settings-audio-tab.png`,
  `32_save-copy-dialog.png` — could not be viewed for the rest of the
  session. Confirmed this was a session-wide cap, not a problem with those
  specific files: re-tried one of them after resizing it smaller and after
  converting to a fresh PNG copy, both attempts returned empty; a file
  already viewed earlier in the same session (`02_browser-sounds-tab.png`)
  also failed to re-render once the limit was hit. All 5 files themselves
  are confirmed valid (checked with `file`, correct PNG headers, normal
  1936x1048 dimensions).
- **Did not guess instructions for those 5 from filenames** — consistent
  with the caution already noted in session 3's handoff. A
  `# TODO(session 4)` comment in `gui_grounding_benchmark.py` marks exactly
  this, right above `TASKS`.
- The user is pushing this session's updated `gui_grounding_benchmark.py` to
  GitHub and starting a fresh chat next, specifically so the next session
  gets a full image-view budget to finish the remaining 5.

### First things to do next session

1. `git pull`/`git clone` to confirm the script pushed at the end of this
   session actually landed (see the discrepancy warning above — verify
   `MODELS` has only GTA1-7B/UI-TARS-1.5-7B and the TODO comment says
   "session 4" before trusting it; if it looks like the *session-3* version
   again, ask the user to re-upload the file directly rather than trusting
   the repo).
2. View the 5 remaining images directly: `14_locators-timeline-markers.png`,
   `23_device-chain-3-stacked-devices.png`, `26_return-track-send-knob.png`,
   `29_settings-audio-tab.png`, `32_save-copy-dialog.png`. Write 1-2 grounded
   instructions per image, same style as the 20 already in `TASKS`
   (exact visible text, no guessing from filenames).
3. Remove the `# TODO(session 4)` comment once all 5 are filled in — at that
   point all 15 images / ~30 tasks are ready for a Colab run.
4. Hand back the finished script, then repeat the session-2 evaluation
   approach: open every annotated PNG and visually verify GTA1 vs UI-TARS
   rather than trusting raw coordinates, and update `report.md` + this file
   with the findings.
5. If the image-view limit is hit again partway through step 2, don't burn
   turns retrying blindly — either ask the user to paste the remaining
   images directly as message attachments (a different path than the `view`
   tool, not subject to the same cap) or note precisely which images are
   still unviewed and hand off cleanly the same way this session did.

---

## Session 5 (finished the TASKS list — all 15 images, 30 tasks)

Confirmed on clone that the session-4 script had landed correctly on GitHub
(no discrepancy this time): `MODELS` had only GTA1-7B/UI-TARS-1.5-7B, and
the `# TODO(session 4)` comment was present exactly as expected, listing
the same 5 unviewed filenames.

Viewed all 5 remaining images directly (no image-view limit hit this
session) and added 2 grounded instructions each: `14_locators-timeline-markers`,
`23_device-chain-3-stacked-devices`, `26_return-track-send-knob`,
`29_settings-audio-tab`, `32_save-copy-dialog`. Removed the TODO comment.
`TASKS` now has all 30 entries (2 per image × 15 images) and the script
parses cleanly (`ast.parse` + a task/image count check, both passed).

**Not yet done / caveat for next session:** the 10 new instructions were
written from direct viewing but favor structurally-obvious elements (e.g.
"the loop brace," "the middle device's title bar," "the Send knob") over
guessing exact custom text where the UI showed user-editable/ambiguous
labels (e.g. custom locator names, custom chain names) — same caution as
prior sessions about not inventing text that might not exactly match. If a
Colab run shows these instructions map to the wrong element, that's the
first thing to revisit before blaming the model.

### First things to do next session

1. Hand the finished script to the user to run in Colab (all 15 images / 30
   tasks, GTA1-7B + UI-TARS-1.5-7B only — Qwen control already dropped).
2. Same evaluation approach as sessions 2 onward: open every annotated PNG
   and visually verify GTA1 vs UI-TARS rather than trusting raw
   coordinates.
3. Update `gui_grounding_results/report.md` and this file with the findings
   from the 30-task run — first full-scale comparison between the two
   specialists on this shortlist.
4. If any of the 10 new instructions from this session turn out to be
   ambiguous or mismatched (see caveat above), revise the wording in
   `TASKS` before re-running, rather than assuming the model was wrong.
