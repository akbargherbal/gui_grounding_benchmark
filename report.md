# GUI Grounding Benchmark — Open-Source Models on Ableton Live Screenshots

Models tested: GTA1-7B, UI-TARS-1.5-7B

Environment note: run on Google Colab Pro, single L4 GPU (24GB VRAM), models loaded one at a time in bf16.

| Model | Screenshot | Instruction | Predicted (x, y) | Latency (s) | Raw output | Annotated image |
|---|---|---|---|---|---|---|
| GTA1-7B | 01_browser-and-device-view-collapsed.png | Click the Solo (S) button on track '4 Audio' | (385, 952) | 4.94 | (384,941) | `gui_grounding_results/annotated/GTA1-7B__task0__01_browser-and-device-view-collapsed.png` |
| GTA1-7B | 01_browser-and-device-view-collapsed.png | Click the 'Cue Out' dropdown in the Main return track strip | (1903, 727) | 2.46 | (1899,719) | `gui_grounding_results/annotated/GTA1-7B__task1__01_browser-and-device-view-collapsed.png` |
| GTA1-7B | 02_browser-sounds-tab.png | Click the 'Sounds' item in the left sidebar under Library | (158, 276) | 2.38 | (158,273) | `gui_grounding_results/annotated/GTA1-7B__task2__02_browser-sounds-tab.png` |
| GTA1-7B | 02_browser-sounds-tab.png | Click '5ths Detuned Pad.adv' in the browser file list | (476, 365) | 2.38 | (475,361) | `gui_grounding_results/annotated/GTA1-7B__task3__02_browser-sounds-tab.png` |
| GTA1-7B | 04_browser-plugins-tab.png | Click 'Plug-Ins' in the Library section of the browser sidebar | (154, 434) | 2.4 | (154,429) | `gui_grounding_results/annotated/GTA1-7B__task4__04_browser-plugins-tab.png` |
| GTA1-7B | 04_browser-plugins-tab.png | Click 'Instruments' in the Library section of the browser sidebar | (74, 322) | 2.35 | (74,318) | `gui_grounding_results/annotated/GTA1-7B__task5__04_browser-plugins-tab.png` |
| GTA1-7B | 05_device-view-empty-midi-track.png | Click the Solo (S) button on track '5 MIDI' | (1823, 698) | 2.46 | (1819,690) | `gui_grounding_results/annotated/GTA1-7B__task6__05_device-view-empty-midi-track.png` |
| GTA1-7B | 05_device-view-empty-midi-track.png | Click the 'MIDI From' dropdown for track '5 MIDI' | (1150, 416) | 2.47 | (1148,411) | `gui_grounding_results/annotated/GTA1-7B__task7__05_device-view-empty-midi-track.png` |
| GTA1-7B | 05_view-menu.png | Click 'Mixer' in the View menu | (381, 635) | 2.41 | (380,628) | `gui_grounding_results/annotated/GTA1-7B__task8__05_view-menu.png` |
| GTA1-7B | 05_view-menu.png | Click 'Groove Pool' in the View menu | (423, 328) | 2.41 | (422,324) | `gui_grounding_results/annotated/GTA1-7B__task9__05_view-menu.png` |
| GTA1-7B | 06_navigate-menu.png | Click 'Device View' in the Navigate menu | (422, 234) | 2.43 | (421,231) | `gui_grounding_results/annotated/GTA1-7B__task10__06_navigate-menu.png` |
| GTA1-7B | 06_navigate-menu.png | Click 'Help View' in the Navigate menu | (415, 335) | 2.43 | (414,331) | `gui_grounding_results/annotated/GTA1-7B__task11__06_navigate-menu.png` |
| GTA1-7B | 07_device-view-instrument-rack-chains.png | Click the 'Crush' chain title in the instrument rack | (425, 810) | 2.43 | (424,801) | `gui_grounding_results/annotated/GTA1-7B__task12__07_device-view-instrument-rack-chains.png` |
| GTA1-7B | 07_device-view-instrument-rack-chains.png | Click the 'Hide' button in the instrument rack header | (720, 812) | 2.46 | (719,803) | `gui_grounding_results/annotated/GTA1-7B__task13__07_device-view-instrument-rack-chains.png` |
| GTA1-7B | 07_options-menu.png | Click 'Computer MIDI Keyboard' in the Options menu | (683, 106) | 2.44 | (682,105) | `gui_grounding_results/annotated/GTA1-7B__task14__07_options-menu.png` |
| GTA1-7B | 07_options-menu.png | Click 'Settings...' at the bottom of the Options menu | (594, 1036) | 2.5 | (593,1024) | `gui_grounding_results/annotated/GTA1-7B__task15__07_options-menu.png` |
| GTA1-7B | 08_help-menu.png | Click 'Load Demo Set' in the Help menu | (635, 133) | 2.46 | (634,131) | `gui_grounding_results/annotated/GTA1-7B__task16__08_help-menu.png` |
| GTA1-7B | 08_help-menu.png | Click 'Check for Updates...' in the Help menu | (644, 354) | 2.46 | (643,350) | `gui_grounding_results/annotated/GTA1-7B__task17__08_help-menu.png` |
| GTA1-7B | 12_automation-lane-breakpoint-envelope.png | Click the breakpoint node on the panning automation envelope in the '3 ui_survey' track | (1644, 454) | 2.52 | (1641,449) | `gui_grounding_results/annotated/GTA1-7B__task18__12_automation-lane-breakpoint-envelope.png` |
| GTA1-7B | 12_automation-lane-breakpoint-envelope.png | Click the 'Reverse' button in the clip's Sample panel | (259, 959) | 2.45 | (258,948) | `gui_grounding_results/annotated/GTA1-7B__task19__12_automation-lane-breakpoint-envelope.png` |
| GTA1-7B | 14_locators-timeline-markers.png | Click the loop brace region below the timeline ruler | (28, 187) | 2.41 | (28,185) | `gui_grounding_results/annotated/GTA1-7B__task20__14_locators-timeline-markers.png` |
| GTA1-7B | 14_locators-timeline-markers.png | Click a locator marker flag on the timeline ruler | (846, 189) | 2.49 | (844,187) | `gui_grounding_results/annotated/GTA1-7B__task21__14_locators-timeline-markers.png` |
| GTA1-7B | 23_device-chain-3-stacked-devices.png | Click the second chain's title in the Chain List panel | (307, 457) | 2.47 | (306,452) | `gui_grounding_results/annotated/GTA1-7B__task22__23_device-chain-3-stacked-devices.png` |
| GTA1-7B | 23_device-chain-3-stacked-devices.png | Click the title bar of the middle device in the stacked chain | (821, 129) | 2.48 | (819,128) | `gui_grounding_results/annotated/GTA1-7B__task23__23_device-chain-3-stacked-devices.png` |
| GTA1-7B | 26_return-track-send-knob.png | Click the Send knob on the track strip that routes to the return | (1011, 661) | 2.55 | (1009,653) | `gui_grounding_results/annotated/GTA1-7B__task24__26_return-track-send-knob.png` |
| GTA1-7B | 26_return-track-send-knob.png | Click the return track's name at the top of its mixer strip | (910, 134) | 2.48 | (908,132) | `gui_grounding_results/annotated/GTA1-7B__task25__26_return-track-send-knob.png` |
| GTA1-7B | 29_settings-audio-tab.png | Click the 'Audio Output Device' dropdown in the Audio tab | (1215, 345) | 2.55 | (1212,341) | `gui_grounding_results/annotated/GTA1-7B__task26__29_settings-audio-tab.png` |
| GTA1-7B | 29_settings-audio-tab.png | Click the 'Input Config...' button in the Audio tab | (1155, 370) | 2.55 | (1153,366) | `gui_grounding_results/annotated/GTA1-7B__task27__29_settings-audio-tab.png` |
| GTA1-7B | 32_save-copy-dialog.png | Click the 'Save' button in the dialog | (786, 544) | 2.51 | (784,538) | `gui_grounding_results/annotated/GTA1-7B__task28__32_save-copy-dialog.png` |
| GTA1-7B | 32_save-copy-dialog.png | Click the 'Collect All and Save' checkbox | (935, 66) | 2.43 | (933,65) | `gui_grounding_results/annotated/GTA1-7B__task29__32_save-copy-dialog.png` |
| UI-TARS-1.5-7B | 01_browser-and-device-view-collapsed.png | Click the Solo (S) button on track '4 Audio' | (382, 941) | 2.56 | (382,941) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task0__01_browser-and-device-view-collapsed.png` |
| UI-TARS-1.5-7B | 01_browser-and-device-view-collapsed.png | Click the 'Cue Out' dropdown in the Main return track strip | (1870, 719) | 2.56 | (1870,719) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task1__01_browser-and-device-view-collapsed.png` |
| UI-TARS-1.5-7B | 02_browser-sounds-tab.png | Click the 'Sounds' item in the left sidebar under Library | (158, 273) | 2.48 | (158,273) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task2__02_browser-sounds-tab.png` |
| UI-TARS-1.5-7B | 02_browser-sounds-tab.png | Click '5ths Detuned Pad.adv' in the browser file list | (475, 361) | 2.5 | (475,361) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task3__02_browser-sounds-tab.png` |
| UI-TARS-1.5-7B | 04_browser-plugins-tab.png | Click 'Plug-Ins' in the Library section of the browser sidebar | (154, 428) | 2.5 | (154,428) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task4__04_browser-plugins-tab.png` |
| UI-TARS-1.5-7B | 04_browser-plugins-tab.png | Click 'Instruments' in the Library section of the browser sidebar | (158, 318) | 2.51 | (158,318) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task5__04_browser-plugins-tab.png` |
| UI-TARS-1.5-7B | 05_device-view-empty-midi-track.png | Click the Solo (S) button on track '5 MIDI' | (1731, 711) | 2.56 | (1731,711) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task6__05_device-view-empty-midi-track.png` |
| UI-TARS-1.5-7B | 05_device-view-empty-midi-track.png | Click the 'MIDI From' dropdown for track '5 MIDI' | (1150, 412) | 2.55 | (1150,412) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task7__05_device-view-empty-midi-track.png` |
| UI-TARS-1.5-7B | 05_view-menu.png | Click 'Mixer' in the View menu | (379, 628) | 2.48 | (379,628) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task8__05_view-menu.png` |
| UI-TARS-1.5-7B | 05_view-menu.png | Click 'Groove Pool' in the View menu | (422, 324) | 2.49 | (422,324) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task9__05_view-menu.png` |
| UI-TARS-1.5-7B | 06_navigate-menu.png | Click 'Device View' in the Navigate menu | (422, 231) | 2.47 | (422,231) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task10__06_navigate-menu.png` |
| UI-TARS-1.5-7B | 06_navigate-menu.png | Click 'Help View' in the Navigate menu | (414, 331) | 2.47 | (414,331) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task11__06_navigate-menu.png` |
| UI-TARS-1.5-7B | 07_device-view-instrument-rack-chains.png | Click the 'Crush' chain title in the instrument rack | (424, 801) | 2.49 | (424,801) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task12__07_device-view-instrument-rack-chains.png` |
| UI-TARS-1.5-7B | 07_device-view-instrument-rack-chains.png | Click the 'Hide' button in the instrument rack header | (719, 804) | 2.46 | (719,804) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task13__07_device-view-instrument-rack-chains.png` |
| UI-TARS-1.5-7B | 07_options-menu.png | Click 'Computer MIDI Keyboard' in the Options menu | (668, 105) | 2.47 | (668,105) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task14__07_options-menu.png` |
| UI-TARS-1.5-7B | 07_options-menu.png | Click 'Settings...' at the bottom of the Options menu | (593, 1024) | 2.51 | (593,1024) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task15__07_options-menu.png` |
| UI-TARS-1.5-7B | 08_help-menu.png | Click 'Load Demo Set' in the Help menu | (619, 131) | 2.45 | (619,131) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task16__08_help-menu.png` |
| UI-TARS-1.5-7B | 08_help-menu.png | Click 'Check for Updates...' in the Help menu | (628, 349) | 2.45 | (628,349) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task17__08_help-menu.png` |
| UI-TARS-1.5-7B | 12_automation-lane-breakpoint-envelope.png | Click the breakpoint node on the panning automation envelope in the '3 ui_survey' track | (1671, 449) | 2.52 | (1671,449) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task18__12_automation-lane-breakpoint-envelope.png` |
| UI-TARS-1.5-7B | 12_automation-lane-breakpoint-envelope.png | Click the 'Reverse' button in the clip's Sample panel | (258, 948) | 2.45 | (258,948) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task19__12_automation-lane-breakpoint-envelope.png` |
| UI-TARS-1.5-7B | 14_locators-timeline-markers.png | Click the loop brace region below the timeline ruler | (84, 171) | 2.38 | (84,171) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task20__14_locators-timeline-markers.png` |
| UI-TARS-1.5-7B | 14_locators-timeline-markers.png | Click a locator marker flag on the timeline ruler | (102, 160) | 2.45 | (102,160) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task21__14_locators-timeline-markers.png` |
| UI-TARS-1.5-7B | 23_device-chain-3-stacked-devices.png | Click the second chain's title in the Chain List panel | (788, 128) | 2.45 | (788,128) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task22__23_device-chain-3-stacked-devices.png` |
| UI-TARS-1.5-7B | 23_device-chain-3-stacked-devices.png | Click the title bar of the middle device in the stacked chain | (900, 412) | 2.45 | (900,412) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task23__23_device-chain-3-stacked-devices.png` |
| UI-TARS-1.5-7B | 26_return-track-send-knob.png | Click the Send knob on the track strip that routes to the return | (1042, 577) | 2.5 | (1042,577) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task24__26_return-track-send-knob.png` |
| UI-TARS-1.5-7B | 26_return-track-send-knob.png | Click the return track's name at the top of its mixer strip | (1037, 128) | 2.51 | (1037,128) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task25__26_return-track-send-knob.png` |
| UI-TARS-1.5-7B | 29_settings-audio-tab.png | Click the 'Audio Output Device' dropdown in the Audio tab | (1211, 340) | 2.52 | (1211,340) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task26__29_settings-audio-tab.png` |
| UI-TARS-1.5-7B | 29_settings-audio-tab.png | Click the 'Input Config...' button in the Audio tab | (1155, 366) | 2.52 | (1155,366) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task27__29_settings-audio-tab.png` |
| UI-TARS-1.5-7B | 32_save-copy-dialog.png | Click the 'Save' button in the dialog | (784, 537) | 2.45 | (784,537) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task28__32_save-copy-dialog.png` |
| UI-TARS-1.5-7B | 32_save-copy-dialog.png | Click the 'Collect All and Save' checkbox | (102, 1006) | 2.5 | (102,1006) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task29__32_save-copy-dialog.png` |

## How to read this
- **Predicted (x, y)** is in the original screenshot's pixel coordinates.
- Open the annotated PNGs in `annotated/` to see the predicted click point drawn directly on the screenshot — this is far more informative than the raw numbers, since a model can be numerically 'close' but visually wrong (e.g. clicking the wrong menu item at a similar height).
- 'not parsed' means the regex coordinate parser didn't find a recognizable (x,y) pattern in that model's output — check `raw_results.json` for the full text and adjust `COORD_PATTERNS` if needed.

## Visual quality assessment (session 6, 30-task run)

Both models parsed cleanly on all 30/30 tasks — no errors, no "not parsed" rows.
Load time: GTA1-7B 85.6s, UI-TARS-1.5-7B 251.4s (UI-TARS is ~3x slower to
download/load; per-task generation latency was similar for both, ~2.4-2.6s).

Rather than eyeballing every one of the 60 annotated images, the raw (x, y)
pairs were compared first: where the two models landed within ~15px of each
other, that's a strong signal both are probably right (previously confirmed
in session 2 for the easy/obvious elements — menus, sidebar labels, transport
buttons — and spot-checked again here). Where they diverged sharply, that's
the useful signal to actually open the PNGs. Six tasks showed large
disagreement and were visually inspected:

| Task | Instruction | GTA1-7B | UI-TARS-1.5-7B |
|---|---|---|---|
| 20 | Click the loop brace region below the timeline ruler | ✅ on the brace | 🟡 near it, possibly on the ruler itself rather than the brace |
| 21 | Click a locator marker flag on the timeline ruler | ✅ right on the flag icon | ❌ landed near the loop brace instead — wrong element entirely |
| 22 | Click the second chain's title in the Chain List panel | ✅ correctly in the left Chain List panel | ❌ landed on a device title bar on the right side — looks like it answered task 23 instead |
| 23 | Click the title bar of the middle device in the stacked chain | ✅ on a title bar row | ❌ landed mid-body of a device, not on the title bar |
| 24 | Click the Send knob on the track strip that routes to the return | ✅ precisely on the knob | 🟡 close, but possibly the wrong row/knob |
| 25 | Click the return track's name at the top of its mixer strip | ✅ plausible | 🟡 ~130px over, possibly one track off |
| 29 | Click the 'Collect All and Save' checkbox | ✅ dead-on | ❌ landed in the bottom-left corner, completely outside the dialog |

GTA1-7B was correct on every one of these harder, spatially-ambiguous tasks;
UI-TARS-1.5-7B missed clearly on at least 4 of the 7 (tasks 21, 22, 23, 29),
with two more (24, 25) uncertain-but-suspect. On the tasks where the raw
coordinates agreed closely (menus, browser sidebar, Solo buttons — the
"easy" structurally-unambiguous elements), both models were solid — spot
checked task 0 (Solo button) and task 8 (Mixer in View menu), both correct
for both models.

**Not every image was opened** — the sample above was deliberately chosen
from the tasks where raw-coordinate disagreement flagged a likely error,
which is a more efficient way to find real misses than opening all 60 PNGs
blind. A full 30/30 pass for both models has not been done; if a future
session wants that, it's the natural next step (see context.md).

### External validation (session 6)

This finding — GTA1-7B beating UI-TARS-1.5-7B specifically on tasks that
require distinguishing between structurally similar, spatially adjacent
elements, while the two are roughly tied on direct/obvious element naming
— is not just an artifact of this small sample. It closely replicates a
statistically-validated result from a 2026 robustness study,
**GUI-Perturbed: Domain Randomization Reveals Systematic Brittleness in GUI
Grounding Models** (arXiv:2604.14262), which evaluated this exact model
lineage (Qwen2.5-VL-7B → UI-TARS-1.5-7B → GTA1-7B, all sharing the same base
weights, differing only in post-training).

Key numbers from that paper:
- On the **aggregate ScreenSpot-Pro leaderboard**, UI-TARS-1.5-7B actually
  scores *higher* than GTA1-7B (61.6% vs. 55.5%) — so a naive read of
  published benchmarks would predict UI-TARS should win here, not lose.
- But on **"relational" grounding instructions** (identifying a target by
  its relationship to other elements, rather than direct naming — e.g. "the
  field above X"), GTA1-7B scores 65.8% vs. UI-TARS-1.5-7B's 35.0%, nearly
  double. The paper calls this the "white rectangle problem": models ground
  on visual primitives (shape, position, color) rather than function, so
  they can confuse two structurally similar UI elements — their example is
  a spreadsheet formula bar vs. a browser search bar.
- Mechanistically, GTA1-7B is UI-TARS-1.5-7B plus further RL training
  (GRPO) with a direct click-reward, rather than more supervised
  fine-tuning. The paper's finding is that this additional RL stage
  specifically *recovers* relational/spatial reasoning that the SFT/DPO
  training used for UI-TARS-1.5 actually *degrades* — UI-TARS-1.5 scores
  *worse* on relational accuracy (35.0%) than even the untrained
  Qwen2.5-VL-7B base it started from (45.0%).

This is a close match to what this session found by hand: GTA1 winning on
tasks requiring telling apart structurally similar, spatially close
elements (chain title vs. device title bar; locator flag vs. loop brace),
while the two are close on simple/obvious element naming. The caveat is
proportional: our n=30 informal check obviously isn't powered like
GUI-Perturbed's n=390 matched-pair statistical tests, and Ableton Live isn't
GUI-Perturbed's web-based Mind2Web domain — but the underlying mechanism
appears to be the same one, observed independently in a different domain.
