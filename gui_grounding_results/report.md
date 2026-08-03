# GUI Grounding Benchmark — Open-Source Models on Ableton Live Screenshots

Models tested: Qwen2.5-VL-7B-Instruct, GTA1-7B, UI-TARS-1.5-7B

Environment note: run on Google Colab Pro, single L4 GPU (24GB VRAM), models loaded one at a time in bf16.

| Model | Screenshot | Instruction | Predicted (x, y) | Latency (s) | Raw output | Annotated image |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 02_browser-sounds-tab.png | Click the 'Sounds' item in the left sidebar under Library | not parsed | 3.79 | [10,265,134,287] | `gui_grounding_results/annotated/Qwen2.5-VL-7B-Instruct__task0__02_browser-sounds-tab.png` |
| Qwen2.5-VL-7B-Instruct | 02_browser-sounds-tab.png | Click the Solo (S) button on the '4 Audio' track | (1580, 923) | 2.92 | (1580,923), (1607,923) | `gui_grounding_results/annotated/Qwen2.5-VL-7B-Instruct__task1__02_browser-sounds-tab.png` |
| Qwen2.5-VL-7B-Instruct | 02_browser-sounds-tab.png | Click the 'Edit' button in the Filters bar of the browser | (280, 157) | 2.25 | (280,157) | `gui_grounding_results/annotated/Qwen2.5-VL-7B-Instruct__task2__02_browser-sounds-tab.png` |
| Qwen2.5-VL-7B-Instruct | 02_edit-menu.png | Click 'Group' in the Edit menu | (105, 264) | 2.25 | [105,264] | `gui_grounding_results/annotated/Qwen2.5-VL-7B-Instruct__task3__02_edit-menu.png` |
| Qwen2.5-VL-7B-Instruct | 02_edit-menu.png | Click 'Freeze Track' in the Edit menu | (102, 345) | 2.26 | [102,345] | `gui_grounding_results/annotated/Qwen2.5-VL-7B-Instruct__task4__02_edit-menu.png` |
| Qwen2.5-VL-7B-Instruct | 02_edit-menu.png | Click 'Consolidate' in the Edit menu | (102, 348) | 2.24 | [102,348] | `gui_grounding_results/annotated/Qwen2.5-VL-7B-Instruct__task5__02_edit-menu.png` |
| GTA1-7B | 02_browser-sounds-tab.png | Click the 'Sounds' item in the left sidebar under Library | (158, 276) | 2.53 | (158,273) | `gui_grounding_results/annotated/GTA1-7B__task0__02_browser-sounds-tab.png` |
| GTA1-7B | 02_browser-sounds-tab.png | Click the Solo (S) button on the '4 Audio' track | (1850, 931) | 2.5 | (1846,920) | `gui_grounding_results/annotated/GTA1-7B__task1__02_browser-sounds-tab.png` |
| GTA1-7B | 02_browser-sounds-tab.png | Click the 'Edit' button in the Filters bar of the browser | (617, 164) | 2.43 | (616,162) | `gui_grounding_results/annotated/GTA1-7B__task2__02_browser-sounds-tab.png` |
| GTA1-7B | 02_edit-menu.png | Click 'Group' in the Edit menu | (168, 482) | 2.43 | (168,476) | `gui_grounding_results/annotated/GTA1-7B__task3__02_edit-menu.png` |
| GTA1-7B | 02_edit-menu.png | Click 'Freeze Track' in the Edit menu | (197, 892) | 2.42 | (197,882) | `gui_grounding_results/annotated/GTA1-7B__task4__02_edit-menu.png` |
| GTA1-7B | 02_edit-menu.png | Click 'Consolidate' in the Edit menu | (193, 1003) | 2.42 | (193,992) | `gui_grounding_results/annotated/GTA1-7B__task5__02_edit-menu.png` |
| UI-TARS-1.5-7B | 02_browser-sounds-tab.png | Click the 'Sounds' item in the left sidebar under Library | (158, 273) | 2.49 | (158,273) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task0__02_browser-sounds-tab.png` |
| UI-TARS-1.5-7B | 02_browser-sounds-tab.png | Click the Solo (S) button on the '4 Audio' track | (1846, 921) | 2.48 | (1846,921) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task1__02_browser-sounds-tab.png` |
| UI-TARS-1.5-7B | 02_browser-sounds-tab.png | Click the 'Edit' button in the Filters bar of the browser | (606, 163) | 2.42 | (606,163) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task2__02_browser-sounds-tab.png` |
| UI-TARS-1.5-7B | 02_edit-menu.png | Click 'Group' in the Edit menu | (185, 476) | 2.41 | (185,476) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task3__02_edit-menu.png` |
| UI-TARS-1.5-7B | 02_edit-menu.png | Click 'Freeze Track' in the Edit menu | (197, 882) | 2.42 | (197,882) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task4__02_edit-menu.png` |
| UI-TARS-1.5-7B | 02_edit-menu.png | Click 'Consolidate' in the Edit menu | (195, 991) | 2.44 | (195,991) | `gui_grounding_results/annotated/UI-TARS-1.5-7B__task5__02_edit-menu.png` |

## How to read this
- **Predicted (x, y)** is in the original screenshot's pixel coordinates.
- Open the annotated PNGs in `annotated/` to see the predicted click point drawn directly on the screenshot — this is far more informative than the raw numbers, since a model can be numerically 'close' but visually wrong (e.g. clicking the wrong menu item at a similar height).
- 'not parsed' means the regex coordinate parser didn't find a recognizable (x,y) pattern in that model's output — check `raw_results.json` for the full text and adjust `COORD_PATTERNS` if needed.
