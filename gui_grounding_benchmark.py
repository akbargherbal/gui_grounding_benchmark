"""
GUI Grounding Shootout — Open-Source Models on Ableton Live Screenshots
=========================================================================

Run this in Google Colab Pro on an L4 GPU (24GB VRAM). It loads three
open-weight GUI-grounding models ONE AT A TIME (so they never have to
share VRAM), asks each one to point at a UI element from a natural
language instruction, draws the predicted point on the screenshot, and
writes a markdown report comparing all of them.

Models tested (all fit comfortably on a single L4 in bf16):
  1. Qwen/Qwen2.5-VL-7B-Instruct   -> general-purpose VLM baseline (not
                                       fine-tuned for grounding specifically)
  2. HelloKKMe/GTA1-7B             -> Salesforce's GRPO-trained grounding
                                       specialist (current strongest fully
                                       open 7B-class model on ScreenSpot-Pro)
  3. ByteDance-Seed/UI-TARS-1.5-7B -> ByteDance's native GUI/computer-use agent

NOT included (on purpose, to keep this script simple and robust):
  - xlangai/OpenCUA-7B needs a custom tokenizer/RoPE setup and its own
    inference script rather than plain `transformers` classes. Same idea,
    just more moving parts -- add it later once the 3 above are working.
  - Anything 32B+ (Qwen3-VL-32B, GTA1-32B, OpenCUA-32B) -- these do NOT
    reliably fit on a 24GB L4 without 4-bit quantization, and even then
    it's tight. Stick to 7B-class models on L4.

How to use in Colab
--------------------
1. Runtime -> Change runtime type -> GPU -> L4 (or whatever you're given;
   the script will tell you what it detects and warn if VRAM is tight).
2. In a Colab cell, run:
       !pip install -q -U transformers accelerate qwen-vl-utils pillow huggingface_hub
3. (Recommended) Add a Hugging Face token so the ~15-17GB-per-model downloads
   go faster: click the key icon in Colab's left sidebar -> "Secrets" -> add
   a secret named HF_TOKEN with a token from
   https://huggingface.co/settings/tokens. The script picks it up
   automatically. If you skip this, it'll just prompt you once at runtime
   (or you can hit Enter there to download anonymously, just slower).
4. Upload your screenshots into a folder (default: ./screenshots) --
   easiest way is the Colab file browser on the left, or:
       from google.colab import files
       files.upload()   # then move the files into ./screenshots
5. Edit the TASKS list below to point at your actual screenshot filenames
   and the instructions you want to test.
6. Run this whole file:
       !python gui_grounding_benchmark.py
   or paste it into cells and run top to bottom.
7. Results land in ./gui_grounding_results/ :
     - report.md              <- the comparison report
     - annotated/*.png         <- each screenshot with the model's
                                  predicted click point drawn on it
     - raw_results.json        <- every raw model output, for debugging

Notes for an "amateur Python dev"
----------------------------------
- Everything is wrapped in try/except per-model-per-task, so if one model
  crashes (OOM, missing package, weird output) the script keeps going and
  just records the error instead of dying halfway through.
- `del model; gc.collect(); torch.cuda.empty_cache()` between models is
  the important bit that lets 3 different 7B models run back-to-back on
  one L4 without ever loading more than one at a time.
- Coordinate parsing is regex-based and a bit fragile because every model
  formats its answer slightly differently. If a model's predicted point
  looks wrong, print `raw_text` for that row (it's saved in the JSON) and
  adjust the relevant `parse_*` function.
"""

import gc
import json
import os
import re
import time
import traceback
from pathlib import Path

import torch
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# HF_TOKEN -- optional but recommended: authenticated downloads from the Hub
# are noticeably faster (higher-throughput CDN, higher rate limits) than
# anonymous ones, which matters when you're pulling three separate 7B models
# (~15-17GB each) on a ticking Colab Pro clock.
#
# Get a token (read access is enough) at: https://huggingface.co/settings/tokens
#
# This tries, in order:
#   1. An HF_TOKEN already set as an environment variable
#   2. Colab's "Secrets" panel (the key icon on the left sidebar) under the
#      name HF_TOKEN -- the recommended way, since it's not typed in plaintext
#   3. An interactive, hidden prompt (getpass) as a last resort
#   4. Skips login entirely if you just hit Enter -- downloads still work,
#      just slower / subject to anonymous rate limits
# ---------------------------------------------------------------------------


def setup_hf_token():
    token = os.environ.get("HF_TOKEN")

    if not token:
        try:
            from google.colab import userdata  # only exists inside Colab

            token = userdata.get("HF_TOKEN")
        except Exception:
            pass

    if not token:
        try:
            from getpass import getpass

            token = (
                getpass(
                    "Enter your Hugging Face token (from https://huggingface.co/settings/tokens), "
                    "or just press Enter to skip and download anonymously (slower):\n"
                ).strip()
                or None
            )
        except Exception:
            token = None

    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        os.environ["HF_TOKEN"] = token
        print("HF_TOKEN set -- downloads will use your authenticated rate limit.")
    else:
        print(
            "No HF_TOKEN provided -- continuing with anonymous downloads (may be slower)."
        )


# ---------------------------------------------------------------------------
# 0. Config -- EDIT THIS SECTION for your own screenshots / instructions
# ---------------------------------------------------------------------------

SCREENSHOTS_DIR = Path("./shortlisted_screenshots")  # 15-image diverse shortlist
OUTPUT_DIR = Path("./gui_grounding_results")
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
MAX_NEW_TOKENS = 64

# Each task = one screenshot + one natural-language grounding instruction.
# Add/remove/edit freely. Filenames must exist inside SCREENSHOTS_DIR.
# TODO(session 4): 5 of the 15 shortlisted images still need instructions --
# 14_locators-timeline-markers, 23_device-chain-3-stacked-devices,
# 26_return-track-send-knob, 29_settings-audio-tab, 32_save-copy-dialog.
# These were NOT viewed directly yet (an image-rendering limit was hit
# mid-session after 10 successful views) -- do not guess instructions for
# them from filenames alone, view each one first the same way the 10 below
# were done.
TASKS = [
    # -- 01_browser-and-device-view-collapsed.png --
    {
        "image": "01_browser-and-device-view-collapsed.png",
        "instruction": "Click the Solo (S) button on track '4 Audio'",
    },
    {
        "image": "01_browser-and-device-view-collapsed.png",
        "instruction": "Click the 'Cue Out' dropdown in the Main return track strip",
    },
    # -- 02_browser-sounds-tab.png --
    {
        "image": "02_browser-sounds-tab.png",
        "instruction": "Click the 'Sounds' item in the left sidebar under Library",
    },
    {
        "image": "02_browser-sounds-tab.png",
        "instruction": "Click '5ths Detuned Pad.adv' in the browser file list",
    },
    # -- 04_browser-plugins-tab.png --
    {
        "image": "04_browser-plugins-tab.png",
        "instruction": "Click 'Plug-Ins' in the Library section of the browser sidebar",
    },
    {
        "image": "04_browser-plugins-tab.png",
        "instruction": "Click 'Instruments' in the Library section of the browser sidebar",
    },
    # -- 05_device-view-empty-midi-track.png --
    {
        "image": "05_device-view-empty-midi-track.png",
        "instruction": "Click the Solo (S) button on track '5 MIDI'",
    },
    {
        "image": "05_device-view-empty-midi-track.png",
        "instruction": "Click the 'MIDI From' dropdown for track '5 MIDI'",
    },
    # -- 05_view-menu.png --
    {
        "image": "05_view-menu.png",
        "instruction": "Click 'Mixer' in the View menu",
    },
    {
        "image": "05_view-menu.png",
        "instruction": "Click 'Groove Pool' in the View menu",
    },
    # -- 06_navigate-menu.png --
    {
        "image": "06_navigate-menu.png",
        "instruction": "Click 'Device View' in the Navigate menu",
    },
    {
        "image": "06_navigate-menu.png",
        "instruction": "Click 'Help View' in the Navigate menu",
    },
    # -- 07_device-view-instrument-rack-chains.png --
    {
        "image": "07_device-view-instrument-rack-chains.png",
        "instruction": "Click the 'Crush' chain title in the instrument rack",
    },
    {
        "image": "07_device-view-instrument-rack-chains.png",
        "instruction": "Click the 'Hide' button in the instrument rack header",
    },
    # -- 07_options-menu.png --
    {
        "image": "07_options-menu.png",
        "instruction": "Click 'Computer MIDI Keyboard' in the Options menu",
    },
    {
        "image": "07_options-menu.png",
        "instruction": "Click 'Settings...' at the bottom of the Options menu",
    },
    # -- 08_help-menu.png --
    {
        "image": "08_help-menu.png",
        "instruction": "Click 'Load Demo Set' in the Help menu",
    },
    {
        "image": "08_help-menu.png",
        "instruction": "Click 'Check for Updates...' in the Help menu",
    },
    # -- 12_automation-lane-breakpoint-envelope.png --
    {
        "image": "12_automation-lane-breakpoint-envelope.png",
        "instruction": "Click the breakpoint node on the panning automation envelope in the '3 ui_survey' track",
    },
    {
        "image": "12_automation-lane-breakpoint-envelope.png",
        "instruction": "Click the 'Reverse' button in the clip's Sample panel",
    },
]

MODELS = [
    {"name": "GTA1-7B", "repo": "HelloKKMe/GTA1-7B", "kind": "gta1"},
    {
        "name": "UI-TARS-1.5-7B",
        "repo": "ByteDance-Seed/UI-TARS-1.5-7B",
        "kind": "generic_vlm",
    },
]

# ---------------------------------------------------------------------------
# 1. GPU sanity check
# ---------------------------------------------------------------------------


def print_gpu_info():
    if not torch.cuda.is_available():
        print("!! No GPU detected. This will be extremely slow or fail on 7B models.")
        return
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name} | Total VRAM: {total_gb:.1f} GB")
    if total_gb < 20:
        print(
            "!! Less than 20GB VRAM detected. 7B bf16 models (~15-17GB) may be tight. "
            "Consider load_in_4bit=True (see load_model()) if you hit OOM."
        )


def print_disk_info():
    """Colab's default disk is much smaller than it looks once 2-3 models'
    worth of ~15-30GB downloads pile up. Warn early instead of dying mid-download."""
    import shutil as _shutil

    total, used, free = _shutil.disk_usage("/")
    free_gb = free / 1e9
    print(f"Disk: {free_gb:.1f} GB free")
    if free_gb < 35:
        print(
            "!! Less than 35GB free. A single bf16 7B model can be 15-30GB on disk. "
            "This script now deletes each model's cache after use, but if this warning "
            "fires before ANY model has loaded, clear the HF cache first: "
            "`!rm -rf ~/.cache/huggingface/hub`"
        )


def free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def delete_model_from_disk(repo: str):
    """Remove a model's downloaded weights from the local Hugging Face cache
    (~/.cache/huggingface/hub) to free disk space -- this is the piece that
    was missing before. free_gpu_memory() only frees VRAM; it does nothing
    for the ~15-30GB that stays on disk after a model has loaded."""
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo_info in cache_info.repos:
            if repo_info.repo_id == repo:
                revisions = {rev.commit_hash for rev in repo_info.revisions}
                strategy = cache_info.delete_revisions(*revisions)
                print(
                    f"  Freeing {strategy.expected_freed_size_str} of disk "
                    f"(removing {repo} from local cache)"
                )
                strategy.execute()
                return
        print(f"  (no cached copy of {repo} found to delete)")
    except Exception as e:
        print(f"  !! could not clear disk cache for {repo}: {e}")


# ---------------------------------------------------------------------------
# 2. Coordinate parsing helpers
# ---------------------------------------------------------------------------
# Different models describe a click point in different formats. We try a
# handful of common patterns and return the first match.

COORD_PATTERNS = [
    re.compile(r"\((-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\)"),  # (123, 456)
    re.compile(
        r"click\s*\(\s*x\s*=\s*(-?\d*\.?\d+)\s*,\s*y\s*=\s*(-?\d*\.?\d+)", re.I
    ),  # click(x=1,y=2)
    re.compile(r"\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]"),  # [123, 456]
    re.compile(
        r"x\s*[:=]\s*(-?\d*\.?\d+).{0,20}?y\s*[:=]\s*(-?\d*\.?\d+)", re.I | re.S
    ),
]


def parse_xy(raw_text: str):
    """Best-effort extraction of an (x, y) pair from free-form model output."""
    for pattern in COORD_PATTERNS:
        m = pattern.search(raw_text)
        if m:
            try:
                return float(m.group(1)), float(m.group(2))
            except ValueError:
                continue
    return None


def annotate_image(image: Image.Image, xy, label: str) -> Image.Image:
    """Return a copy of image with a red crosshair drawn at xy."""
    img = image.convert("RGB").copy()
    if xy is None:
        return img
    x, y = xy
    draw = ImageDraw.Draw(img)
    r = 14
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=4)
    draw.line((x - r * 1.5, y, x + r * 1.5, y), fill=(255, 0, 0), width=3)
    draw.line((x, y - r * 1.5, x, y + r * 1.5), fill=(255, 0, 0), width=3)
    draw.text((x + r + 4, y - r), label, fill=(255, 0, 0))
    return img


# ---------------------------------------------------------------------------
# 3. Model loading + prediction, per "kind"
# ---------------------------------------------------------------------------


def load_model(repo: str, kind: str, use_4bit: bool = False):
    """Loads a model+processor pair. Returns (model, processor)."""
    from transformers import AutoProcessor

    quant_kwargs = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

    if kind == "gta1":
        # GTA1 uses the standard Qwen2.5-VL architecture + a specific
        # min/max pixel processor config recommended by the model card.
        from transformers import Qwen2_5_VLForConditionalGeneration

        try:
            attn_impl = "flash_attention_2"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                repo,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map="auto",
                **quant_kwargs,
            )
        except Exception:
            # flash-attn not installed -> fall back to default attention
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                repo,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                **quant_kwargs,
            )
        processor = AutoProcessor.from_pretrained(
            repo, min_pixels=3136, max_pixels=4096 * 2160
        )
        return model, processor

    # "generic_vlm": anything transformers can load via the standard
    # image-text-to-text interface (Qwen2.5-VL base, UI-TARS-1.5, etc.)
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        repo,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **quant_kwargs,
    )
    processor = AutoProcessor.from_pretrained(repo)
    return model, processor


def _get_min_max_pixels(image_processor):
    """Newer `transformers` releases stopped exposing min_pixels/max_pixels as
    direct attributes on Qwen2VLImageProcessor -- they now live nested under
    `.size["shortest_edge"/"longest_edge"]`. GTA1's model-card snippet assumes
    the old attribute-based API, which is what threw:
        'Qwen2VLImageProcessor' object has no attribute 'min_pixels'
    This checks both locations so it works either way, and falls back to
    GTA1's documented defaults (3136 / 4096*2160) as a last resort."""
    if hasattr(image_processor, "min_pixels") and hasattr(image_processor, "max_pixels"):
        return image_processor.min_pixels, image_processor.max_pixels
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict) and "shortest_edge" in size and "longest_edge" in size:
        return size["shortest_edge"], size["longest_edge"]
    return 3136, 4096 * 2160


def predict_generic_vlm(model, processor, image: Image.Image, instruction: str):
    """Ask a standard image-text-to-text model to output a click point."""
    prompt = (
        "You are controlling a desktop application. Look at the screenshot and the "
        "instruction below, then respond with ONLY the pixel coordinates of where to "
        f"click, in the exact format (x,y).\n\nInstruction: {instruction}"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw_text = processor.decode(new_tokens, skip_special_tokens=True)
    return raw_text


def predict_gta1(model, processor, image: Image.Image, instruction: str):
    """GTA1's own prompt format + coordinate rescaling (from the model card)."""
    from qwen_vl_utils import process_vision_info, smart_resize

    system_prompt = f"""
You are an expert UI element locator. Given a GUI image and a user's element
description, provide the coordinates of the specified element as a single
(x,y) point. The image resolution is height {{height}} and width {{width}}.
For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
""".strip()

    width, height = image.width, image.height
    min_pixels, max_pixels = _get_min_max_pixels(processor.image_processor)
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=processor.image_processor.patch_size
        * processor.image_processor.merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    resized_image = image.resize((resized_width, resized_height))
    scale_x, scale_y = width / resized_width, height / resized_height

    messages = [
        {
            "role": "system",
            "content": system_prompt.format(height=resized_height, width=resized_width),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": resized_image},
                {"type": "text", "text": instruction},
            ],
        },
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw_text = processor.decode(new_tokens, skip_special_tokens=True)

    xy = parse_xy(raw_text)
    if xy is not None:
        # GTA1's coordinates are in the *resized* image's space -> rescale
        # back to the original screenshot's pixel space.
        xy = (xy[0] * scale_x, xy[1] * scale_y)
    return raw_text, xy


PREDICT_FNS = {
    "generic_vlm": predict_generic_vlm,
    "gta1": None,  # handled specially below since it also returns rescaled xy
}


# ---------------------------------------------------------------------------
# 4. Main benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    ANNOTATED_DIR.mkdir(exist_ok=True, parents=True)
    setup_hf_token()
    print_gpu_info()
    print_disk_info()

    results = []  # list of dicts, one per (model, task)

    for model_cfg in MODELS:
        model_name, repo, kind = model_cfg["name"], model_cfg["repo"], model_cfg["kind"]
        print(f"\n=== Loading {model_name} ({repo}) ===")

        try:
            t0 = time.time()
            model, processor = load_model(repo, kind)
            print(f"Loaded in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"!! Failed to load {model_name}: {e}")
            traceback.print_exc()
            results.append({"model": model_name, "error": f"load failed: {e}"})
            free_gpu_memory()
            print(
                "  Clearing this model's cache in case the failure was from a "
                "corrupt/partial download (e.g. an earlier interrupted run)..."
            )
            delete_model_from_disk(repo)
            continue

        for i, task in enumerate(TASKS):
            image_path = SCREENSHOTS_DIR / task["image"]
            instruction = task["instruction"]
            row = {
                "model": model_name,
                "image": task["image"],
                "instruction": instruction,
            }

            if not image_path.exists():
                row["error"] = f"missing file: {image_path}"
                results.append(row)
                print(f"  [skip] {image_path} not found")
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                t0 = time.time()

                if kind == "gta1":
                    raw_text, xy = predict_gta1(model, processor, image, instruction)
                else:
                    raw_text = predict_generic_vlm(model, processor, image, instruction)
                    xy = parse_xy(raw_text)

                latency = time.time() - t0
                row.update(
                    {
                        "raw_text": raw_text.strip(),
                        "xy": xy,
                        "latency_sec": round(latency, 2),
                    }
                )

                annotated = annotate_image(image, xy, label=f"{model_name}")
                out_name = f"{model_name}__task{i}__{Path(task['image']).stem}.png"
                out_path = ANNOTATED_DIR / out_name
                annotated.save(out_path)
                row["annotated_path"] = str(out_path)

                print(f"  [{i}] '{instruction[:50]}...' -> xy={xy} ({latency:.1f}s)")

            except Exception as e:
                print(f"  !! task {i} failed: {e}")
                row["error"] = str(e)

            results.append(row)

        # Critical: free this model's VRAM before loading the next one...
        del model, processor
        free_gpu_memory()
        # ...AND its disk footprint, or three 15-30GB models will fill the
        # Colab disk before the third one even finishes downloading.
        delete_model_from_disk(repo)
        print_disk_info()

    with open(OUTPUT_DIR / "raw_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    write_report(results)
    print(f"\nDone. See {OUTPUT_DIR}/report.md")


# ---------------------------------------------------------------------------
# 5. Markdown report
# ---------------------------------------------------------------------------


def write_report(results):
    lines = [
        "# GUI Grounding Benchmark — Open-Source Models on Ableton Live Screenshots",
        "",
        f"Models tested: {', '.join(m['name'] for m in MODELS)}",
        "",
        "Environment note: run on Google Colab Pro, single L4 GPU (24GB VRAM), "
        "models loaded one at a time in bf16.",
        "",
        "| Model | Screenshot | Instruction | Predicted (x, y) | Latency (s) | Raw output | Annotated image |",
        "|---|---|---|---|---|---|---|",
    ]

    for row in results:
        if "error" in row and "raw_text" not in row:
            lines.append(
                f"| {row.get('model','?')} | {row.get('image','-')} | "
                f"{row.get('instruction','-')} | ERROR | - | {row['error']} | - |"
            )
            continue
        xy = row.get("xy")
        xy_str = f"({xy[0]:.0f}, {xy[1]:.0f})" if xy else "not parsed"
        raw_short = (row.get("raw_text", "") or "")[:80].replace("|", "/")
        img_link = row.get("annotated_path", "-")
        lines.append(
            f"| {row['model']} | {row['image']} | {row['instruction']} | "
            f"{xy_str} | {row.get('latency_sec','-')} | {raw_short} | `{img_link}` |"
        )

    lines += [
        "",
        "## How to read this",
        "- **Predicted (x, y)** is in the original screenshot's pixel coordinates.",
        "- Open the annotated PNGs in `annotated/` to see the predicted click point "
        "drawn directly on the screenshot — this is far more informative than the "
        "raw numbers, since a model can be numerically 'close' but visually wrong "
        "(e.g. clicking the wrong menu item at a similar height).",
        "- 'not parsed' means the regex coordinate parser didn't find a recognizable "
        "(x,y) pattern in that model's output — check `raw_results.json` for the "
        "full text and adjust `COORD_PATTERNS` if needed.",
        "",
    ]

    with open(OUTPUT_DIR / "report.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_benchmark()
