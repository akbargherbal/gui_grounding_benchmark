# I Pointed Open-Source Vision Models at My Ableton Screenshots. Here's What Six Sessions of "Just Click the Button" Taught Me.

*A hands-on, unscientific, occasionally-derailed benchmark of GUI-grounding VLMs — and how it accidentally reproduced a real research finding.*

---

## The premise

I had Google Colab Pro credit expiring soon and a question I couldn't answer by reading model cards: **how good are today's open-source GUI-grounding models, really, at looking at a screenshot and pointing at the right pixel?**

Not "what's their ScreenSpot-Pro score." I mean: if I hand a 7B vision-language model a screenshot of my actual DAW — Ableton Live 12, dense with knobs, menus, tiny icons, and overlapping panels — and tell it "click the Solo button on track 4," does the crosshair land where I'd expect a human to click?

There's no labeled ground-truth dataset for my own screenshots, so this was never going to be a rigorous benchmark. It's closer to a structured vibe check: run a few open-source grounding specialists against real screenshots, draw a crosshair on their predicted click point, and look at the picture. "Correct" means *the dot is on the thing*, judged by eye, not scored against a bounding box.

What I didn't expect was for a six-session, multi-week side project to end with me independently reproducing a specific, named failure mode from a research paper I hadn't read when I started.

## The setup

Three models, chosen to fit comfortably in bf16 on a single 24GB L4 GPU:

- **Qwen2.5-VL-7B-Instruct** — a general-purpose vision-language model, *not* trained specifically for GUI grounding. This was the control: the hypothesis going in was that it would be noticeably worse than the specialists.
- **GTA1-7B** (Salesforce) — a GRPO-trained grounding specialist, built by taking UI-TARS-1.5-7B and adding further reinforcement learning with a direct click-reward.
- **UI-TARS-1.5-7B** (ByteDance) — a native GUI/computer-use agent, trained with a large-scale supervised fine-tuning and DPO pipeline.

The workflow, once it stabilized: a Python script loads one model at a time (freeing VRAM between runs), feeds it a screenshot plus a natural-language instruction like *"Click 'Group' in the Edit menu,"* parses the predicted `(x, y)` coordinate out of the model's raw text output, and draws a red crosshair on the screenshot at that point. Everything — raw text, parsed coordinates, latency, and the annotated image — gets logged so the results can be inspected later without re-running anything.

Simple in concept. It took six sessions to get right, and most of those sessions weren't about the models at all.

## Session 1–2: the boring bugs that eat all your time

The first attempts didn't fail because a model was bad — they failed because the *infrastructure* was bad. A full run blew through Colab's disk allocation because model weights (15–17GB each, in bf16) were never being cleared from the Hugging Face cache between models. GTA1 also crashed on load because of an API drift in `transformers` — the newer image processor moved `min_pixels`/`max_pixels` off their old direct attributes and into a nested `.size["shortest_edge"/"longest_edge"]` dict, which broke the code copied from GTA1's official model card.

Neither bug had anything to do with grounding quality. Both had to be fixed before a single useful data point existed. This is the unglamorous truth of most "AI benchmark" projects: the majority of early effort goes into keeping the harness alive, not evaluating the thing you actually care about.

Once fixed, the first clean run — 3 models × 6 tasks across 2 screenshots — produced a result stark enough to end the debate about whether Qwen2.5-VL needed to stay in the lineup:

- **GTA1-7B: 6/6 visually correct.**
- **UI-TARS-1.5-7B: 6/6 visually correct**, landing within 10–15px of GTA1 on nearly every task — reassuring cross-agreement given there was no labeled ground truth to check against.
- **Qwen2.5-VL-7B-Instruct: 0/6.** Not narrowly wrong — three distinct failure modes in six tasks: it returned a bounding box instead of a point on one task (correctly flagged as "not parsed" by the regex parser); it missed the target element by ~270px on others; and on two different Edit-menu items about 100px apart ("Freeze Track" vs. "Consolidate"), it returned nearly identical coordinates for both, as if it wasn't reading the specific instruction at all, just aiming for "somewhere in the lower Edit menu."

Qwen2.5-VL is a fine general-purpose model. It was simply never trained to be a pixel-precise pointer, and the gap showed immediately. Verdict: drop the control, keep the two specialists, and go find harder screenshots.

## Session 3–5: the unglamorous work of not guessing

The next problem was more interesting than it sounds: I had roughly 40 Ableton screenshots on disk, many of them near-duplicates of each other — the same panel with a different item highlighted, "maybe 90% visually similar" by my own estimate. Throwing all 40 at a Colab GPU would waste time re-testing the same layout repeatedly.

So a companion script, `select_diverse_screenshots.py`, got written to do the curation automatically: compute a perceptual hash (dHash) for every image, cluster near-duplicates together, keep the most "typical" representative of each cluster, and if more distinct images survived than the target shortlist size, run greedy farthest-point sampling to maximize visual spread across the final picks. It writes a full CSV report so the selections can be sanity-checked, not just trusted blindly. Run once against a synthetic near-duplicate set (23 crosshair-annotated variants of 2 base images) to confirm it correctly collapsed them back down to 2 originals, then run for real against the 40-file folder, producing a 15-image shortlist: browser tabs, device views, three more menus, an automation lane, locator markers, and a save dialog.

Turning those 15 screenshots into good test instructions took two more sessions — partly because of a genuinely mundane obstacle: a per-conversation image-viewing limit got hit mid-session, and five screenshots simply couldn't be inspected until a fresh session started. The discipline that mattered here was refusing to guess. It would have been easy to infer an instruction from a filename like `26_return-track-send-knob.png` without actually looking at the image, but a mismatched instruction wastes an entire Colab run on the model trying to ground something that isn't precisely what's on screen. Every one of the 30 final instructions — two per image, phrased to reference exact visible text or a clearly identifiable element wherever possible — came from directly viewing the screenshot first.

## Session 6: the real run

With all 30 tasks locked in and the model list trimmed to just the two specialists, the actual Colab run was almost anticlimactic: 30/30 tasks parsed cleanly for both models, no errors, no unparseable outputs. GTA1-7B loaded in 85.6 seconds; UI-TARS-1.5-7B took 251.4 seconds — about three times slower to download and initialize, though per-task inference latency was nearly identical for both once loaded (roughly 2.4–2.6 seconds per click prediction).

Rather than opening all 60 annotated images one by one, I used the raw coordinates as a triage step: where GTA1 and UI-TARS predicted points within about 15 pixels of each other, that's a strong signal both are probably right — a pattern already confirmed by hand in session 2 for the "easy" elements like menu items and transport buttons. Where the two models' predictions diverged sharply, *that's* the signal worth opening the image for, because disagreement usually means someone missed.

Seven tasks showed sharp disagreement. Opening those specific annotated screenshots told a consistent story:

- **Click a locator marker flag on the timeline ruler.** GTA1 landed right on the flag icon. UI-TARS landed near the loop brace instead — a different, nearby, but functionally distinct UI element.
- **Click the second chain's title in the Chain List panel.** GTA1 correctly landed in the left-hand Chain List panel. UI-TARS landed on a device title bar on the *right* side of the screen — it looked, honestly, like it had answered a different task in the set.
- **Click the title bar of the middle device in the stacked chain.** GTA1 landed on a title bar row. UI-TARS landed mid-body of a device, not on the title bar at all.
- **Click the 'Collect All and Save' checkbox** in a save dialog. GTA1 landed dead-on the checkbox. UI-TARS's predicted point was in the *bottom-left corner of the entire screenshot* — nowhere near the dialog box.

GTA1-7B was correct on every one of these harder cases I checked. UI-TARS-1.5-7B missed clearly on at least four of the seven. On the easy, structurally unambiguous elements — menu items, sidebar labels, Solo buttons — both models were solid, matching the session 2 pattern.

That's a small, informal result. I wanted to know if it meant anything.

## Checking it against the literature

Here's where it got genuinely interesting. If you look up the two models' scores on **ScreenSpot-Pro** — the standard, professional-software-focused benchmark for this exact task — the published leaderboard says UI-TARS-1.5-7B actually *beats* GTA1-7B: 61.6% vs. 55.5%. By that number, my result should have gone the other way.

But a robustness study published in 2026, **GUI-Perturbed: Domain Randomization Reveals Systematic Brittleness in GUI Grounding Models** (arXiv:2604.14262), tested this *exact* model lineage — Qwen2.5-VL-7B, UI-TARS-1.5-7B, and GTA1-7B, all sharing the same base weights and differing only in how they were post-trained — and found something the aggregate leaderboard number doesn't show. When a grounding instruction requires understanding an element's *relationship* to other things on screen, rather than naming it directly, accuracy collapses for all three models — but by wildly different amounts. On these "relational" instructions, GTA1-7B scores 65.8%. UI-TARS-1.5-7B scores 35.0%. Nearly half.

The paper calls this the **"white rectangle problem"**: a model scoring 85%+ on standard benchmarks can still confuse a spreadsheet's formula bar with a browser's search bar, because both are white rectangles near the top of the screen. The models are grounding on shape, position, and color — visual primitives — rather than on what the element actually *does*. Standard benchmarks mostly ask "where is the Submit button," a direct-naming task these models have memorized well. They rarely ask "click the thing above the Submit button," which requires actually reasoning about layout.

The mechanistic explanation is the satisfying part. GTA1-7B isn't a from-scratch model — it's UI-TARS-1.5-7B *plus* additional reinforcement learning (GRPO) using a direct click-reward, rather than more supervised fine-tuning. The GUI-Perturbed authors found that this extra RL stage specifically *recovers* the relational/spatial reasoning that UI-TARS-1.5's SFT/DPO training actually damages — UI-TARS-1.5 scores *worse* on relational accuracy (35.0%) than even the untrained Qwen2.5-VL base model it started from (45.0%). Supervised fine-tuning on labeled click targets seems to teach the model to match visual patterns extremely well, at the cost of the geometric reasoning needed when patterns aren't enough.

Read back against my own failures: "second chain's title" vs. "device title bar" is a relational/structural disambiguation, not a direct naming task — the model has to know *which panel* it's looking at, not just recognize "title bar" as a visual pattern. "Locator marker flag" vs. "loop brace" are two visually distinct but spatially adjacent marks on the same timeline ruler. These are close cousins of the exact category the paper measured — and my two-model split (GTA1 solid, UI-TARS specifically weak on these) landed in the same direction as their statistically validated 65.8%-vs-35.0% gap.

I want to be honest about the scale mismatch. GUI-Perturbed ran controlled statistical tests — bootstrap confidence intervals, McNemar's test — across 390 matched sample pairs per condition. I ran an informal 30-task check on Ableton Live screenshots, a domain nobody's benchmark actually covers. This isn't a replication in any rigorous sense. But the *mechanism* — a model built on additional RL-for-clicks outperforming its SFT-trained parent specifically on tasks that require telling apart structurally similar, nearby elements — showed up independently, in a completely different application, using nothing but eyeballing crosshairs on screenshots. That's a reassuring kind of convergence, even from a small sample.

## What I'd actually take away from this

A few things, roughly in order of how much I trust them:

1. **General-purpose VLMs are not GUI-grounding models**, even good ones. Qwen2.5-VL-7B-Instruct going 0/6 wasn't subtle — it's a different capability that needs its own training, not something that falls out of general visual competence for free.
2. **Aggregate benchmark scores can point the wrong way for your specific use case.** If your actual task leans on distinguishing between structurally similar elements — separate panels that look alike, adjacent icons, elements identified by their position relative to something else — the model that wins on the standard leaderboard might not be the one you want.
3. **How a grounding model was post-trained matters more than its parameter count or its headline benchmark score.** RL-with-click-reward and SFT-on-labeled-data produce models with genuinely different failure profiles, not just different average accuracy.
4. **Cheap triage beats exhaustive review.** Comparing two models' raw coordinates to find disagreement, then only opening the images where they diverge, found every interesting failure in this project without manually reviewing all 60 annotated screenshots.
5. **The unglamorous bugs will eat more of your time than the actual research question.** Disk management and a `transformers` API change cost more debugging time across this project than anything related to model quality.

## Try it yourself

Everything here — the benchmark script, the diversity-selection tool, the screenshots, the raw results, and the annotated crosshair images — is public:

**[github.com/akbargherbal/gui_grounding_benchmark](https://github.com/akbargherbal/gui_grounding_benchmark)**

It's built to run end-to-end on a free-tier-adjacent Colab GPU (an L4 with 24GB VRAM is comfortable; a T4 with 16GB will likely OOM on these 7B models in bf16). Clone the repo, open the notebook, run it, and you'll get your own `report.md` and a folder of annotated PNGs to eyeball. If you swap in your own screenshots — from your own software, your own workflow — I'd genuinely be curious whether the same "GTA1 wins on the hard, structurally-ambiguous ones" pattern holds, or whether Ableton Live's particular density of similar-looking panels made this an unusually favorable domain for it to show up in.

That's the honest state of this project: not a benchmark paper, but a real, repeatable way to point a couple of open-source models at your own screen and see, with your own eyes, whether they can actually find the thing you asked for.
