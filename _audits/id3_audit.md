# ID3 Pipeline Audit

**Date:** 2026-05-17 · **Branch:** main · **Author:** Claude (Opus 4.7, 1M ctx)
**Subject:** Post-refactor audit of `ai/id3.py`, `ai/dt_pipeline.py`, `ai/dt_player.py`

This document is an honest assessment of the current state of the
decision-tree work, written to inform the next prompt's experiment
design. It is not a marketing piece — the gaps are real and the
recommendations are opinionated.

---

## Section A — What the current implementation does

The pipeline is now organised as `id3` (algorithm) → `dt_pipeline`
(pre-processing, training, evaluation, CLI) → `dt_player` (game-engine
adapter — stub). Below is the per-step inventory in pipeline order.

| Step | Function | What it does | Notable choice |
|---|---|---|---|
| **Load + inspect** | `inspect_dataset` ([ai/dt_pipeline.py:223](ai/dt_pipeline.py:223)) | Reads CSV, auto-detects the class column (case-insensitive `class`/`move`/`label`), converts string moves (`drop_3`, `pop_5`) → integers (0–13), reports per-class counts and imbalance ratio, partitions columns into "categorical" (≤3 unique values) vs "continuous". | The continuous detector is heuristic — picks up `move_count` (69 unique values), `own/opp_pieces_bottom_row` (7 each) but classifies all 42 cells as categorical because each only takes 0/1/2. Works for our schema but is fragile to schema drift. |
| **Balance classes** | `balance_classes` ([ai/dt_pipeline.py:303](ai/dt_pipeline.py:303)) | Caps each class at `cap=50000` rows; under-represented classes (POP moves: 8k–25k) kept whole; shuffles result. | 49.3:1 raw imbalance (drop_3: 388k vs pop_6: 7.9k) → after cap, still ≈6:1 imbalance between drop_0 (50k) and pop_6 (7.9k). **Balancing improves but doesn't equalise.** |
| **Split** | `stratified_split` + `split_dataset` ([ai/dt_pipeline.py:360](ai/dt_pipeline.py:360)) | Class-stratified 80/20 train+val/test, then 90/10 of train+val → train/val. Pure numpy/pandas. | Three-way split — uses validation set for hyper-parameter tuning. **No k-fold cross-validation.** |
| **Bin features** | `bin_features` ([ai/dt_pipeline.py:425](ai/dt_pipeline.py:425)) | Applies static, hardcoded bins from `POPOUT_BIN_DEFINITIONS` to the 3 continuous columns: `move_count` → {early, mid, late, endgame}, `own/opp_pieces_bottom_row` → {low, mid, high}. | **Bins are hand-designed, not data-driven** — no information-gain optimisation, no quantile binning. Reasonable for our problem (the bin boundaries correspond to roughly meaningful game phases) but undocumented why these specific cuts were chosen. |
| **Tune** | `tune_pruning` ([ai/dt_pipeline.py:475](ai/dt_pipeline.py:475)) | 5 × 4 grid over `max_depth ∈ {5,8,10,15,20}` × `min_samples ∈ {10,20,50,100}` → 20 trees trained on train, scored on val. Tracks val accuracy, POP-class accuracy (classes 7–13), node count, time. Tie-broken by smaller tree. | **Only pre-pruning.** No post-pruning (reduced-error or cost-complexity). Default is `SKIP_TUNING=True` with `BEST_DEPTH=15`, `BEST_MIN=20` hardcoded into `_cli_train` — the grid search has presumably been run once but the results aren't tracked in the repo. |
| **Train final** | `train_final_tree` ([ai/dt_pipeline.py:570](ai/dt_pipeline.py:570)) | Re-trains on train+val combined with the best hyper-parameters; reports node/leaf/depth/root-split stats. | Uses the same `id3` function — no special handling of "final" beyond the larger training set. |
| **Visualise** | `visualise_tree` ([ai/dt_pipeline.py:670](ai/dt_pipeline.py:670)) | Box-drawing ASCII/emoji printer; truncates beyond `max_display=5` levels with `···`. | Pretty but **plain text only.** No Graphviz, no SVG, no notebook-friendly rendering. The spec asks for visual presentation; this works in a terminal but won't look great in a slide deck. |
| **Evaluate** | `evaluate` ([ai/dt_pipeline.py:715](ai/dt_pipeline.py:715)) | Overall accuracy + per-class accuracy + DROP/POP roll-up against the test set; majority-class baseline reported. | **No confusion matrix, no macro-F1, no top-k accuracy, no precision/recall.** Per-class accuracy is computed but the per-class precision (false-positive rate) is not. |
| **Persist** | `save_tree` / `load_tree` ([ai/dt_pipeline.py:790](ai/dt_pipeline.py:790)) | Plain `pickle.dump` / `pickle.load`. Default path: `data/id3_tree.pkl`. | One artefact, no metadata. If we ever train multiple trees (baseline vs optimised), they overwrite each other unless the caller is careful. |
| **Iris demo** | `run_iris_demo` ([ai/dt_pipeline.py:115](ai/dt_pipeline.py:115)) | Loads `iris.csv`; for each numeric feature, picks the **single binary threshold** that maximises information gain on the full dataset; bins into `low`/`high`; trains; reports accuracy + full tree. | **Reaches only ~66.7% test accuracy on iris** — see Section D for why this is a bug, not a feature. |

### Surprising choices

1. **Iris demo accuracy.** The per-feature binary discretisation in
   `run_iris_demo` (each feature gets ONE optimal threshold globally)
   perfectly separates `setosa` (using `petal_length≤2.45` or
   `petal_width≤0.8`), then runs out of discriminative power — the
   remaining 4 binary features cannot distinguish `versicolor` from
   `virginica`. Result: 100% on setosa, ~50% on the others, ≈66.7%
   overall. The dead `decision_tree.py` had the same flaw.
   The smoke test in `tests/test_id3.py` explicitly asserts >60% (not
   >90%) for this reason.

2. **CLI runs iris warm-up every time before PopOut training.** Wastes
   ~1s on every full pipeline invocation; arguably should be opt-in.

3. **`tune_pruning(skip=True)` is the CLI default.** Repeats the same
   tree on every run rather than re-discovering it. Saves time but
   means the grid is essentially dead code unless flipped manually.

4. **`current_player` is included as a feature.** Trees may end up
   splitting on whose turn it is, which is correct for capturing
   per-player tactical asymmetry but worth noting — the comparison
   project (`outro`) chose to *include* it; some imitation-learning
   work would prefer canonicalising the board to a "current-player's
   perspective" representation, eliminating the feature.

---

## Section B — How the comparison project (`outro/lib/d_tree`) differs

The `outro` team built a comparable Connect-4 + DT system. Their tree
is implemented at `outro/lib/d_tree/build_dt.py` with helpers in
`read_dataset.py`. Major divergences:

| Dimension | Ours (`ai/`) | Theirs (`outro/lib/d_tree`) | Comment |
|---|---|---|---|
| **Tree representation** | Nested `dict` with `is_leaf`/`feature`/`majority`/`children` | `Node` dataclass with `feature`/`value`/`true_branch`/`false_branch`/`results` | Theirs is binary-tree-of-objects; ours is multi-way-dict. Both fine. |
| **Split type** | Categorical multi-way (one child per unique value of the chosen feature) | Continuous binary (`<= value` vs `> value`) on every feature, with percentile candidate thresholds for high-cardinality features | Ours is closer to "classical ID3"; theirs is closer to C4.5/CART. |
| **Entropy implementation** | `pd.value_counts()` → probs → `-Σ p log p` | `np.bincount(data)` → probs → `-Σ p log p` | Theirs requires integer-encoded labels (uses `bincount`); ours works on any hashable. |
| **Pruning** | Pre-pruning only (`max_depth`, `min_samples`) | Pre-pruning + **Reduced Error Pruning (REP)** + **Cost-Complexity Pruning (CCP, α-sweep)** | **Major gap on our side.** Theirs explicitly compare unpruned/REP/CCP in `compare_pruning_methods`. |
| **Class balancing** | Cap each class at 50k (undersampling) | None — train on raw distribution | Theirs explicitly notes the dataset is dominated by drop_3; we mitigate, they don't. |
| **Evaluation methodology** | Single train/val/test split | `k_fold` with k=10 cross-validation; also `k_fold_with_pruning` | **Theirs is more rigorous;** ours has higher variance estimate. |
| **Metrics** | Overall + per-class accuracy + DROP/POP roll-up + baseline | Average accuracy + std-dev across folds; no per-class breakdown | Both miss confusion matrix / macro-F1. |
| **Sample-efficiency plot** | None | `plot_accuracy_vs_dataset_size` (accuracy vs N) | Useful for the slides — shows whether more data would help. |
| **Iris discretisation** | Per-feature binary IG-optimal split | No discretisation — uses the binary `<= value` test natively | Theirs avoids our iris-accuracy problem entirely by not discretising. |
| **Feature engineering** | 42 cells + `move_count` + `current_player` + `own/opp_pieces_bottom_row` (46 features) | 43 features: `turn` + 42 board positions | We have 4 derived features they don't (move_count, current_player, own/opp pieces in bottom row). Their report claims switching from bitboards (2 ints) to per-cell features lifted accuracy from 17% to 34% — our schema already uses per-cell. |
| **DT-vs-MCTS play** | DTPlayer is a stub | Reported "Decision Tree vs MCTS: 0%-100%" (DT loses every game) and **"Decision Tree vs Decision tree: Not possible"** because the DT plays illegal moves | We must implement legality fallback in `DTPlayer.choose_move`; the outro team didn't and it bit them. |
| **Reported test accuracy** | Unknown (CLI default hides grid search; the colleague's commit message implies ≈30-something%) | 34.7% (unpruned), 33.9% (REP), 33.7% (CCP) on their best configuration | Our cap-50k balancing means our test set is more uniform than theirs, so accuracy numbers are not directly comparable. |

**Bottom line on the comparison:** the `outro` project is methodologically
more thorough in *evaluation* (k-fold, multiple pruning methods,
sample-size sweep, comparison tables) but methodologically simpler in
*pre-processing* (no balancing, no domain-knowledge bins). They also
admit their DT player is unusable for self-play. Our pipeline is the
inverse: better pre-processing, weaker evaluation, no DT player yet.

---

## Section C — Spec requirements coverage

Mapping §4.2 of the assignment spec ([project_description.pdf](../../project_description.pdf), p. 3) and §3 (game interface, p. 2):

| Requirement | Met? | Where |
|---|---|---|
| ID3 implemented from scratch (no scikit-learn) | ✓ | [ai/id3.py](../ai/id3.py) |
| Iris warm-up runs | ✓ (runs end-to-end, accuracy is poor — see Section D) | [ai/dt_pipeline.py:115](../ai/dt_pipeline.py:115) |
| Discretise iris numerical attributes "to minimize tree size" | ✓ (single-threshold-per-feature IG split) | [ai/dt_pipeline.py:144-162](../ai/dt_pipeline.py:144) |
| PopOut self-play dataset generated via MCTS | ✓ | `data/popout_dataset_150k.csv` (1.46M rows, 150k games, generated by `scripts/generate_dataset.py`) |
| Train tree on PopOut data | ✓ | [ai/dt_pipeline.py:570](../ai/dt_pipeline.py:570) |
| Visualise tree | ✓ (terminal text + emoji) | [ai/dt_pipeline.py:670](../ai/dt_pipeline.py:670) |
| Classify new test examples | ✓ (CLI: `--predict <csv>`) | [ai/dt_pipeline.py:823](../ai/dt_pipeline.py:823) |
| §3 game scenario 3: **computer-vs-computer (two different algorithms)** | ✗ | `DTPlayer.choose_move` is a stub; `scripts/run_tournament.py` only constructs `MCTSConfiguredPlayer` |
| §3 game scenario 2: **human-vs-computer with the DT** | ✗ | Same blocker as above |
| §3 game scenario 1: **human-vs-human** | ✓ (`HumanPlayer` exists) | [game/player.py:70](../game/player.py:70) |

**The single biggest spec gap is computer-vs-computer with two different
algorithms** (DTPlayer vs MCTSPlayer). This is the only path to honestly
evaluate the tree's *playing strength* — classification accuracy on
held-out MCTS labels is a proxy at best.

---

## Section D — Experimental coverage gaps

The grading rubric weights "rigour in performance evaluation" at 30%
(p. 4 of the spec). The following experiments are missing or
under-developed. Each entry: **what** / **why** / **effort**.

### High value, low effort

1. **Iris discretisation fix.** *What:* Replace
   `run_iris_demo`'s single-threshold-per-feature binning with either
   (a) recursive IG splits during tree building (allows feature reuse
   with different thresholds), or (b) per-class quantile binning into
   3+ bins, or (c) skip discretisation and let the algorithm split on
   `<= value` natively (CART-style). *Why:* Iris is *trivially*
   separable by a real ID3; reporting 66.7% on the warm-up dataset
   undermines the credibility of the whole DT story. The slides will
   look bad. *Effort:* small (~1 prompt). Note that option (a) requires
   the only modification to the core ID3 logic we'd make — needs an
   "allow continuous splits" path.

2. **Confusion matrix + macro-F1.** *What:* Add a confusion-matrix
   helper to `evaluate`; compute macro-F1 (and micro-F1) for the test
   set. *Why:* With 14 imbalanced classes (drop_3 dominates even after
   capping), per-class accuracy on its own is misleading — a tree that
   predicts drop_3 always would score ~25% accuracy but 0% recall on
   POP moves. Macro-F1 forces the class imbalance to be visible. The
   web confirms macro-F1 is standard for imbalanced multi-class
   ([arXiv 2008.05756](https://arxiv.org/pdf/2008.05756)). *Effort:*
   small. Pure pandas/numpy, no new dependencies.

3. **Top-k accuracy (k=2, 3).** *What:* For each test row, ask "is
   the true MCTS move in the top-k predictions of the tree?". *Why:*
   MCTS itself is non-deterministic and often has 2-3 near-equivalent
   moves; the tree being "wrong" by predicting a slightly-different
   high-quality move shouldn't be punished as hard as predicting
   garbage. Reading from the imitation-learning literature (the Scripts
   of Tribute paper at [AAU](https://projekter.aau.dk/projekter/files/822057972/MasterThesis_LT_02_2026.pdf)
   reports BC plateaus around 56–59% top-1 accuracy on MCTS labels —
   our 30-something % is in that same regime). *Effort:* small —
   `id3.predict_one` needs to return the leaf's full class distribution
   rather than just the argmax; thin helper to rank.

4. **Feature importance / first-split analysis.** *What:* Walk the
   trained tree, count how often each feature appears as a split, and
   weight by the size of the data partition at that split. Report the
   top-10. Also note which feature is at the root and the second
   level. *Why:* Useful slide content — "the tree learns to look at
   the centre column first" tells a story. *Effort:* small. Pure tree
   traversal.

### High value, medium effort

5. **Baseline-vs-optimised tree comparison.** *What:* We have *two*
   MCTS datasets — the early 5k-iteration baseline and the current
   20k-iteration "optimised" one (per the v0.3.0 commit `d614f38`).
   Train one tree on each, report classification accuracy +
   macro-F1 + tree-vs-tree tournament. *Why:* The whole MCTS tuning
   effort needs to pay off downstream. If the optimised dataset
   doesn't produce a better tree, that's an important null result
   (and surprising — could mean the bottleneck is the tree, not the
   data). *Effort:* medium. Requires re-running training twice (each
   takes O(minutes) on the cap-50k data) plus the comparison
   apparatus. Doesn't require any new MCTS runs since both datasets
   already exist.

6. **DTPlayer implementation + Tree-vs-MCTS tournament.** *What:*
   Finish `DTPlayer.choose_move` (feature extraction → bin → predict
   → decode → legality check → fallback). Run, say, 200 games of
   DT vs MCTS(5k iter) with alternating starts. Report win rate as
   P1, as P2, and overall. *Why:* This is the spec's
   computer-vs-computer scenario AND the most honest signal of
   the tree's actual playing strength. Per the imitation-learning
   literature, expect the tree to win 0–20% — that's not a failure,
   that's the BC-vs-search baseline regime ([Scripts of Tribute paper
   reports 12–15% vs MCTS](https://projekter.aau.dk/projekter/files/822057972/MasterThesis_LT_02_2026.pdf)).
   The `outro` team reported 0% in their write-up. *Effort:* medium
   (tournament runner needs DT-aware factory; the DT player itself is
   maybe 80 lines).

7. **Tree-vs-Random tournament.** *What:* 500 games of DT vs
   `RandomPlayer`. *Why:* Sanity check that the tree learned
   *anything* meaningful beyond "play in the centre". If the tree
   doesn't beat random ≥80%, something is broken. *Effort:* small
   once DTPlayer exists (10 minutes of compute).

### Medium value, medium-to-high effort

8. **Post-pruning ablation: REP + CCP.** *What:* Implement reduced-
   error pruning (post-order traversal of the trained tree, swap each
   internal node for a majority-class leaf, keep the swap if val
   accuracy improves) and cost-complexity pruning (sweep α ∈ {0.001,
   0.01, 0.05}). Compare unpruned/REP/CCP on test accuracy, macro-F1,
   node count. *Why:* The web literature consistently flags
   post-pruning as the bigger lever for generalisation in noisy ID3
   trees — pre-pruning only is "leaving accuracy on the table" per the
   sklearn docs and [the Springer empirical comparison study](https://link.springer.com/content/pdf/10.1023/A:1022604100933.pdf).
   The `outro` team also did this — comparison parity matters for
   grading. *Effort:* medium (REP is ~30 lines; CCP is ~80 lines).
   Note: the original `popout_id3.py` has zero pruning code; we'd be
   adding it from scratch.

9. **k-fold cross-validation.** *What:* 5- or 10-fold CV instead of a
   single train/val/test split. Report mean ± std accuracy across
   folds. *Why:* Single-split variance can be ±2–3% with only ~90k
   test rows; CV gives an honest variance estimate. Standard practice
   in academic write-ups; the `outro` team did this. *Effort:* medium
   (~50 lines, but the full pipeline runs k× longer — needs to be
   parallelised or run on the cap-50k data not the full dataset to
   stay tractable).

### Lower value (still worth flagging)

10. **Sensitivity to `cap`.** *What:* Train trees at
    `cap ∈ {10k, 50k, 100k, ∞}` and compare test macro-F1. *Why:*
    Defensible to know if 50k was the right call. *Effort:* small.

11. **Sample-efficiency curve.** *What:* Accuracy vs training-set
    size (5k, 10k, 50k, 100k, full). *Why:* Tells us whether the
    plateau is data-bound or model-bound. The `outro` team did this
    too. *Effort:* small.

12. **Tree-depth-vs-accuracy curve.** *What:* Hold all else fixed,
    sweep `max_depth ∈ {3, 5, 8, 10, 15, 20, 30}`. *Why:* Lets us
    show the overfitting curve. The current grid search covers some
    of this but doesn't plot it. *Effort:* small.

13. **POP-move analysis.** *What:* The POP classes are 7× rarer than
    average even after capping (no class with >25k samples). Examine
    per-POP-class metrics and whether the tree learned them at all.
    *Why:* The MCTS engine uses pops strategically; if the DT never
    predicts a pop, that's a meaningful behavioural gap. The colleague
    already tracks `pop_acc` in `tune_pruning` — extend it to test
    set and confusion matrix. *Effort:* small.

14. **Graphviz tree rendering.** *What:* Add an optional
    `tree_to_graphviz` export that writes a `.dot` file. *Why:* The
    spec calls for visual presentation; emoji-in-terminal is fine
    for development but won't scale to a slide deck — even the top 5
    levels of a tree with 14 children per split is dense. *Effort:*
    small (~50 lines, no new dependencies — graphviz reads `.dot`
    natively).

---

## Section E — Recommended next steps (prioritised)

Order reflects (a) impact on the grade and (b) work to unlock
downstream experiments. Each item below should be a separate prompt.

### Tier 1 — must-do before slides

1. **Implement `DTPlayer.choose_move` + DT-vs-MCTS tournament.**
   This is the only spec-mandated item (§3 scenario 3 — different
   algorithms) we don't currently meet, and it's the most credible
   way to evaluate playing strength. Plan: implement the feature
   extraction in `dt_player.py`, extend `scripts/run_tournament.py`
   to accept a `"type": "dt"|"mcts"` field per player config, run a
   200-game tournament alternating starts, write results to
   `data/tournament_dt_vs_mcts.csv`. *Tracks gap 6 + spec gap §3-3.*

2. **Train baseline-vs-optimised trees, head-to-head comparison.**
   The MCTS tuning effort produced the optimised dataset; we need to
   prove (or disprove) that downstream tree quality moved. Plan:
   train one tree on each dataset, compare overall accuracy,
   macro-F1, confusion matrix, and run a DT-baseline-vs-DT-optimised
   tournament. *Tracks gap 5.*

3. **Add confusion matrix + macro-F1 to `evaluate`.** Cheap, central
   to honest reporting on imbalanced multi-class, blocks proper
   evaluation of items 1 and 2 above. *Tracks gap 2.*

### Tier 2 — strengthens rigour, fits in remaining time

4. **Fix the iris demo.** Either CART-style continuous splits (allow
   `id3` to optionally accept numeric thresholds) or multi-bin
   quantile discretisation. >90% on iris is the canonical sanity
   check; current 66.7% is embarrassing on slides. *Tracks gap 1.*

5. **Feature-importance + first-split analysis** for the slide deck
   narrative. Pure traversal of the trained tree, ~30 lines.
   *Tracks gap 4.*

6. **Top-k accuracy** for both trees (baseline + optimised), to put
   the 30-something % overall accuracy in context vs the
   imitation-learning literature. *Tracks gap 3.*

### Tier 3 — nice-to-have, prioritise if time permits

7. **REP pruning ablation.** Adds parity with the `outro` team and
   tests a methodologically standard claim. Skip CCP if pressed for
   time — REP is simpler and likely sufficient. *Tracks gap 8.*

8. **Tree-vs-Random sanity check.** 5 minutes of compute once the
   tournament runner accepts a `RandomPlayer` — useful as a baseline
   number in the slides. *Tracks gap 7.*

9. **Sample-efficiency curve** OR **depth-accuracy curve** (pick one
   based on which story is more interesting after the trees are
   trained). *Tracks gaps 11, 12.*

### Explicit non-priorities

- **k-fold CV** is methodologically nice but takes k× the training
  time. With a single ~95k-row test set we already have ±2-3% noise;
  one CV pass would tighten that but not change conclusions.
- **CCP pruning** — REP gives you ~80% of the post-pruning benefit
  at ~30% of the implementation effort. Skip unless Tier 2 finishes
  early.
- **Graphviz tree rendering** — only matters if the slide deck
  reviewer is going to squint at the tree. The current text printer
  can be screenshotted.

---

## Appendix — known issues not for the next prompt

- `README.md` at the repo root still references `decision_tree.py`,
  which has been deleted. Out of scope per the refactor constraints
  ("do not touch any file outside `ai/`, `tests/`, `_audits/`, and
  `data/popout_dataset_150k.csv`"); flagged here so it's not lost.
- The CLI emoji output uses `📦`/`🍃`, which raise
  `UnicodeEncodeError` on Windows cp1252 consoles. The CLI now
  reconfigures `sys.stdout` to UTF-8 best-effort at the `__main__`
  guard; this was a pre-existing bug in `popout_id3.py` and the fix
  is environment shim, not a behavioural change.
- The `--predict` CLI mode silently drops rows where binning maps a
  value to `NaN`. Not a refactor change — pre-existing behaviour
  inherited verbatim. Worth handling explicitly when the DTPlayer is
  implemented, since the player will need to validate that all
  features bin cleanly before calling `predict_one`.

---

## Sources consulted (web research)

- [Decision tree pruning — Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_pruning)
- [scikit-learn: cost complexity pruning](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)
- [Esposito, Malerba, Semeraro — empirical comparison of pruning methods (Springer)](https://link.springer.com/content/pdf/10.1023/A:1022604100933.pdf)
- [ID3, C4.5, CART and pruning — Bitmask93 blog](https://bitmask93.github.io/ml-blog/ID3-C4-5-CART-and-Pruning/)
- [Combining Gameplay Data with MCTS to Emulate Human Play (AAAI)](https://cdn.aaai.org/ojs/12858/12858-52-16374-1-2-20201228.pdf)
- [Learning-Based Decision Making in a Competitive Card Game — Aalborg University MSc thesis (2026)](https://projekter.aau.dk/projekter/files/822057972/MasterThesis_LT_02_2026.pdf)
- [Grandini, Bagli, Visani — Metrics for multi-class classification: an overview (arXiv 2008.05756)](https://arxiv.org/pdf/2008.05756)
- [`imitation` library — Behavioral Cloning docs](https://imitation.readthedocs.io/en/latest/algorithms/bc.html)
