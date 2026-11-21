# ASPred — working constraints

This is the **parent** repo. The directory `ASPred/` inside it is a **separate git repo
with the same remote URL** (`github.com/wangjeffrey42/ASPred.git`), tracked as a gitlink.

## Git: read this before any git operation

The self-referential setup above has already caused a diverged `main` and an unreachable
gitlink. Therefore:

- **Never force push.** Either repo.
- **Never rebase the submodule.**
- **Order of operations:** commit in the submodule → push the submodule → bump the parent
  pointer → push the parent. The pointer must never name a commit that isn't on the remote,
  or the gitlink is unreachable for anyone else.
- There is **no `.gitmodules` entry** for `ASPred/`, so `git submodule` commands fail with
  `no submodule mapping found`. The gitlink is real and tracked; only the mapping is absent.
  Don't "fix" this by adding a mapping without being asked — the shared remote URL is why it
  was left out.

### Current known state (2026-11-20, unresolved)

- Submodule `main` and `origin/main` have **diverged 3-and-3** from `9ff4bdb`. Both sides
  carry versions of the same epistasis work. Reconciling them is a deliberate content
  decision — do not merge, rebase, or force either side without being asked.
- Analysis work belongs on **`analysis-epistasis`**, which is clean and fast-forwards.
- **162 files are staged but uncommitted** in the submodule. They must not be swept into
  unrelated commits. Commit with an explicit pathspec (`git commit <path> -m ...`), never
  `git commit -a` or a bare `git commit` — a bare commit takes all 162.

  They are *not* confined to `data/` as earlier notes suggested. Actual spread:
  `models/` 64 (LoRA adapters and heads for runs 1–7, incl. ABANDONED/INVALID runs),
  `results/` 44 (poster charts, training logs, test predictions), `src/` 30 (training and
  plotting scripts), `data/` 24 (13 in `data/paired/`, 11 in `data/rbd_dms/`).

  `results/.DS_Store` is among them and should be dropped from the index, not committed.
  Large binaries (`adapter_model.safetensors`, `head.pt`) are staged too — worth deciding
  on LFS or exclusion before committing any of this.
- The parent has its own uncommitted changes. This file is one of them.

## The three datasets

| where | library | what |
|---|---|---|
| `RBD Data/` (**parent**, untracked by the submodule) | B, C, M | Full upstream release, zipped, + `File_Details.csv`/`.xlsx` metadata and per-library raw-processing folders |
| `ASPred/data/rbd_dms/` | M only | The 8 `MABAa_*` phenotype CSVs actually used, unzipped |
| `ASPred/data/paired/` | M only | Model-ready `(antibody_sequence, rbd_sequence, affinity)` pairs; `rbd_split_{train,val,test}.csv` = 821,868 / 102,714 / 102,780 rows over 6 antibodies |

- **Library B**: 1,152 variants, whole mutational landscape of BA.1/BA.2/BA.5/BA.2.12.1 on
  the ancestral Omicron background. Phenotype is KD.
- **Library C**: 16,800 variants — BA.2 mutations in all combinations on each of 7
  backgrounds. The 7 loci are **not biallelic**: allele counts are 5/5/4/2/3/2/2
  (346/444/452/460/486/490/493), so the design is 5×5×4×2×3×2×2 = 2,400 per background,
  not 2⁷ = 128. Coverage is 95.7–98.7% per background. `pos_background` also has a
  catch-all value `9` (37,061 rows, 19,306 strains) that is not one of the 7 and must be
  excluded from per-background counts. Antibody files are *fluorescence* (`Fluor at 1uM`, `Fluor at 10nM`, two
  replicates each), not KD; only the ACE2 file is KD.
- **Library M**: the one the model is trained on. ~300,000 of 63.7M theoretical variants,
  4 backgrounds × combinations of 26 JN.1/XBB.1 mutations. Antibody phenotype is
  `Enrichment at 100nM`.

## Unresolved: the six groups sit on two incompatible scales

`File_Details.csv` labels all six analysis antibodies identically — `Enrichment at 100nM`,
column `ph_mean`. The metadata does **not** distinguish them. The values do:

| set | antibodies | range | point mass |
|---|---|---|---|
| floor | 002-S21F2, AZD8895, S309 | `[0.00, 3.3–5.2]` | 49–76% sit exactly at 0 |
| ceiling | AZD1061, LY-CoV1404, REGN10987 | `(−2.8, 1.00]` | 55–61% sit exactly at 1 |

The upstream `RBD Data/README.txt` explains the mechanism: the enrichment score depends on
which subpopulation was sorted. An evading-population sort yields `s` (escape; 0 = none,
may exceed 1); a binding-population sort yields `1 − s` (1 = full binding, may go negative).
The 3/3 split matches that exactly, so **the sign of the phenotype is inverted between the
two sets** — a positive coefficient means "more escape" for the floor set and "more binding"
for the ceiling set.

This is inferred from the bound structure, not recorded anywhere upstream. Until it is
confirmed, **do not compare coefficient signs across the two sets**, and do not average or
pool raw affinity across them. Within-set comparisons and all per-group ratios are
unaffected. `ASPred/analysis/epistasis/README.md` lists every result and figure this
touches; the two that bite hardest:

- **`label == 1` does not mean the same thing in every group.** The labeled datasets
  (`data/paired/{group,variant}_holdout_labeled/`) and
  `src/plot_binds_escaped_split.py` call it "Binds", but for the floor set
  `affinity > median` means *more escape*. The legend is inverted for three of six
  antibodies. `group_holdout_labeled/` is the live hazard — train is 3 ceiling + 1 floor
  and test is S309 (floor), so an antibody-generalization run there transfers an inverted
  label. Fix the convention before using that split.
- **Per-group model metrics are fine; pooled ones are not.** The `pooled` row of
  `results/paired_classifier_test_metrics.csv` mixes both conventions.

## Metadata defect in `RBD Data/`

`File_Details.csv` has a row for `XABA_ACE2_rep_c_phenotype_data.csv`, but the file on disk
is `XABA_ACE2_rep_d_phenotype_data.csv.zip`. There is no `rep_c` file and no `rep_d` row —
a `rep_c`/`rep_d` mismatch, so the ACE2 KD file for Library C has no metadata row under its
actual name. (`XABA_Exp_rep_d` is listed correctly, so the `_d` suffix is otherwise valid.)

## Environment

Python work needs the `aspred` conda env (`conda activate aspred`); `run.sh` enforces this.
The base env has no pandas, and `aspred` has no `openpyxl` — read `File_Details.csv.zip`
rather than the `.xlsx`.

## Canonical model checkpoint

`ASPred/models/paired_affinity/run5_rbd_split_continued` (see its `README_STATUS.txt`).
`run6` is a plateau-confirmation run and is **not** canonical.
