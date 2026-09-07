# Paper architecture

This directory is a venue-neutral, modular scaffold for the causal-role
erasure paper. The current driver uses the standard article class only to make
the architecture inspectable and compilable. When the venue is frozen, change
main.tex and keep paper.tex, supplement.tex, section files, labels, figures,
and tables stable.

## Paper spine

The manuscript follows one claim-driven chain:

1. A named source can become less visible while its downstream physical
   footprint remains.
2. Treating a noun as an unconditional erasure target also damages legitimate
   receiver or bystander uses.
3. Causal-role erasure therefore targets the source role and its dependent
   footprint while preserving the receiver, video quality, and noncausal uses.
4. Source-slot Randomized Counterfactual Distillation varies the identity in
   the causal-source slot while holding a dynamic no-source target fixed.
5. Main, generalization, identification, specificity, and failure evidence
   test the resulting behavioral interpretation.

The recurring three-part vocabulary is:

> source - footprint - specificity

Use these terms consistently instead of cycling through synonyms.

## Main-paper architecture

| Section | Function | Main evidence | Budget |
| --- | --- | --- | ---: |
| Abstract | Complete problem-to-evidence arc in five sentences | Final headline only | 0.25 page |
| 1 Introduction | Phenomenon, gap, method idea, evidence, contributions | Task teaser | 1.10 pages |
| 2 Related Work | Position the task without bibliography dumping | Three precise gaps | 0.55 page |
| 3 Causal-Role Erasure | Define source, receiver, footprint, and no-source world | Task diagram and desiderata | 0.85 page |
| 4 Method | Pair construction, source-slot intervention, objectives | Method overview | 1.55 pages |
| 5 Experiments | Efficacy, role evidence, generalization, preservation, failures | Main tables and compact qualitative figure | 3.15 pages |
| 6 Scope and Limitations | Freeze what the evidence does and does not establish | Explicit boundaries | 0.35 page |
| 7 Conclusion | Recover the problem, intervention, and supported behavior | No new claims | 0.20 page |

The eight-page budget is a planning target rather than a venue decision.

## Appendix architecture

| Appendix | Reviewer question |
| --- | --- |
| A Task Scope and Ontology | What exactly are the seven mechanisms and their causal roles? |
| B Counterfactual Data | How were targets built and screened without leakage or cherry-picking? |
| C Training and Baselines | Is the Matched-versus-method comparison single-factor, and are baselines faithful? |
| D Evaluation and Metrics | What did VLMs and humans see, and how do atomic labels become CES/SU? |
| E Complete Results | Are all methods, mechanisms, guardrails, contrasts, and secondary outcomes visible? |
| F Controls and Sensitivity | Is the behavior role-conditioned rather than lexical, and is it aggregation-robust? |
| G Qualitative and Failures | What changes over time, where does the method fail, and how were examples chosen? |
| H Reproducibility | Can every table be traced to frozen registries, commands, and receipts? |

## Writing contract

Each subsection must answer one question. Its first paragraph follows:

> claim or question -> decisive evidence -> interpretation -> boundary

Method paragraphs follow:

> failure mode -> design response -> definition or equation -> plain-language
> mechanism -> experiment that tests it

Use short bold run-in headings inside dense setup or results subsections.
Captions must be independently readable and state the observation the figure
supports. A caption must not claim preservation when the displayed receiver,
action, or scene has visibly changed.

The main text keeps only evidence that changes the paper's conclusion.
Complete category tables, prompts, rubrics, hyperparameters, provenance, and
large qualitative grids belong in the appendix.

## Claim gates

Current preliminary results are isolated from final paper tables. Until
human-canonical scoring is complete, do not turn preview values into final
scientific claims.

The paper must also disclose that a separate read-only audit opened the full
method key before human canonicalization and before the exploratory A/B-mean
rule was materialized. Human reviewers may remain answer-key blind, but the
global pre-unblinding state cannot be restored.

Allowed only when supported by the final artifact:

- highest observed macro CES;
- a positive observed gain over the same-backbone Matched Control;
- behavior consistent with role-conditioned generalization;
- preservation within prespecified non-inferiority margins.

Never infer from the current design alone:

- significant superiority over every baseline;
- discovery of an internal true causal representation;
- complete or universal source-and-footprint removal;
- one universal adapter across all seven mechanisms;
- a pure method comparison between Wan and CogVideoX raw scores.

The non-inferiority table should report estimate, confidence interval, margin,
and pass/fail. A superiority p-value relative to zero must not be labeled as a
non-inferiority p-value.

## Figure and table plan

Main figures:

1. Task teaser: source, receiver, and lingering footprint.
2. Method overview: structured pair, randomized source slot, fixed no-source
   target, distillation, and preservation.
3. Compact temporal comparison: one success, one implicit-footprint case, and
   one failure, selected by a metadata-only deterministic rule frozen after
   aggregate unblinding and labeled descriptive.

Main tables:

1. Eight methods by seven mechanisms plus equal-weight macro CES, separated
   into Wan and CogVideoX blocks.
2. Macro efficacy and preservation guardrails.
3. Generic-paraphrase and bystander-token identification controls, including
   specificity.
4. Implicit-footprint and held-out-source evidence.

All preliminary tables live under tables/preliminary and must carry an
explicit VLM-only, non-canonical label. Final generated tables will live under
tables/main and tables/appendix.

## Build and submission checks

Build the full architecture, including appendices, from this directory:

    latexmk -pdf main.tex

Build only the main-paper architecture:

    latexmk -pdf main_only.tex

The generated files are written to build and ignored by Git.

Before submission, run the fail-closed gate:

    python3 scripts/check_submission.py

Also require:

- submission.tex explicitly selects final and main-only modes;
- no preliminary table included by a final section;
- all figure/table paths relative to paper;
- all labels semantic rather than position-dependent;
- all claims regenerated from the final human-canonical result source;
- all baseline names consistent with preamble/macros.tex.
