# Figure 2 specification: method overview

Status: design specification only. Do not generate the figure until Sections
3 and 4 notation is reviewed together.

## Figure job

The figure must make one fact visually unmistakable:

> SRCD changes the identity occupying the registered causal-source slot while
> holding the independently generated source-free target, schedule, and
> stochastic draws fixed.

It explains the treatment. It does not claim that the treatment succeeds or
that the model discovers causality.

## Four-panel layout

### A. Registered event anatomy

- Show one factual event as a short temporal strip.
- Label source, trigger, receiver, and source-dependent footprint.
- Use the same colors for these roles throughout the figure.
- Display the structured factual renderer below the strip.

### B. Dynamic source-free target

- Remove source, trigger, and dependent footprint.
- Retain receiver, scene context, and natural temporal variation.
- Add the explicit label: independently generated; not frame-aligned.
- Show the independent target renderer, registered no-source receiver state,
  generation-only exclusion condition, and frozen target video/latent.
- Distinguish the media-generation condition from the positive target-prompt
  embedding used by the online frozen teacher.

### C. Single-factor source-slot contrast

- Matched row: original source identity s_i.
- SRCD row: assigned source identity q_i.
- Highlight only the source slot.
- Gray-lock receiver, trigger type, footprint semantics, target prompt/video,
  target latent, target-prompt embedding, initialization, active index set,
  sample order,
  noise/sigma draws, optimizer, and inference settings.
- State that q_i comes from a frozen balanced 64-item mechanism bank and is
  different from s_i.

Optional compact arm inset:

| Arm | Causal-source slot | Extra noun placement | Coverage |
| --- | --- | --- | --- |
| Matched | original s_i | none | seven mechanisms |
| Generic Paraphrase | original s_i | none; wording changes | Water/Fracture |
| Bystander Token | original s_i | assigned q_i in noncausal slot | Water/Fracture |
| SRCD | assigned q_i | none | seven mechanisms |

### D. Optimization paths

- Student input: noised target latent plus randomized factual prompt.
- Frozen teacher input: identical noised target latent plus the cached
  source-free-target prompt embedding. The teacher prediction is computed
  online, not cached.
- Mark target-flow matching and frozen-teacher matching as the erasure update.
- Show a separate preservation update on generic non-target prompts.
- Do not draw the two update types as a simultaneous summed minibatch; the
  implementation alternates them.

## Caption contract

The caption must:

- identify the only intended Matched-versus-SRCD difference and the
  100-of-178 active erase schedule;
- state that the counterfactual target is independently generated and
  distributional;
- state that the teacher is frozen;
- mention separate preservation updates;
- avoid claims about efficacy, preservation guarantees, or learned internal
  causality.

## Asset and review gate

- Use vector labels and arrows; video examples may be raster.
- Keep all text readable at single-column width before considering a
  double-column version.
- The final asset receives a generating-script path, input hashes, and a
  metadata record.
- Review the notation against Sections 3 and 4 before generation.
- Review the final caption against the implementation registry and Matched
  equality receipt.
