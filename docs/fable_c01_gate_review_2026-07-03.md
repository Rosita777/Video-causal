# Fable Review: C0.1 Factorial Gate

`claude-fable-5` reviewed the C0.1 seed-matched factorial gate as a method
advisor, not as a video judge.

## Main Attacks

1. The original `weak` label was too subjective and could become an escape
   hatch for hard examples.
2. The relaxed `footprint_only` threshold needs an explicit explanation and
   prompt-validity check, because consequence-without-cause prompts may be
   physically incoherent.
3. Scene drift was too vague as a rejection criterion.
4. A single-reviewer pilot should not directly authorize C1 claims.
5. "All four cells look identical" needs an operational reviewer question or
   metric.

## Accepted Edits

The C0.1 design now:

- replaces `weak` with forced-choice `present`, `absent`, or `uncertain`;
- treats `uncertain` as a failed cell unless adjudicated;
- adds `scene_structure_preserved` as an explicit review field;
- defines cell indistinguishability as a separate rejection test;
- requires structured rejection codes;
- marks single-reviewer passes as provisional unless confirmed by a second
  reviewer;
- adds a `footprint_only` prompt-validity check before scaling.

## Framing Constraints

Avoid saying:

- C0.1 validates causal independence.
- C0.1 tests internal causal reasoning.
- Same-seed matching makes a true counterfactual trajectory.
- Passing items prove target and footprint disentanglement.

Safe phrasing:

- C0.1 tests prompt-conditioned expressibility under matched initial noise.
- Passing items are eligible for C1 intervention testing.
- The gate is a generation-validity screen, not a causal discovery method.

## Implementation Readiness

Fable's verdict: C0.1 is good enough to implement as a 3-item by 5-seed pilot
if the operationalized review fields and rejection codes are included. The
pilot should be described as a process test, not as a validated benchmark.
