# Water-impact dynamic v4 causal-screening preregistration amendment

Version: `v4_dev72_v3`

Preregistered: 2026-08-17

Amended before any v3 graph, secret, Stage-0 wrapper, or media existed:
2026-08-17. The amendment registers the independent graph-assignment salt
that the already-preregistered receiver-slot permutation requires. It does not
change an ontology size, graph topology, prompt, gate, seed rule, or budget.

Scope: causal Stage 0 through causal Stage 1 only

## 1. Status and authority

This amendment starts a new, immutable causal-screening data version. It does
not reopen, extend, retry, or repair `v4_dev72_v2`.

The formal terminal outcome for `v4_dev72_v2` is
`preflight_dataset_invalid`, reason
`causal_screening_cell_quota_infeasible`. The Original-only screen and its
review freeze were valid, but two registered cells contained one eligible
candidate each when four were required. The selector consequently wrote no
selected-24 manifest and no U72 manifest. No eligible checkpoint existed, so
`inconclusive_invalid_evaluation` is not the applicable category.

The only empirical input permitted for this amendment is the public aggregate
in `results/water_impact_dynamic_v4_causal_screening_termination_v2.md`:

| Bound public evidence | Required SHA-256 |
|---|---|
| v2 termination record bytes | `fc6171711a73f4a6eeb30d1f2d005439b7ff7fb7a91d064642fe5da02461ad77` |
| v2 causal Stage-0 wrapper identified by that record | `29696ad8031bb164fe1c6819c8c382d7e4e828835f750f0d245e4877d4167b38` |
| v2 screening freeze identified by that record | `52ac630edd16d234742f4b3cdd75b840e2c2b3a3d2f73f23797fae91ecd59cb5` |

The v3 design validator reads and hashes the termination record itself. It
requires the latter two digest literals to occur in the bound record and this
amendment, but does not open either v2 Stage-0 or private freeze path. Selector
stderr is not a design input.

| Cell | Eligible out of 8 |
|---|---:|
| `holdout_source_new_receiver:direct` | 4 |
| `holdout_source_new_receiver:natural` | 1 |
| `holdout_source_seen_receiver:direct` | 6 |
| `holdout_source_seen_receiver:natural` | 8 |
| `seen_source_new_receiver:direct` | 4 |
| `seen_source_new_receiver:natural` | 1 |

The v3 designer, curators, reviewers, and implementers must not read v2
candidate identities, prompts, media, seeds, row-level reviews, row-level
eligibility, or adjudication content. Sealed-final36 remains unopened. An
isolated set-intersection auditor may compare committed private identifier
sets mechanically, but may release only file digests, counts, and a zero/nonzero
intersection result.

This document authorizes design and later construction of public/private v3
Stage-0 inputs. It does not authorize Original generation until a standard
Stage-0 wrapper has been created and independently validated. It does not
authorize specificity construction, training, checkpoint generation,
treatment generation, final review, or scoring.

## 2. Fixed candidate counts and capacity risk

### 2.1 Analytic planning models

Candidate counts use only the six public binomial aggregates above. The
primary planning model M0 uses a uniform `Beta(1,1)` prior. For a cell with `x`
eligible among the eight v2 candidates, its posterior is `Beta(x+1, 9-x)`.
For a proposed v3 cell size `n`, the M0 posterior-predictive probability of
fewer than four eligible candidates is

```text
q_M0(n, x) = sum over k=0..3 of
             choose(n,k) * B(k+x+1, n-k+9-x) / B(x+1, 9-x).
```

The required M0 familywise cell-shortage risk is

```text
1 - product over cells of (1 - q_M0(n_cell, x_cell)) <= 0.05.
```

The registered sensitivity model M1 replaces the prior with Jeffreys
`Beta(0.5,0.5)` and requires its corresponding familywise risk to be at most
`0.15`. M0 and M1 are planning models, not power calculations,
significance tests, or claims that future eligibility rates are known.

The final graph contains 576 candidates:

| Registered cell | v2 `x/8` | v3 fixed `n` | M0 cell shortage | Jeffreys cell shortage |
|---|---:|---:|---:|---:|
| `holdout_source_new_receiver:direct` (G1-D) | 4/8 | 48 | 0.0013772508 | 0.0020423675 |
| `holdout_source_new_receiver:natural` (G1-N) | 1/8 | 168 | 0.0213109327 | 0.0567120666 |
| `holdout_source_seen_receiver:direct` (G2-D) | 6/8 | 24 | 0.0008142951 | 0.0007782908 |
| `holdout_source_seen_receiver:natural` (G2-N) | 8/8 | 24 | 0.0000057043 | 0.0000022088 |
| `seen_source_new_receiver:direct` (G3-D) | 4/8 | 96 | 0.0000659546 | 0.0001283477 |
| `seen_source_new_receiver:natural` (G3-N) | 1/8 | 216 | 0.0134069232 | 0.0400662581 |

The M0 familywise cell-shortage risk is `0.0366161849`; the M1 Jeffreys
sensitivity is `0.0971766186`. Both pass their preregistered limits.

### 2.2 Structural corrections and rejected smaller designs

The 272-candidate proposal `24/96/16/16/24/96` has an M0 cell-shortage risk of
`0.1524362009`. Its small plug-in estimate treats six rates from eight trials
as known and is rejected. Its eight artificial G1 blocks also confuse a data
layout with a physical independence unit.

A subsequent 480-candidate proposal passed the M0 (`0.0455874987`) and
Jeffreys (`0.1145303753`) cell-only checks, but failed the preregistered
two-level anchor-correlation check. Under the exact M0 posterior plus
Beta-binomial anchor ICC `rho=0.10`, an initial public-seed run estimated
global selector failure above 30 percent. It is therefore rejected; passing a
cell-total check cannot excuse empty physical anchors.

The scientific correction defines G1's 24 fresh heads as its physical anchors
and permits any eight distinct heads, whereas G3 retains the eight original
source anchors and must select each exactly once. The G3 bottleneck is enlarged
to 12 direct and 27 natural edges per original source. G1 remains two direct
plus seven natural edges per fresh head; G2 remains three heads and one edge
per variant per historical-receiver anchor. This gives `216 + 48 + 312 = 576`
candidates without changing a prompt, rubric, threshold, or final U72 size.

### 2.3 M2 frozen Monte Carlo capacity gate

M2 is the exact, public-only correlated planning simulation. Its binding gate
is the one-sided 95% Monte Carlo upper bound at `rho=0.10 <= 0.15`;
`rho=0.20` is a report-only sensitivity and cannot authorize or veto a design.

The simulation is:

1. For every iteration and cell `c`, draw
   `p_c ~ Beta(x_c+1, 9-x_c)`.
2. For every physical anchor `a` in cell `c`, draw
   `theta_c,a ~ Beta(p_c*kappa, (1-p_c)*kappa)`, where
   `kappa=(1-rho)/rho`. Physical-anchor counts in cell order
   `G1-D,G1-N,G2-D,G2-N,G3-D,G3-N` are `24,24,8,8,8,8`.
3. Draw every candidate-edge eligibility independently conditional on its
   `theta_c,a`.
4. Run the exact completion oracle in Section 7, including distinct-head,
   direct/natural, source, receiver, and perfect-matching constraints. A rank
   is irrelevant to feasibility and is not simulated.

Search, confirmation, `rho=0.20`, and shared-frailty runs all use the exact
R1/R3 incidence formulas in Section 4.2. The later salted permutation from
receiver slots to identities is graph-isomorphic and cannot change a result.

The search lattice starts at the 576 graph. If it fails, the only permitted
capacity changes are preregistered G3 steps
`(D,N)=(12+3j,27+3j)` per original anchor for integer `j>=1`; G1, G2,
templates, rubric, and thresholds remain fixed. Test lattice points in order
and stop at the first passing point. A failed final confirmation cannot be
used to test the next point with the same confirmation seed; it requires a
new amendment and independently committed seed.

The search run uses NumPy `2.4.6`, `PCG64`, 200,000 iterations, batches of
5,000, and seed
`uint64(first_8_bytes(SHA256("water-impact-dynamic-v4-dev72-v3-capacity-search-v1\n")), big_endian)` =
`12169367837932875788`; the seed-string SHA-256 is
`a8e24792910d700ced6dff45d9817be05fe5370e2e102cf6e0363ed5e8244580`.
For `rho=0.10`, the exact 576-graph search result was 28,527 failures
(`0.1426350000`), with one-sided 95% Wilson upper bound `0.1439260337`.
Because the first lattice point passed the search ceiling `0.145`, no larger
point was inspected.

Final confirmation uses one million iterations and a disjoint seed for each
reported ICC. The `rho=0.10` domain is
`water-impact-dynamic-v4-dev72-v3-anchor-risk-mc-confirm-rho010-v1\n`;
its SHA-256 is
`8b6015284b8d55a49246904746fb115437b7a04e818a471942a41cc7f4c2fe8d`
and its uint64 seed is `10043050431846634916`. The result was:

| Quantity | Failures | Rate |
|---|---:|---:|
| G1 exact completion | 42,316 | 0.042316 |
| G2 exact completion | 8,292 | 0.008292 |
| G3 exact completion | 98,317 | 0.098317 |
| global exact selector | 143,547 | 0.143547 |

The one-sided 95% Wilson upper bound for global failure is `0.1441246991`,
which passes the required `<=0.15` gate. Cell-total shortage counts in the
same run, in the fixed six-cell order, were
`1,566 / 25,794 / 1,150 / 9 / 197 / 25,641`. The gap between cell shortages
and group failures is why the exact completion simulation is mandatory.

The report-only `rho=0.20` domain SHA-256 is
`8a85d1c114b988dddb75670fc322a0beee98403c9f61468b48205c0a581d6aad`
(uint64 seed `9981614776343169245`). It produced 264,002 global failures,
rate `0.2640020000`, and Wilson upper `0.2647276898`. This sensitivity does not
change the preregistered `rho=0.10` gate and is not permission to tune capacity.

Because increasing within-cell ICC does not itself correlate direct and
natural outcomes at the same physical anchor, M2 also requires a report-only
shared-frailty sensitivity. Start from the `rho=0.10` model. After drawing all
cell-specific anchor probabilities, draw one
`F_a ~ Beta(9,1)/0.9` for every physical anchor and multiply both its direct
and natural probabilities by the same `F_a`, clipping each at one. Draw
frailties in group order G1/G2/G3 after all six theta arrays and before any
edge uniforms. The domain
`water-impact-dynamic-v4-dev72-v3-anchor-risk-mc-confirm-shared-frailty-v1\n`
has SHA-256
`3027d87c835f977849a031eb2a530438c33378d91eba80137aaa88709ec7f1ed`
and uint64 seed `3469980067203880824`. In one million exact-graph iterations it
produced marginal G1/G2/G3 failure counts `42,565 / 13,251 / 99,458` and
149,245 global failures: rate `0.1492450000`, Wilson upper `0.1498320593`.
This sensitivity is mandatory to report but is not a second authorization
gate and cannot be used to change the preregistered first-passing 576 lattice
point.

M0, M1, and M2 do not model a systematic failure tied to the single public
screening seed, correlated reviewer error, ontology misspecification, model or
runtime failure, or provenance failure. Their percentages are registered
working-model resource risks, not the true probability that v3 will succeed.

For `F` failures in `N` iterations, the one-sided Wilson bound uses
`z=1.6448536269514722` and exactly

```text
(F/N + z^2/(2N) + z*sqrt((F/N)*(1-F/N)/N + z^2/(4N^2))) /
(1 + z^2/N).
```

Within each seed stream, use `numpy.random.Generator(PCG64(seed))`. For a batch
of size `B` (5,000 except the final partial batch), make exactly one call
`rng.beta(x+1,9-x,size=(B,6))` for `p`. Then, once per cell in fixed order,
call
`rng.beta(p[:,c,None]*kappa,(1-p[:,c,None])*kappa,size=(B,A_c))`, where
`A_c=24,24,8,8,8,8`. Finally, once per `(cell,anchor)` in fixed order, call
`rng.random((B,E_c,a))` for that anchor's edge uniforms, with edges in the
canonical graph order. The shared-frailty stream inserts exactly three calls
`rng.beta(9,1,size=(B,A_group))`, in G1/G2/G3 order, after all six theta calls
and before the first edge-uniform call. No vectorized call may be split,
combined, reordered, or skipped. No simulation output may change the frozen
graph after the independent confirmation.

## 3. Fixed resource budget

All 576 candidates use Wan 2.1 T2V 1.3B, 25 denoising steps, CFG 5, 49 frames,
832x480, 8 fps, and bf16. Therefore the exact screening workload is:

| Resource | Fixed amount |
|---|---:|
| Original videos | 576 |
| generated frames | 28,224 |
| video-denoising steps | 14,400 |
| pixel-frames | 11,271,536,640 |
| workload relative to v2 screening | 12.0x |
| A/B full-video review assignments | 1,152 |
| maximum atomic adjudications | 2,880 |

Before Stage-0 authorization, five public, non-v2 calibration prompts must be
run with the exact frozen model, runtime, and render configuration. The public
calibration artifact must bind hardware identity, all five prompt hashes,
per-render wall times, maximum wall time, and the generation configuration.
The maximum must not exceed 600 seconds. The screening allocation is capped at
100 A100-GPU-hours: 96.0 hours for 576 renders at the ten-minute cap plus the
calibration and orchestration allowance. This is a GPU-time budget, not
authorization for any worker count or sharding layout. Calibration failure
blocks Stage 0; after Stage 0, any timeout or failed render invalidates v3 and
cannot be rerun.

Human staffing is budgeted, not used as a scoring timeout:

- 1,152 independent A/B reviews at 90 seconds each: 28.8 reviewer-hours;
- at most 2,880 third-review atomic decisions at 45 seconds each: 36.0 hours;
- double review of 48 evaluation-holdout source, 56 new-receiver, and eight
  historical-receiver anchor records at three minutes per record: 11.2 hours;
- package, seed, hash, and freeze audit allowance: 4.0 hours.

The conservative staffing envelope is 80.0 human-hours. A reviewer may take
longer; the protocol must never truncate or synthesize a decision to meet the
budget.

## 4. New ontologies and anchor construction

All v3 ontologies and schedules are frozen before any Original render.

### 4.1 Frozen training upstream and fresh evaluation sources

V3 refreshes evaluation only. The training bank and mapping remain the exact
v2 bytes and must not be regenerated:

| Upstream artifact | Required SHA-256 |
|---|---|
| `data/water_impact_dynamic_v4/source_bank_public64_registry_v2.json` | `473af632f8100e9e7c46c35e5fd679c9729bc80d19af12aaa78a1a0c69c9f814` |
| `data/water_impact_dynamic_v4/source_mapping_v2.json` | `6988d19ef759b6fd7c15543a5b1774656a20dace82ea82fe5812004f763bb4c2` |

There is no v3 source-bank permutation, training mapping, cache, or training
sidecar in this amendment.

Create exactly 48 fresh evaluation-holdout source identities under the
unchanged strict physical audit. All 48 have distinct normalized head lemmas
and are semantically and lexically disjoint from v1/v2 sources, both frozen
training-bank identities, historical evaluation sources, all receivers, and
event/mechanism vocabulary. The original eight training sources are the only
intentional historical-source exception and remain subject to the complete
49-frame Original screen.

The candidate graph has these physical anchors:

- G1, `holdout_source_new_receiver`: its 24 fresh holdout heads are the
  physical anchors. For each head create two direct and seven natural edges.
  G1 has 48 direct plus 168 natural candidates. Stage 1 may select any eight
  different heads, at most one case per head.
- G2, `holdout_source_seen_receiver`: assign three different fresh holdout
  heads to every historical-receiver anchor. For each head create one direct
  and one natural edge. Each anchor therefore has three direct and three
  natural candidates, and G2 has 24 direct plus 24 natural candidates.
- G3, `seen_source_new_receiver`: anchor `i` uses original training source
  `i`. Connect it to 12 direct and 27 natural receiver edges. G3 has 96 direct
  plus 216 natural candidates and Stage 1 must select every anchor exactly
  once.

The 24 G1 heads and 24 G2 heads are mutually exclusive. Within a group, head
sets are also disjoint across G2 anchors. Selecting eight different G1 heads,
one row from every G2 anchor, and one row from every G3 anchor yields 16
different holdout heads and all eight original sources without inspecting an
eligibility result. The complete 576-edge graph, including anchor and edge
ordinals, is frozen before generation and cannot be edited.

### 4.2 Receivers

Create exactly 56 new receiver identities. They must be disjoint by normalized
phrase, head lemma, and semantic-equivalence review from every training,
historical, v1, v2, source, and mechanism identity. Partition them into
mutually disjoint R1 and R3 pools of 24 and 32 identities. Actual receiver
identities are assigned to the public index graphs below by independent salted
permutations. The sole salt for both pool permutations is the independently
sampled `causal_graph_assignment_salt_v3`; R1 and R3 remain domain-separated
by their exact pool labels. The graph topology itself is not searched against
outcomes.

For a receiver in pool `P`, compute

```text
sha256(utf8("causal-graph-receiver-permutation-v3") || NUL ||
       utf8(graph_assignment_salt) || NUL || utf8(P) || NUL ||
       utf8(receiver_id)).
```

Sort each pool by this digest and map that order to its numbered receiver
slots. A digest tie invalidates the data version; receiver ID is not a
tie-breaker. The graph binds the salt-file SHA-256 and the canonical digest of
each ordered pool without revealing the salt.

For R1, number heads and receiver slots `0..23`. Head `t` has natural
neighbors `(t+o) mod 24` for offsets `o={0,3,7,11,15,19,22}` and direct
neighbors for offsets `o={0,11}`. Direct is a subset of natural; source-side
degree is 2/7 and receiver-side degree is exactly 2/7.

For R3, number receiver slots `0..31`. Original-source anchor `a` has direct
neighbors

```text
{4*((a+j) mod 8)+k | j in {1,2,3}, k in {0,1,2,3}}
```

and natural neighbors equal to all 32 slots except

```text
{4*a,4*a+1,4*a+2,4*a+3,
 4*((a+4) mod 8)+(a mod 4)}.
```

Thus G3 source-side degree is 12/27, receiver-side direct degree is exactly
3, receiver-side natural degree is 6 or 7, and direct is a subset of natural.
Every `(source_id, receiver_id, prompt_variant)` triple is unique.

Before Stage 0, exhaustively verify that the unlabelled R1 graph retains an
eight-distinct-head, four-direct/four-natural receiver matching after deleting
any zero, one, or two receiver nodes. For R3, exhaustively verify all
`choose(8,4)=70` four-direct/four-natural anchor assignments and the same
zero-to-two-receiver deletion condition. These static checks precede and are
separate from the correlated eligibility Monte Carlo in Section 2.3. The
reference topology passes all 301 R1 deletion sets and all `70 x 529 = 37,030`
R3 assignment/deletion combinations.

For G2, freeze exactly eight mutually distinct historical receiver identities,
one per anchor. They must be present in the historical training receiver
inventory. These eight historical anchor nodes and the original eight G3
source nodes are the only identity-level historical exceptions. Each G2
anchor's six candidates use its one historical receiver with its three fresh
heads and two variants. G2 anchors are disjoint, so the selected eight
historical receivers are necessarily unique.

Every candidate receives a new `case_id` with the v3 namespace. No v1/v2 case
ID, canonical candidate record, source-receiver pair, or
`(source_id,receiver_id,prompt_variant)` triple may be reused. Fresh R1/R3
receiver identities may not reuse a v1/v2 receiver ID. The eight registered
G2 historical receiver anchors may reuse their receiver nodes, but every v3
pair, triple, and case around them must be new. An isolated disjointness report
must bind both old and new set digests and report these exact exception and
zero-intersection claims without revealing either set.

## 5. Secrets, domains, and seed audit

All raw secrets are newly sampled after the v3 ontology and candidate-builder
code are frozen. They are lower-case hexadecimal strings of 64 characters
except the screening seed, which is a canonical unsigned decimal uint32.
They must be pairwise different and different from every v1/v2 causal,
specificity, holdout-assignment, source-mapping, and review-assignment secret.

The exact v3 domains are:

| Purpose | Exact domain/namespace |
|---|---|
| graph receiver permutation | `causal-graph-receiver-permutation-v3` |
| rank | `causal-selector-v3` |
| evaluation seed | `causal-eval-seed-v3` |
| screening namespace | `v4-causal-stage0-screening-v3` |
| evaluation namespace | `v4-causal-evaluation-v3` |
| screening commitment name | `causal_screening_seed_v3` |
| graph-assignment commitment name | `causal_graph_assignment_salt_v3` |
| selector commitment name | `causal_stage0_selector_salt_v3` |
| evaluation-salt commitment name | `causal_evaluation_seed_salt_v3` |

Rank canonical candidate bytes with

```text
sha256(utf8("causal-selector-v3") || NUL || utf8(selector_salt) ||
       NUL || canonical_candidate_record_bytes).
```

Derive each candidate evaluation seed with

```text
uint32(first_4_bytes(
  sha256(utf8("causal-eval-seed-v3") || NUL || utf8(evaluation_salt) ||
         NUL || utf8(case_id) || NUL || utf8(decimal_replicate))),
  big_endian).
```

Replicates are exactly `0,1,2`. Before Stage-0 authorization, derive all
`576 x 3 = 1,728` seeds. They must be pairwise unique and disjoint from the v3
screening seed and the independently frozen forbidden-seed inventory, which
must cover training, historical evaluation, v1, v2, specificity, and every
other registered screening namespace. Bind the ordered `(case_id, replicate,
seed)` inventory through its canonical digest and publish only digest and
count. Stage 1 may use only the precommitted 72-row subset corresponding to
the selected 24 cases.

## 6. Original-only screening and qualification

The prompt and qualification constructs are byte-frozen from v2. The v3
canonical-template file must be byte-for-byte equal to the `canonical_templates`
artifact committed by causal Stage 0 v2, SHA-256
`76d3b2be61389a26cc5feb9b1211c5e7b0830a85369e27783fb56e5286ce0559`.
The field-normalization artifact must also be byte-identical, SHA-256
`a1e23230b199a96e9f458c135e6ce2d18bf377966a6867dc5a2cca88d124e2ce`.
The v3 selection-rules file adds the preregistered
anchor graph, but its canonical `qualification` object and its requirement of
exactly four qualified cases per `group x prompt_variant` cell must equal the
corresponding v2 canonical JSON bytes. Any prompt, fill rule,
non-substitution rule, rubric threshold, or cell-quota change invalidates v3
before generation; changing one would use the v2 outcome to change the
estimand. Equivalence is supplied only by the isolated construct auditor in
Section 8.4; v3 runtime code and the Stage-0 authorizer never open these v2
private files.

1. Generate exactly one Original video for every one of the 576 candidates
   with the frozen screening seed, full-model inventory, runtime, and exact
   configuration: 25 steps, CFG 5, 49 frames, width 832, height 480, 8 fps,
   bf16, base model, `adapter=null`, no persistent gate, and no target or
   suppression phrase.
2. There is no skip, rerender, replacement prompt, alternate seed, repair, or
   reserve. All 576 videos must decode to 49 nonempty frames and match the
   exact raw manifest before a review package can be committed.
3. Before reviews start, bind raw videos, anonymous full videos, composites,
   candidate aliases, review order, model/runtime/code registries, generation
   manifest, and every byte hash in a public/private screening-package
   commitment. Reviewers receive all 49 frames, not only a composite.
4. Two screening reviewers independently score source visibility, footprint
   visibility, receiver, quality, and `causal_link`. They cannot be final
   treatment reviewers and cannot access candidate identities or prior
   reviews.
5. Derive the atomic dispute manifest exactly from the two frozen review
   sheets. A third reviewer adjudicates every and only disputed atomic field.
   Exact CSV header order, unique row keys, complete coverage, and immutable
   media hashes are mandatory.
6. A candidate is eligible exactly when source visibility is `2`, footprint
   visibility is at least `1`, receiver is at least `1`, quality is at least
   `1`, and `causal_link` is `2`. Unusable or undecodable media are ineligible;
   they are never regenerated.

## 7. Deterministic Stage-1 selection

The selector opens only the frozen Stage-0 wrapper and the hash-bound screening
freeze. It must reconstruct every candidate row from the Stage-0 ontology,
template, normalization, and candidate manifest before using eligibility.

Select eight eligible cases per group and 24 total. Within every group,
exactly four selected cases are direct and four natural. G1 selects eight of
its 24 heads with at most one case per head. G2 and G3 select exactly one case
from each of their eight anchors. All groups require distinct receivers. The
graph construction then guarantees the unchanged global constraints:

- 16 distinct holdout head lemmas across the two holdout groups;
- every original training source exactly once in
  `seen_source_new_receiver`;
- eight different historical receivers in
  `holdout_source_seen_receiver`;
- 16 different new receivers across the two new-receiver groups;
- all 24 selected receiver identities unique.

For every qualified candidate, recompute the v3 rank from canonical raw bytes.
Equal ranks invalidate the version. Sort all candidates by rank and scan once.
At a candidate, tentatively force it into the selected set. Include it if and
only if an exact completion still exists; otherwise permanently exclude it.
Stop after 24 forced inclusions and verify the complete constraints again. If
the initial instance or any forced prefix has no completion, the version is
invalid. Candidate IDs are not tie-breakers. There is no reserve queue.

The completion oracle is exact, not heuristic:

- G1 solves the finite 0-1 edge system with eight selected edges, four direct,
  four natural, head degree at most one, receiver degree at most one, and all
  forced/excluded edges respected;
- G2 enumerates the 70 four-direct/four-natural assignments of its eight
  historical-receiver anchors and requires an eligible edge of the assigned
  variant for every anchor. It respects the greedy prefix exactly: at most one
  candidate may be forced in an anchor; a forced candidate locks that exact
  anchor, variant, head, and row; every excluded candidate is unavailable;
  and an unforced anchor may use only a nonexcluded eligible row of its assigned
  variant;
- G3 enumerates the same 70 assignments and runs an eight-node bipartite
  perfect matching over eligible R3 edges, respecting forced candidates and
  receiver uniqueness.

G1, G2, and G3 use disjoint head/source/receiver pools, so a global completion
exists exactly when all three group oracles succeed with their forced prefixes.
The G1 0-1 system may be implemented by exhaustive branch-and-bound or an
integer-feasibility solver, but it must use integer/exact comparisons, a
deterministic variable order, and independently recompute the returned
certificate. A solver timeout is failure, never permission to exclude an edge.

Proof sketch: let `F_i` be the forced prefix after rank position `i`. If a
candidate can occur in any feasible completion of `F_i`, including it makes
the next rank in the selected tuple as small as possible. If it cannot, every
feasible completion excludes it. Induction over the sorted ranks therefore
produces the lexicographically smallest feasible complete rank tuple. The
argument requires exact completion answers and fails if a rank tie is allowed,
which is why ties are terminal.

Implementation tests must cover rank ties; forced include/exclude boundaries;
G1 repeated-head and receiver collisions; G2 forced-row, forced-variant,
same-anchor double-force, and excluded-last-row boundaries; G2/G3 missing
anchors; wrong 4/4
variant counts; graph-degree or edge tampering; all 70 G2/G3 assignments;
zero-, one-, and two-receiver deletion robustness; agreement between the G1
oracle and brute-force enumeration on reduced graphs; and equality of greedy
output with brute-force lexicographic selection on bounded synthetic fixtures.

After selection, copy the already committed three seeds for each selected
case to the U manifest. Require 24 unique case IDs, exactly three replicates
per case, 72 unique uint32 seeds, exact pair-ID binding, and a second
forbidden/screening disjointness check. U remains 24 semantic clusters and 72
generation units; uncertainty uses the semantic case as the clustering unit,
exactly as in v2.

## 8. Exact versioned paths and registry schemas

### 8.1 Public repository paths

The v3 implementation must use these exact evaluation paths. The two explicitly
listed v2 upstream files are required byte-for-byte inputs; every other v1/v2
alias is forbidden:

```text
docs/water_impact_dynamic_v4_dev72_v3_preregistration.md
results/water_impact_dynamic_v4_causal_screening_termination_v2.md
data/water_impact_dynamic_v4/source_bank_public64_registry_v2.json
data/water_impact_dynamic_v4/source_mapping_v2.json
data/water_impact_dynamic_v4/holdout_public_commitment_v3.json
data/water_impact_dynamic_v4/causal_stage0_public_commitment_v3.json
data/water_impact_dynamic_v4/causal_stage0_commitment_v3.json
data/water_impact_dynamic_v4/causal_stage1_commitment_v3.json
data/water_impact_dynamic_v4/causal_preflight_dataset_invalid_v3.json
data/water_impact_dynamic_v4/v4_model_content_inventory_v3.json
data/water_impact_dynamic_v4/v4_runtime_registry_v3.json
data/water_impact_dynamic_v4/v4_eval_code_registry_v3.json
data/water_impact_dynamic_v4/v4_screening_cost_calibration_v3.json
data/water_impact_dynamic_v4/v4_causal_capacity_model_v3.json
data/water_impact_dynamic_v4/v4_causal_capacity_search_v3.json
data/water_impact_dynamic_v4/v4_causal_capacity_confirm_v3.json
data/water_impact_dynamic_v4/v4_causal_static_graph_audit_v3.json
data/water_impact_dynamic_v4/v4_causal_identity_disjointness_v3.json
data/water_impact_dynamic_v4/v4_causal_v2_construct_equivalence_v3.json
```

The invalid-outcome path and Stage-1 path are mutually exclusive. A successful
version has a Stage-1 wrapper and no invalid-outcome record. A failed version
has an invalid-outcome record and no Stage-1 wrapper.

The invalid-outcome JSON uses protocol
`water_impact_dynamic_v4_preflight_dataset_outcome_v3` and has exactly these
top-level keys:

```text
protocol, dataset, dataset_version, status, failure_phase, reason_code,
stage0_registry_sha256, candidate_count, eligible_count,
cell_eligible_counts, selector_output_created, unit_manifest_created,
stage1_registry_created, sealed_final36_status, bound_artifacts
```

Required constants are `dataset="causal"`,
`dataset_version="v4_dev72_v3"`, `status="preflight_dataset_invalid"`, all
three `*_created` fields `false`, and `sealed_final36_status="unopened"`.
If a terminal scientific-integrity failure occurs before the standard Stage-0
wrapper exists, `stage0_registry_sha256` is null. For every later failure it is
the lower-hex64 byte hash of the exact standard wrapper; no self-reported or
pending-commitment hash is accepted. `bound_artifacts.stage0_registry` must be
the same value or the same null.
`failure_phase` is exactly one of
`stage0_authorization,original_generation,screening_package,screening_review,screening_freeze,selection,stage1_publication`.
`reason_code` is exactly one of:

```text
stage0_authorization_integrity_failure
screening_generation_incomplete
screening_package_integrity_failure
screening_review_coverage_failure
screening_adjudication_integrity_failure
screening_cell_quota_infeasible
screening_anchor_coverage_infeasible
selection_rank_tie
global_subset_infeasible
seed_contract_failure
stage1_publication_failure
```

`candidate_count` is 576. `eligible_count` and the six-key
`cell_eligible_counts` object are nonnegative integers only after a canonical
freeze exists and otherwise null. `bound_artifacts` has exactly
`stage0_registry,screening_generation_manifest,screening_package_commitment,screening_freeze_manifest,canonical_eligibility,selector_stderr`;
each value is lower hex64 or null when that phase was never reached. No free
text, path, identity, prompt, seed, score, media label, or traceback field is
permitted.

### 8.2 Private basenames

All private files live below one new, mode-700, non-symlink `PRIVATE_V3_ROOT`.
Every private directory is mode 700. Every private regular file is mode 600,
is not a symlink, has link count one, and resolves below that root. No private
hardlink is permitted. No path may be below or resolve through a v1/v2,
sealed, quarantine, or final36 root. Required Stage-0 basenames are:

```text
eval_holdout_source_ontology_private48_v3.json
holdout_registry_private48_v3.json
receiver_ontology_private56_v3.json
historical_receiver_anchors_private8_v3.json
causal_stage0_candidates_private576_v3.json
causal_stage0_candidate_graph_private576_v3.json
causal_stage0_templates_private_v3.json
causal_stage0_field_rules_private_v3.json
causal_stage0_render_config_private_v3.json
causal_stage0_selection_rules_private_v3.json
causal_stage0_secrets_private_v3.json
causal_stage0_bundle_private_v3.json
causal_screening_seed_v3.txt
causal_graph_assignment_salt_v3.txt
causal_stage0_selector_salt_v3.txt
causal_evaluation_seed_salt_v3.txt
causal_forbidden_seed_inventory_v3.json
causal_preselection_seed_audit_1728_v3.json
causal_generation_spec_v3.json
causal_selection_binding_v3.json
```

Private JSON schemas are exact and versioned. The evaluation-holdout ontology
has top-level keys
`protocol,dataset_version,source_count,sources,curation_audit,disjointness_commitment`;
`source_count=48`. A source row has exactly
`source_id,source_phrase,normalized_phrase,head_lemma,origin,food_status,shape_class,color_family,material_family,texture_class,impact_plausibility,physical_audit_status,curator,curation_stratum,group_pool,head_ordinal`.
Every row has `physical_audit_status="strict_physical_pass_v3"`.

The receiver ontology has top-level keys
`protocol,dataset_version,receiver_count,pools,receivers,curation_audit,disjointness_commitment`;
`receiver_count=56` and `pools={"R1":24,"R3":32}`. A receiver row has exactly
`receiver_id,receiver_phrase,normalized_phrase,head_lemma,receiver_type,pool,receiver_ordinal,curator_note,curator`.
The historical-anchor file has top-level keys
`protocol,dataset_version,anchor_count,training_receiver_inventory_sha256,v2_disjointness_commitment,anchors`;
`anchor_count=8`. Each anchor row has exactly
`anchor_id,receiver_id,receiver_phrase,normalized_phrase,head_lemma,historical_training_binding_sha256`.

The candidate graph has top-level keys
`protocol,dataset_version,status,candidate_count,cell_counts,topology,graph_assignment_salt_sha256,r1,r3,anchors,edges,graph_sha256`.
`graph_assignment_salt_sha256` is the SHA-256 of the exact lower-hex64 salt
file including its single trailing LF. The `r1` and `r3` objects each have
exactly `pool,receiver_count,receiver_ids,permutation_sha256`, where
`permutation_sha256` hashes the ordered `receiver_ids` JSON array encoded with
`json.dumps(ids, ensure_ascii=True, separators=(",", ":"))`, followed by one
LF byte, then encoded as ASCII. Array order is preserved; there is no sorting,
indentation, additional whitespace, alternate escaping, or platform newline.
Its 576 edge rows have exactly
`case_id,group,prompt_variant,physical_anchor_id,edge_ordinal,source_membership,source_id,source_phrase,source_head_lemma,receiver_membership,receiver_id,receiver_phrase,canonical_prompt,canonical_record_sha256`.
The separate candidate manifest is an exact order-preserving projection of
those rows; no alias may change a value. All top-level inventories and row-key
sets are allowlists, not minimum schemas. `graph_sha256` is computed over the
canonical graph with that self-hash field removed.

Generation, review, and selection outputs use exclusive directories:

```text
PRIVATE_V3_ROOT/causal_original_screening_generation_v3/
PRIVATE_V3_ROOT/causal_original_screening_review_private_v3/
PRIVATE_V3_ROOT/causal_original_screening_review_public_v3/
PRIVATE_V3_ROOT/causal_original_screening_blind_inputs_v3/
PRIVATE_V3_ROOT/causal_stage1_execution_v3/freeze/
PRIVATE_V3_ROOT/causal_stage1_execution_v3/selector/
```

Directories and public wrappers are created exclusively and atomically. An
existing target is a fatal error.

### 8.3 Stage registries

Use protocol `water_impact_dynamic_v4_eval_commitment_registry_v3` and dataset
version `v4_dev72_v3`. A Stage-0 wrapper has exactly these top-level keys:

```text
protocol, dataset, dataset_version, stage, status,
sealed_final36_status, artifacts
```

Stage 1 adds exactly `stage0_registry_sha256`. Required values are
`dataset="causal"`, `status="committed"`, and
`sealed_final36_status="unopened"`. Every artifact record is exactly
`{sha256, size_bytes, row_count}`; hashes are lower hex64, sizes are positive,
and `row_count` is either the exact registered count or null. Stage 1 must bind
the current Stage-0 wrapper byte hash.

The Stage-0 registry contains exactly these semantic artifacts:

| Artifact | Row count |
|---|---:|
| `candidate_manifest_576` | 576 |
| `upstream_source_bank_registry_64_v2` | 64 |
| `upstream_source_mapping_178_v2` | 178 |
| `eval_holdout_source_ontology_48` | 48 |
| `holdout_registry_48` | 48 |
| `receiver_ontology_56` | 56 |
| `historical_receiver_anchors_8` | 8 |
| `candidate_graph_576` | 576 |
| `canonical_templates` | null |
| `field_normalization` | null |
| `raw_root_bundle` | null |
| `raw_render_configuration` | null |
| `stage0_secrets` | null |
| `screening_seed` | null |
| `graph_assignment_salt` | null |
| `screening_generation_spec` | null |
| `selector_salt` | null |
| `ranking_formula` | null |
| `constrained_subset_algorithm` | null |
| `evaluation_seed_salt` | null |
| `seed_derivation_formula` | null |
| `forbidden_seed_inventory` | variable, positive |
| `preselection_seed_audit_1728` | 1,728 |
| `selection_binding` | null |
| `model_content_inventory` | null |
| `runtime_registry` | null |
| `eval_code_registry` | null |
| `screening_cost_calibration` | 5 |
| `capacity_model_spec` | null |
| `capacity_search_result_200000` | null |
| `capacity_confirm_result_1000000` | null |
| `static_graph_robustness_report` | null |
| `identity_disjointness_report` | null |
| `v2_construct_equivalence_report` | null |
| `preregistration` | null |
| `v2_public_aggregate_design_input` | 6 |

The pending public Stage-0 commitment uses protocol
`water_impact_dynamic_v4_causal_stage0_public_commitment_v3` and exactly these
top-level keys:

```text
protocol, schema, registry, dataset_version, stage, status,
authorization_status, candidate_count, cell_counts, sizing_rule,
design_input, curation_audit, public_metadata, component_commitments,
remaining_blockers
```

It must record candidate count 576, the six exact cell counts, both analytic
risks, both ICC thresholds, and exact references to the search and confirmation
artifacts in Section 2, plus the byte hashes of this preregistration and the v2
public termination record. It contains digests/counts only and
no private identity, prompt, seed, salt, or review value. A separate standard
Stage-0 authorizer opens and validates every private component, identity-set
intersection result, cost calibration, model/runtime/code registry, secret
commitment, forbidden inventory, and 1,728-seed audit before exclusively
writing `causal_stage0_commitment_v3.json`.

The pending commitment and private selection binding both bind the
graph-assignment salt-file commitment. The authorizer reopens that salt,
recomputes both receiver permutations and their digests, reconstructs every
candidate edge, and requires exact equality with the committed graph. The
graph-assignment, selector, evaluation-seed, and screening secrets are all
pairwise distinct; substituting one for another is fatal.

The Stage-1 registry contains exactly these semantic artifacts:

| Artifact | Row count |
|---|---:|
| `screening_generation_manifest_576` | 576 |
| `screening_raw_video_inventory_576` | 576 |
| `screening_candidate_binding_576` | 576 |
| `screening_anonymous_video_inventory_576` | 576 |
| `screening_composite_inventory_576` | 576 |
| `screening_public_package_manifest_576` | 576 |
| `screening_private_package_manifest_576` | 576 |
| `screening_package_commitment` | null |
| `screening_review_template_576` | 576 |
| `screening_review_a_576` | 576 |
| `screening_review_b_576` | 576 |
| `screening_dispute_manifest` | 0..2,880 |
| `screening_adjudication` | equal to disputes |
| `screening_adjudication_audit` | equal to disputes |
| `screening_freeze_manifest` | null |
| `eligibility_table_576` | 576 |
| `selector_summary` | null |
| `selected_case_manifest_24` | 24 |
| `unit_manifest_U_72` | 72 |

`selector_summary` has exactly these top-level keys:

```text
protocol, dataset_version, status, candidate_count, eligible_count,
cell_eligible_counts, selected_count, unit_count, stage0_registry_sha256,
screening_freeze_sha256, eligibility_table_sha256,
selected_case_manifest_sha256, unit_manifest_sha256,
selection_rank_tuple_sha256, constraints
```

It requires protocol `water_impact_dynamic_v4_causal_selector_summary_v3`,
status `selected`, counts `576/24/72`, and lower-hex64 hashes. `constraints`
has exactly
`cell_quota_pass,g1_distinct_head_pass,g2_anchor_coverage_pass,g3_anchor_coverage_pass,original_source_coverage_pass,holdout_head_uniqueness_pass,receiver_uniqueness_pass,rank_tie_free,seed_contract_pass`,
all boolean true. The summary contains no candidate identity, prompt, seed,
rank value, or row score; only `selected_case_manifest_24` contains 24 selected
rows.

The Stage-1 builder must reopen and recompute the Stage-0 ontology, candidate
projection, package, full-video hashes, A/B reviews, exact dispute set,
adjudication, canonical eligibility, ranks, anchor/quota constraints, globally
lexicographic subset, and all selected seeds. Hash agreement alone is not
sufficient.

### 8.4 Independent implementation boundary

V3 must be implemented through independent, hard-bound entry points. These
exact paths are preregistered; a wrapper that forwards to a v2 entry point is
not an implementation:

```text
scripts/water_impact_dynamic_v4_eval_protocol_v3.py
scripts/build_water_impact_dynamic_v4_causal_candidates_v3.py
scripts/authorize_water_impact_dynamic_v4_causal_stage0_v3.py
scripts/run_water_impact_dynamic_v4_causal_screening_v3.py
scripts/freeze_water_impact_dynamic_v4_causal_screening_v3.py
scripts/select_water_impact_dynamic_v4_causal_v3.py
scripts/validate_water_impact_dynamic_v4_causal_v3.py
scripts/validate_water_impact_dynamic_v4_causal_capacity_v3.py
scripts/audit_water_impact_dynamic_v4_v3_v2_disjointness.py
scripts/audit_water_impact_dynamic_v4_v3_v2_construct_equivalence.py
tests/test_water_impact_dynamic_v4_causal_v3.py
```

The v3 code registry at
`data/water_impact_dynamic_v4/v4_eval_code_registry_v3.json` uses protocol
`water_impact_dynamic_v4_eval_code_registry_v3` and has exactly
`protocol,status,dataset_version,v2_read_allowlist,artifacts`. Status is
`frozen`; dataset version is `v4_dev72_v3`. `artifacts` has exactly
`protocol,candidate_builder,stage0_authorizer,screening_runner,screening_freezer,selector,validator,capacity_validator,identity_disjointness_auditor,construct_equivalence_auditor,tests,generator`.
The first eleven records bind the paths above; `generator` binds only the generic
`scripts/generate_wan_clean.py`. Every record is exactly `{path,sha256}` and is
rehash-validated before output reservation and again before publication.

The complete v2 read allowlist is exactly:

| Purpose | Path | SHA-256 |
|---|---|---|
| public design evidence | `results/water_impact_dynamic_v4_causal_screening_termination_v2.md` | `fc6171711a73f4a6eeb30d1f2d005439b7ff7fb7a91d064642fe5da02461ad77` |
| immutable training-bank upstream | `data/water_impact_dynamic_v4/source_bank_public64_registry_v2.json` | `473af632f8100e9e7c46c35e5fd679c9729bc80d19af12aaa78a1a0c69c9f814` |
| immutable training-mapping upstream | `data/water_impact_dynamic_v4/source_mapping_v2.json` | `6988d19ef759b6fd7c15543a5b1774656a20dace82ea82fe5812004f763bb4c2` |

The latter two are the only v2 data upstreams. All three files are opened
read-only, must be regular non-symlink files at the exact canonical paths, and
must retain the same byte hashes before and after every action. V3 binds the
v2 Stage-0 and freeze digests from Section 1 as literals in the termination
record; it must not open those v2 wrapper/freeze paths.

No v3 entry point or transitive import may import a v2 protocol, builder,
authorizer, runner, selector, validator, or test module. No other v2 path may
be read, resolved, globbed, aliased, symlinked, hardlinked, or used as a
fallback. Static AST/import and literal-path scans plus a runtime open-path
audit enforce this allowlist. A missing v3 artifact is fatal; it is never
replaced by a same-named v2 artifact. The code registry validator rejects
extra/missing artifacts, code-byte drift, a v2 import, a nonallowlisted `_v2`
path, or an allowed path reached through an alias.

Tests must include direct and transitive v2-import rejection, each of the three
allowed hashes tampered, an extra v2 path, v2 Stage-0/freeze opening, symlink
and hardlink aliases, missing v3 artifacts, code-registry drift, v2 fallback,
and before/after proof that the three allowed v2 files remained byte-identical.

The identity-disjointness auditor is a separate isolated role exception, not a
v3 runtime import or transitive dependency. Its complete v2 read allowlist is
only:

```text
data/water_impact_dynamic_v4/causal_stage0_commitment_v2.json
PRIVATE_V2_ROOT/causal_stage0_candidates_private_v2.json
```

It first verifies the public wrapper hash
`29696ad8031bb164fe1c6819c8c382d7e4e828835f750f0d245e4877d4167b38`,
then requires the private candidate bytes to equal that wrapper's committed
`candidate_manifest_48` digest and row count. `PRIVATE_V2_ROOT` is an explicit
mode-700 non-symlink root; the candidate is a mode-600 regular non-symlink
file with link count one. An open-path sandbox rejects every other v2 private
file, including media, reviews, eligibility, freeze, secrets, salts, seeds,
and sealed/final36 content.

The auditor may additionally read only the four committed v3 private inputs
`eval_holdout_source_ontology_private48_v3.json`,
`receiver_ontology_private56_v3.json`,
`historical_receiver_anchors_private8_v3.json`, and
`causal_stage0_candidate_graph_private576_v3.json`. It writes only
`data/water_impact_dynamic_v4/v4_causal_identity_disjointness_v3.json`, with
exact top-level keys
`protocol,status,dataset_version,v2_stage0_registry_sha256,v2_candidate_manifest_sha256,v3_candidate_graph_sha256,v3_ontology_bundle_sha256,compared_counts,allowed_identity_exceptions,intersection_counts`.
Its protocol is
`water_impact_dynamic_v4_v3_v2_identity_disjointness_audit_v1`, status is
`passed`, and dataset version is `v4_dev72_v3`.
The allowed exceptions are aggregate counts of eight original source nodes and
eight historical receiver nodes. Intersection counts for case IDs, canonical
records, fresh source IDs, fresh receiver IDs, source-receiver pairs, and
source-receiver-variant triples must all be zero. No identity, phrase, prompt,
row, seed, score, or path is emitted. The v3 authorizer reads only this public
digest/count report; it never reads the v2 wrapper or private candidate. The
auditor executable is code-registry-bound but must never be imported by a v3
runtime entry point. Auditor tests must attempt every forbidden v2-private
category, reject aliases, verify the two allowed v2 files against their
commitments, and enforce the aggregate-only output schema.

Construct equivalence uses a second isolated auditor role. Its complete v2
read allowlist is only the exact public Stage-0 wrapper plus:

```text
PRIVATE_V2_ROOT/causal_stage0_templates_private_v2.json
PRIVATE_V2_ROOT/causal_stage0_field_rules_private_v2.json
PRIVATE_V2_ROOT/causal_stage0_selection_rules_private_v2.json
```

It verifies the public wrapper hash from Section 1 and requires those three
private files to match the wrapper commitments. The required whole-file hashes
for templates and field rules are respectively
`76d3b2be61389a26cc5feb9b1211c5e7b0830a85369e27783fb56e5286ce0559`
and `a1e23230b199a96e9f458c135e6ce2d18bf377966a6867dc5a2cca88d124e2ce`;
the v2 selection-rules file is
`aa41a6da40ae107fafa36ef96c18db6fc7446b9a504123aaaff9a0465f53ed36`.
Its v3 allowlist is exactly the corresponding v3 templates, field-rules, and
selection-rules files. The same private-root permission, non-symlink,
single-link, exact-basename, and open-path sandbox rules apply.

The auditor writes only
`data/water_impact_dynamic_v4/v4_causal_v2_construct_equivalence_v3.json`.
Its exact top-level keys are
`protocol,status,dataset_version,v2_stage0_registry_sha256,v2_file_sha256,v3_file_sha256,qualification_sha256,cell_quota_sha256,exact_equal`.
Protocol is
`water_impact_dynamic_v4_v3_v2_construct_equivalence_audit_v1`, status is
`passed`, and dataset version is `v4_dev72_v3`. The four map objects have exact
inventories: `v2_file_sha256` and `v3_file_sha256` have keys
`templates,field_rules,selection_rules`; `qualification_sha256` and
`cell_quota_sha256` have keys `v2,v3`. `exact_equal`
has exactly `templates,field_rules,qualification,cell_quota`, all boolean true.
Only hashes and booleans are emitted; prompt/template/rule content is forbidden.

The v3 Stage-0 authorizer recomputes all three v3 file hashes plus the canonical
v3 qualification and cell-quota subobject hashes, verifies this public report
and the hard-coded public v2 whole-file hashes, and never opens v2 private
construct files. The construct auditor is code-registry-bound but never
imported by a v3 runtime entry point. Its tests must reject every extra v2
file, tampered commitment, alias, noncanonical subobject encoding, contentful
output field, or false/missing equality flag.

## 9. Success and one-shot failure boundary

V3 has two distinct boundaries. First publication of
`causal_stage0_public_commitment_v3.json` freezes all scientific input bytes:
ontologies, graph, candidate records, templates, rubric, render rules, salts,
seeds, and model/runtime/code registries. It does not authorize generation.
Before the standard wrapper exists, a pending commitment or authorizer package
may be repaired only for a static schema/path/serialization defect, only while
no Original media, review, eligibility, or selector output exists, and only
with every frozen scientific byte unchanged. A scientific authorizer failure
such as an ontology/disjointness, graph, seed, model, runtime, or code-integrity
failure is terminal `preflight_dataset_invalid`; it is not a repairable static
defect. Its invalid-outcome record has `stage0_registry_sha256=null`.

Accordingly, a pre-wrapper authorizer rejection caused solely by a static
schema, path spelling, permission, packaging, or serialization defect is
nonterminal: it writes no invalid-outcome record and may be rerun only after
that static defect is repaired with all scientific bytes unchanged. A
scientific-integrity rejection becomes terminal only after the pending data
freeze exists, and then the null Stage-0 hash is mandatory. After the standard
wrapper exists, every failure category is one-shot terminal; neither static
nor scientific repair is permitted within v3.

The second and one-shot generation boundary is the first successful, exclusive
publication of the standard authorizing wrapper
`causal_stage0_commitment_v3.json`. A scientific-byte change before or after
that publication requires a new amendment/version. From the standard-wrapper
boundary:

- candidate identities, ontologies, pairings, aliases, prompts, templates,
  media, model/runtime/code bytes, salts, seeds, review assignments, and rules
  cannot change;
- all 576 candidates are generated and reviewed exactly once;
- there is no reserve, partial rerun, failed-video replacement, seed swap,
  reviewer-driven prompt edit, or same-version selector retry.

Stage 1 succeeds only if all 576 generations and package records validate, the
review/adjudication freeze is complete, G1 can supply eight distinct eligible
heads with a 4/4 variant and receiver matching, every G2/G3 anchor is covered
under its exact 4/4 matching, the exact 24-case global subset exists, ranks
have no ties, and U contains exactly 72 valid, unique, disjoint seeds. Only
then may the builder atomically create
`causal_stage1_commitment_v3.json`.

Any missing/malformed artifact, model/runtime/code drift, generation failure,
media mismatch, review coverage error, adjudication error, seed collision,
cell with fewer than four eligible candidates, uncovered G2/G3 anchor,
insufficient distinct G1 heads, rank tie, or absence of a globally feasible
subset produces
`preflight_dataset_invalid`. The failure writer may publish only aggregate
counts, reason code, and bound artifact hashes in
`causal_preflight_dataset_invalid_v3.json`; it must not publish identities,
prompts, seeds, row scores, or media. The selector output directory, selected
manifest, U72, and Stage-1 wrapper must remain absent.

A failed v3 may not be repaired. Any future attempt requires a new dataset
version, new ontologies, new case and receiver identities, new media, and fully
rotated salts and seeds. Neither eligible nor ineligible v3 rows may be reused.

## 10. Downstream boundary

A valid causal Stage-1 wrapper is only a prerequisite for a separately
versioned specificity Stage-0/Stage-1 chain. Specificity candidates, W36, M6,
training authorization, prompt sidecars, training outputs, checkpoint
eligibility, three-arm generation, final blind-review commitments, and scoring
must all bind the exact v3 causal Stage-0 and Stage-1 bytes. None may be built
or run under this amendment alone.
