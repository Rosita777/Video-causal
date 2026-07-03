# Fable Review: Method C Counterfactual Grid

## Skeptical Review: Method C0/C1

**Core vulnerabilities:**

1. **VLM verifier circularity**: Your evaluation depends entirely on a VLM's ability to detect targets and footprints. If the VLM fails on subtle footprints (ripples, shadows, indentations), your method reports false negatives. You're measuring "what VLMs can see" not "what causally remains."

2. **Prompt grid contamination**: "no target + footprint" (C) vs "no target + no footprint" (B) with same seed is **not guaranteed to isolate footprint**. Seed controls noise initialization, not semantic parsing. The model may interpret "circular ripples on water" differently in isolation vs. in context of "after removing pebble." You risk conflating prompt ambiguity with causal reasoning.

3. **Negative control weakness**: What validates that B is truly "naturally undisturbed"? If the diffusion model *never* generates ripples without objects, B might just reflect model bias, not a causal counterfactual.

4. **Cherry-picking via C1 search**: "Verifier-guided search over counterfactual prompts" is dangerous. You'll find prompts where the VLM *happens* to detect what you want, not where the model *actually* exhibits the failure mode. This becomes a VLM prompt-engineering exercise, not a diffusion model causal evaluation.

---

## Concrete attacks you must address:

**Attack 1: VLM blindness**  
Run your VLM verifier on ground-truth image pairs (real photos: ball on grass, ball removed leaving indentation). If the VLM misses the indentation, your entire method collapses. **Defense**: Establish VLM recall baseline first. Report it. If <90%, acknowledge as limitation.

**Attack 2: Seed ≠ semantic control**  
Generate B and C with same seed. Manually inspect: do they differ *only* in footprint, or also in background details (lighting, texture)? If background drifts, you're not isolating causality. **Defense**: Add LPIPS/SSIM on non-footprint regions between B and C. High divergence = invalid counterfactual pair.

**Attack 3: Model prior confound**  
Does the diffusion model *ever* generate footprints without targets in its training distribution? If "ripples on still water" is OOD, B might be artificial. **Defense**: Generate 100 "undisturbed water" samples from random seeds. Measure footprint-like patterns. If rare, you're studying model bias, not causal failure.

**Attack 4: Prompt leakage in C1**  
In verifier-guided search, you optimize prompts until VLM detects footprint. But diffusion model might not *intend* the footprint—VLM might be detecting noise, JPEG artifacts, or coincidental textures. **Defense**: Human validation on searched prompts. If humans disagree with VLM on footprint presence, the search is broken.

---

## Defensible C0/C1 protocol (what to code now):

### **Phase 1: VLM verifier validation** (prerequisite)
```python
# Before anything else
1. Collect real photo pairs: object present/absent with clear footprints
   (ball on sand, removed → indentation remains)
2. Run VLM verifier: does it detect target & footprint correctly?
3. Report precision/recall. If recall < 80%, switch VLM or add human loop.
```

### **Phase 2: C0 implementation with controls**
```python
# For each test case (e.g., "pebble in pond"):
for seed in seeds:
    A = generate("pebble splashing in pond", seed)
    B = generate("still undisturbed pond", seed)
    C = generate("pond with circular ripples, no object", seed)
    D = generate("pebble hovering above pond, not touching water", seed)
    
    # Verifier checks
    vlm_check(A) → target=True, footprint=True
    vlm_check(B) → target=False, footprint=False
    vlm_check(C) → target=False, footprint=True  # ← failure mode
    vlm_check(D) → target=True, footprint=False
    
    # CRITICAL: semantic consistency check
    # Reject pair if B and C differ in non-footprint regions
    background_divergence = lpips(crop_non_footprint(B), crop_non_footprint(C))
    if background_divergence > threshold:
        reject_pair()  # seed doesn't give stable counterfactuals
    
    # CRITICAL: negative control
    # Generate N "undisturbed pond" samples, measure false-footprint rate
    false_positive_rate = vlm_detects_footprint_in_undisturbed_samples()
    if false_positive_rate > 5%:
        flag_unreliable_baseline()
```

### **Phase 3: C1 search with human gate**
```python
# Only after C0 shows clear failures
1. Search prompts for C ("ripples without object"):
   candidates = ["circular ripples on water", "concentric waves on pond", ...]
   
2. Generate with same seed as A/B, filter by VLM
   
3. **Mandatory human validation**:
   For top-10 VLM-detected footprints, humans label:
   - Is footprint clearly visible? (yes/no)
   - Does image look like target was removed? (yes/no)
   If human-VLM agreement < 70%, discard C1 results.

4. Report both VLM-detected and human-validated counts separately.
```

### **Phase 4: Ablations to report**
```
1. Different VLMs (GPT-4V, Gemini, LLaVA) → do results replicate?
2. Different diffusion models (SD1.5, SDXL, Flux) → is this model-specific?
3. Human-only evaluation on random sample → ground truth comparison
4. Random prompt baseline: does random C sometimes *not* show footprints?
   (If model ignores "ripples" prompt, that's also interesting)
```

---

## What makes this defensible:

- **VLM validation upfront** → you're not blindly trusting black-box verifier  
- **Background consistency checks** → you're actually isolating causality, not prompt noise  
- **Negative controls** → you distinguish model failure from model prior  
- **Human validation on C1** → you avoid search optimization artifacts  
- **Multi-VLM replication** → you prove it's not one verifier's quirk  

Without these, any reviewer will correctly say: "You've shown that you can find prompts where VLMs detect things, not that diffusion models fail at causal reasoning."
