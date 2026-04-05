# Security & Responsible Use

## What Kaari Is

Kaari is a **detection and monitoring tool** for prompt injection in LLM applications. It measures semantic deviation between user intent and model response to flag suspicious behavior.

## What Kaari Is Not

Kaari is **not a firewall, filter, or safety guarantee**.

- It does not prevent injection — it detects it after the response is generated
- It does not guarantee detection of all injection types
- It does not replace human review for high-stakes applications
- It does not validate the factual correctness of responses

## Responsible Use

### DO use Kaari to:
- Monitor LLM responses for deviation from user intent
- Log and alert on suspicious response patterns
- Add a detection layer to existing LLM pipelines
- Research and study prompt injection patterns

### DO NOT rely on Kaari as:
- The sole security measure for safety-critical systems
- A replacement for input sanitization and prompt hardening
- A guarantee against adversarial attacks
- A content moderation or factual accuracy system

## Known Limitations

1. **Adversarial robustness is untested.** An attacker who knows Kaari is in use could craft responses that maintain low cosine distance while still being injected. This is an active research area.

2. **Threshold calibration matters.** The default threshold (0.245) is calibrated on research data (N=2,228) with specific models and prompt types. Your deployment may need different thresholds. Run `python -m kaari.calibrate` on your own data.

3. **Embedding model dependency.** Detection quality depends on the embedding model. Validated across 3 embedding models (nomic-embed-text, all-MiniLM-L6-v2, bge-base-en-v1.5) with AUC spread +/-0.006, confirming encoder independence. However, optimal thresholds vary per model — recalibrate for your deployment.

4. **Simple prompt bias.** Research used short, single-turn prompts. Performance on long documents, multi-turn conversations, or system prompts with extensive context is not yet validated.

5. **Natural divergence.** Some legitimate conversation styles produce elevated scores. Creative writing, debate-style prompts, and open-ended exploration may trigger YELLOW zone alerts. This is expected — Kaari measures real semantic distance. See README for guidance on threshold adjustment.

## Reporting Vulnerabilities

If you find a way to consistently bypass Kaari detection, we'd like to know. This helps us improve the tool for everyone.

Contact: tatu@sollucidlabs.com

Please include:
- The prompt and injection used
- The model and embedding provider
- The Kaari score and expected vs actual classification
- Whether you believe this represents a general bypass or a specific edge case

## Accuracy Statement

As of v0.95.0, Kaari achieves:

| Metric | Value | Conditions |
|--------|-------|------------|
| AUC-ROC (dv2) | 0.770 | N=2,228, Option B pipeline (raw prompt, no LLM) |
| AUC-ROC (C2) | 0.822 | N=2,228, 4 LLM architectures, 3 embedding models |
| Cohen's d | 1.72 | Combined effect size |

These numbers are from controlled research conditions using the Option B pipeline (raw prompt embedding, no LLM summarization). Real-world performance may vary. We encourage users to validate on their own data before deploying to production.

The zone system (GREEN < 0.210, YELLOW 0.210-0.245, RED >= 0.245) is calibrated to reduce false positives compared to the Youden-optimal threshold. For applications requiring maximum sensitivity at the cost of more false positives, lower the threshold via custom calibration.

## Citation

If you use Kaari in research, please cite:

```bibtex
@article{lertola2026intent,
  title={Intent Vectoring: Black-Box Prompt Injection Detection via Semantic Deviation Measurement},
  author={Lertola, Tatu Samuli},
  journal={SSRN preprint},
  year={2026}
}
```
