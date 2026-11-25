# AI-generated alternative proposal (raw record)

- Tool / model: Claude Opus 5.5 (Anthropic), via the Claude desktop app (Claude Code)
- Date: 10 November 2026
- Prompt: "Propose a practical educational/research system for clearly labelled, silent synthetic videos depicting Donald Trump. Cover data, identity and temporal consistency, hardware budget, evaluation, and misuse controls; state assumptions for a short 720p, 24 fps output."
- Note: the response below is kept unedited. It was generated after Part 1 had been written; it is not an experimental result.

---

**Assumptions.** 5-second clips at 1280×720, 24 fps (120 frames), no audio, neutral settings (podium, walking, seated interview framing). Research use only, under institutional approval.

**1. Data**
- Sources: U.S. federal government footage (generally public domain in the U.S., e.g. official White House recordings) plus archive footage licensed for research. Keep a provenance log for every clip.
- Target: about 1–3 hours of usable footage after filtering, roughly 500–1,500 clips of 3–6 s, covering varied angles, lighting and years.
- Processing: scene-cut detection, face detection and tracking, keep clips where the subject is the dominant face, crop/resize to 16:9 720p, remove audio, auto-caption with a vision-language model and add a unique trigger token to every caption. Split 80/10/10 by event so that the same speech does not appear in both training and test data.

**2. Model**
- Base: Wan2.1-I2V-14B-720P (open weights, Apache-2.0). Image-to-video gives a strong identity anchor, because the first frame is a reference image of the subject.
- Identity: train a subject LoRA (rank 32–64) on the transformer's attention layers using the captioned clips; at inference, condition on a reference frame plus the trigger token.
- Temporal consistency: rely on the model's spatio-temporal attention; generate 81 frames at 16 fps (about 5 s), then interpolate to 24 fps with RIFE and optionally upscale.
- Optional: add an IP-Adapter/ArcFace identity module if LoRA alone drifts.

**3. Hardware and budget**
- LoRA fine-tuning: 1–2 × 80 GB H100 for about 24–48 hours with gradient checkpointing and bf16.
- Inference: one 80 GB GPU per clip.
- Cloud cost: roughly US$300–800 for training experiments plus evaluation.

**4. Evaluation**
- Identity: ArcFace cosine similarity to reference images; target mean > 0.6 across frames.
- Temporal: VBench subject consistency, motion smoothness and temporal flickering.
- Quality: FVD against held-out real clips; CLIP score for prompt alignment.
- Human study: 20+ raters judge realism and identity.

**5. Misuse controls**
- Visible "AI-generated" label on every frame; C2PA metadata; an invisible video watermark so the clip can still be identified if the visible label is cropped.
- Prompt filtering to block political statements, real events, violence or sexual content; no audio or voice cloning.
- Access restricted to the research team; all outputs logged.

**Main risks.** Identity drift in longer shots, uncanny facial motion, dataset licensing, and misuse if clips leak.
