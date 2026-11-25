# Educational and Research System for Clearly Labelled, Silent Synthetic Videos Depicting Donald Trump

Use a **reference-image-to-video system with a frozen pretrained model**, followed by automated checks and human approval. This is practical for a small research project and gives a clear baseline for studying identity drift and motion consistency.

Assume a **six-second clip: 48 frames, 720×480 pixels, 8 fps**, one person, a fixed camera, neutral background, and modest movement such as blinking or turning slightly. Every frame carries **“AI-GENERATED — NOT REAL FOOTAGE.”** The file contains no audio track. The intended use is classroom demonstration or controlled research.

## 1. Data and provenance

Collect roughly 20–40 rights-cleared reference photographs covering frontal and slight side views, with consistent apparent age. Record source URLs, licenses, dates, and permitted uses; publicly accessible images are not automatically reusable.

Split references by original photo session or event, keeping near-duplicates together. Reserve several sessions for evaluation. Use one approved portrait to condition each generation; the remaining images support identity checks. This baseline requires **no identity-specific training dataset**.

Maintain a manifest linking each output to its reference, prompt, model version, settings, seed, and review decision. Document that the pretrained model’s original training-data provenance may be incomplete.

## 2. Identity and temporal consistency

A concrete baseline is [CogVideoX-5B-I2V](https://huggingface.co/THUDM/CogVideoX-5b-I2V). Its documented example produces **49 frames at 720×480 and exports at 8 fps**; retain 48 frames for an exact six-second deliverable. Review its model license before adoption.

Generate the entire shot jointly from the reference portrait. Keep appearance, lighting, clothing, and background fixed, and request only modest movement. Consistency comes primarily from image conditioning and the model’s temporal processing; a fixed seed supports reproducibility but does not guarantee identity.

Reject clips with changing facial proportions, unstable hair or clothing, disappearing features, or substantial unintended mouth movement. Use face alignment for evaluation. Independently replacing or restoring each face can introduce flicker.

## 3. Hardware and budget

| Resource | Practical starting allocation |
|---|---|
| GPU | One NVIDIA GPU with 24 GB VRAM; 48 GB provides more flexibility |
| Computer | 64 GB RAM and 1–2 TB SSD |
| Memory controls | CPU offloading and VAE tiling; test quantization separately |
| Local purchase allowance | Approximately US$2,000–3,500 for a complete workstation |
| Cloud pilot allowance | Reserve 20 GPU-hours; at an assumed US$1–4/hour, US$20–80 |

These are planning allowances, not current vendor quotes. Benchmark the first five clips to establish runtime, peak memory, and **cost per accepted clip**, including rejected attempts. Published minimum-memory figures depend heavily on offloading and do not establish comfortable operating requirements.

Keep fine-tuning outside the initial scope: it adds data, memory, and evaluation costs before the baseline’s limitations are known.

## 4. Evaluation

Run approximately 30 generations across several approved portraits, motion prompts, and seeds. Retain failed generations in the research results.

- **Identity:** Compare face embeddings against held-out photographs, reporting median similarity, worst-frame similarity, and drift across time. Calibrate thresholds using real same-person and different-person examples; there is no universal cosine cutoff.
- **Temporal quality:** Measure motion-compensated perceptual differences in visible regions, landmark stability, and background flicker. Check that low motion scores do not merely reward frozen images.
- **Human review:** Have two reviewers independently score recognizability, facial integrity, motion plausibility, and label readability.
- **Delivery:** Verify exactly 48 frames, resolution, constant frame rate, absence of an audio stream, and readable disclosure throughout.

Human approval is required for every released clip. Report measured acceptance rates rather than promising identity preservation.

## 5. Misuse controls

Restrict the interface to approved neutral scenes and motions. Exclude fabricated speeches, endorsements, news events, and realistic claims that Trump performed a particular consequential act.

Burn disclosure into the frames after generation, add a visible research identifier, and include signed provenance metadata where supported. Test label readability after resizing and recompression. Labels and metadata help establish origin but cannot prevent deliberate removal.

Keep generation access authenticated, rate-limited, and logged. Publish approved labelled outputs with their limitations; restrict access to unlabelled intermediates and any future identity-specific weights. Provide a reporting and withdrawal process.

The first research question should be narrow: **How reliably does reference conditioning preserve recognizable identity during modest movement, and how does that reliability change across poses and seeds?** This produces a useful, measurable educational system within a single-GPU budget.
