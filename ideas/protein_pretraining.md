# Protein language model pretraining, 2024 to 2026: recipes and what transfers to a 10-minute 36x768 MLM

This survey covers protein language model (pLM) pretraining papers and models from 2024 to 2026, plus the older masked language modeling (MLM) studies that the newer work relies on. The target is the benchmark in this repository: a 36-block transformer with hidden size 768 and MLP intermediate size 2048, trained with 15 percent MLM (MASK-only, bidirectional) for 10 minutes on one H100-class GPU, on 1.75 million AlphaFold DB proteins (50 to 2,046 residues, about 390 million residues) with Foldseek 3Di strings for every protein, and evaluated on a test split that shares no sequence or structural homology with training (0.3 identity by MMseqs2 and Foldseek).

**Summary of the literature:** the evidence most consistently points to data and objective choices, not architecture. A diverse, cluster-aware corpus, a masking rate that is varied rather than fixed at 15 percent, and structure tokens used as extra signal are the levers with reported effects. No published study we found measures MLM convergence speed under a fixed wall-clock budget, so every transfer below is a hypothesis to test by validation loss at equal training time.

**Conventions.** Each evidence label means the following. *Ablated*: the paper isolates the change with a controlled comparison. *Reported*: the paper gives a result that is not isolated by a controlled comparison. *Anecdotal*: a model card, blog or secondary summary. Where a claim comes only from a search summary and the full text was not read, the entry says "(summary only)". Where the full text could not be retrieved, the entry says so. Items under "Not found" are gaps, not negative results.

**Our arithmetic (not from the literature).** A 36-block model with hidden size 768 and a two-matrix MLP has roughly 0.2 billion parameters. At about 6N FLOPs per token and an assumed 300 TFLOP/s effective throughput, 10 minutes covers on the order of 10^8 residues, well under one pass over the 390 million residue corpus. The run is therefore step-limited and single-pass, which matters for the repetition findings below.

---

## 1. Recipes of recent pLMs

### 1.1 ESM C (EvolutionaryScale, 2024)

Source: [EvolutionaryScale, ESM C blog](https://www.evolutionaryscale.ai/blog/esm-cambrian) (fetched; the blog is the only primary source retrieved, and no technical report was found).

| Item | 300M | 600M | 6B |
|---|---|---|---|
| Layers / width / heads | 30 / 960 / 15 | 36 / 1152 / 18 | 80 / 2560 / 40 |
| Peak learning rate | 5e-4 | 4.6e-4 | 3.1e-4 |
| Batch size | 4.2M tokens | 4.2M tokens | 4.2M tokens |
| Steps | 1.5M | 1.5M | 1.5M |
| Training tokens | 6.2T | 6.2T | 6.2T |

- **Data:** UniRef, MGnify and JGI sequences, each clustered at 70 percent identity (83M, 372M and 2B clusters respectively). Raw sequence counts are not stated.
- **Stages:** stage 1 runs 1M steps at context 512 with metagenomic data at 64 percent of the mix. Stage 2 runs the final 500K steps at context 2048 with metagenomic data at 37.5 percent.
- **Architecture:** Pre-LN transformer, rotary embeddings, SwiGLU, no biases in linear layers or norms.
- **Objective:** MLM. Masking rate, tokenizer, optimizer and schedule are not stated.
- **Finding that matters:** the blog reports near-linear gains in structure information with compute, and says the 300M and 600M models were trained past the compute-optimal point predicted by the scaling experiments. No ablation beyond the scaling experiments is reported.

*Transfer.* (a) A context schedule that starts at 512 and extends to 2048 late in training is a concrete length curriculum. For our 50 to 2,046 residue data, train at crop 512 for most steps and lengthen near the end. Expected effect: higher throughput early, effect on test loss unknown. Evidence: reported. (b) Cluster the training set before sampling (section 4). The 36-layer depth matches the ESM C 600M shape, but the width differs (1152 against our 768).

### 1.2 ESM3 (Hayes et al., bioRxiv 2024; Science 2025)

Sources: [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.07.01.600583.full.pdf), [EvolutionaryScale ESM3 blog](https://www.evolutionaryscale.ai/blog/esm3).

- **Scale:** the largest model has 98 billion parameters, trained on 2.78 billion proteins (summary only).
- **Objective:** generative masked modeling over discrete tokens for sequence, structure and function tracks. The share of masked positions is drawn from a noise schedule, so the model sees many masking combinations rather than one fixed rate (summary only).
- **Not retrieved:** the structure tokenizer, codebook size, learning rate, batch size and token count. The full-text fetch was rate limited.
- **Finding that matters:** mask rate varied during training is part of the core objective. No isolated ablation was retrieved.

*Transfer.* Draw the mask rate per sequence from a range (for example 15 to 30 percent) instead of fixing it. Expected effect: unknown on test loss; the reported design is consistent with the masking results in section 2. Evidence: reported.

### 1.3 AMPLIFY (Fournier et al., bioRxiv 2024)

Sources: [bioRxiv preprint "Protein Language Models: Is Scaling Necessary?"](https://www.biorxiv.org/content/10.1101/2024.09.23.614603.full.pdf) (full text not retrieved; rate limited), [chandar-lab/AMPLIFY repository](https://github.com/chandar-lab/AMPLIFY) (fetched), [Mila summary](https://mila.quebec/en/insight/is-bigger-always-better-democratizing-ai-protein-discovery).

| Item | Value (source) |
|---|---|
| Training sets listed | UniRef100, UniRef50, OAS and SCOP (repository) |
| Dataset sizes, UniRef release, clustering | not stated (repository) |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.01 (repository, 120M example) |
| Peak learning rate, schedule | 1e-3, cosine decay, 1,000 warmup steps, 1M steps (repository, 120M example) |
| Gradient clipping | 1.0 (repository) |
| Effective batch | 4,096 sequences, derived from per-device batch and accumulation (repository) |
| Precision and hardware | bf16; 120M on 8 GPUs with DeepSpeed ZeRO-3 (repository) |
| Training tokens, 350M | 2T tokens (stated in the recAMPLIFY summary below) |
| Masking rate | not stated in retrieved sources |

- **Finding that matters:** the authors argue that data quality (sample bias) and an efficient codebase matter more than parameter count. The Mila summary quotes developer claims that the 350M model competes with and in places surpasses ESM-2 650M while using about 17 times less compute. These are developer claims, not independent evaluations.

*Transfer.* The repository's 120M optimizer settings (AdamW, betas 0.9 and 0.95, peak 1e-3, cosine decay with 1,000 warmup steps) are a concrete starting point for the learning rate schedule. The 4,096-sequence batch is much larger than a 10-minute run would use, so scale it by tokens. Evidence: reported (developer claims).

### 1.4 recAMPLIFY (Jha et al., arXiv 2026)

Sources: [papers.cool listing of arXiv 2610.26747](https://papers.cool/arxiv/2610.26747), [Hugging Face chandar-lab/recamplify](https://huggingface.co/chandar-lab/recamplify). Summary only; the full text was not read.

- A weight-tied core that refines hidden states over a tunable number of recursive cycles, with 155M unique parameters, trained on the same 2T tokens as AMPLIFY-350M.
- Reported: beats AMPLIFY on contact recovery and three of four supervised tasks, and raises fold accuracy by 22.5 percent (abstract).
- Gated Mixture Recurrence reduces expected depth by up to 37 percent, with task-dependent accuracy trade-offs.

*Transfer.* Weight-tied recursion gives more effective depth per parameter, but the benchmark fixes the block count at 36. Whether a recursive core counts as 36 blocks is a question for the benchmark owners, not a literature question. Evidence: reported.

### 1.5 ProGen3 (Bhatnagar et al., bioRxiv 2025; NeurIPS 2025)

Sources: [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.04.15.649055v1), [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/115098). The full text was not retrieved; the rate limit blocked the fetch.

- Sparse generative (causal) pLM family with compute-optimal scaling laws, up to 46B parameters trained on 1.5T amino acid tokens (abstract).
- Pretraining samples from an "optimized data distribution" over the Profluent Protein Atlas v1, 3.4B full-length proteins (abstract).
- Reported: larger models generate viable proteins for more families and respond better to alignment with lab data (abstract).
- Mixture-of-experts with about 27 percent of parameters active per pass (summary only).

*Transfer.* The data reweighting is the transferable idea: sample clusters with a designed distribution rather than raw frequency. The model is causal, so the architecture does not transfer. Evidence: reported.

### 1.6 xTrimoPGLM (2024)

Source: [arXiv 2401.06199](https://arxiv.org/abs/2401.06199) (abstract read via search).

- Mixed objective: BERT-style masking on a bidirectional prefix, plus GLM-style autoregressive generation on a suffix. Reported scale of 100B parameters on 1T tokens, with results on 18 understanding benchmarks (abstract).

*Transfer.* None for an MLM-only benchmark. The mixed objective costs compute that the 10-minute budget does not allow. Evidence: reported.

### 1.7 DPLM and DPLM-2 (Wang et al., arXiv 2024; ICLR 2025)

Source: [arXiv 2410.13782](https://arxiv.org/html/2410.13782v1) (fetched).

- **Data:** about 20K clustered PDB structures and about 200K AlphaFold-predicted SwissProt structures filtered to pLDDT above 85. The main text and appendix disagree on the total; the appendix gives about 200K in total.
- **Cropping:** maximum length 512, with random cropping. Structures with pLDDT below 50 at the ends are cropped.
- **Tokenizer:** lookup-free quantization (LFQ) with codebook size 8192, from a frozen pretrained GVP-Transformer encoder with an IPA decoder.
- **Model sizes:** 150M, 650M and 3B. Layer counts are not given.
- **Optimization:** AdamW, betas (0.9, 0.95), weight decay 0.01, gradient clipping 0.5, maximum learning rate 1e-4, 2K warmup steps, linear decay to 10 percent, 100K steps. Batch size and tokens are not stated.
- **Diffusion:** absorbing-state (mask) discrete diffusion with 500 timesteps, with a separate noise scheduler per modality.
- **Warm start:** initialized from the pretrained sequence-only DPLM, with LoRA to limit forgetting.
- **Ablation (Table 3, sc-TM at length 500):** neither 0.4511; sequence pretraining only 0.7673; synthetic structures only 0.8288; both 0.9012.

*Transfer.* The ablation shows that structure data and sequence data contribute together, with structure data alone recovering much of the gain. For our from-scratch run, the pLDDT filter and end-cropping are cheap to apply to the AlphaFold DB 3Di inputs. Warm-starting from a sequence model does not apply to a 10-minute from-scratch run. Evidence: ablated (for structure generation, not MLM convergence).

### 1.8 SaProt (Su et al., ICLR 2024)

Sources: [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.10.01.560349.full.pdf), [ICLR 2024 poster](https://iclr.cc/virtual/2024/poster/19394), [project README mirror](https://www.gitlink.org.cn/NSCCN/SaProt) (summary only).

- **Vocabulary:** residue and Foldseek 3Di tokens combined into one structure-aware token per residue (a vocabulary of residue-plus-state pairs).
- **Data:** about 40 million sequence-structure pairs.
- **Results:** reported to beat baselines across 10 downstream tasks (ICLR abstract, summary).
- **Sequence-only variant:** the project README says a residue-only variant performs similarly to ESM-2 35M (summary only). The same README says frozen embeddings from the 35M and 650M models work best with structure input.

*Transfer.* The combined vocabulary is the cheapest way to add 3Di, but it changes the MLM target from 20 symbols to a joint token. Test loss would then no longer be comparable to residue-only loss. The benchmark must define its test loss on residues, which argues for an auxiliary-target design (section 3). Evidence: reported.

### 1.9 ProstT5 (Heinzinger et al., bioRxiv 2023; journal version not confirmed)

Sources: [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.07.23.550085.full.pdf), [PMC copy](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11616678/).

- T5 encoder-decoder initialized from ProtT5, trained on 17 million AlphaFold DB predictions converted to 3Di.
- Two stages: span denoising on both sequence and 3Di tracks, then bidirectional translation between the two.
- Reported: deriving 3Di tokens from a structure is about three orders of magnitude faster than structure-based computation (PMC copy).

*Transfer.* Precompute 3Di for all 1.75 million proteins once with Foldseek. Span denoising on the 3Di track alone is an untested variant. Evidence: reported.

### 1.10 ProSST (Li et al., NeurIPS 2024)

Source: [NeurIPS 2024 abstract](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3ed57b293db0aab7cc30c44f45262348-Abstract.html).

- Structure is quantized by a GVP encoder and k-means into discrete tokens. A Transformer with sequence-structure disentangled attention is pretrained with MLM on millions of structures.
- Reported: state-of-the-art zero-shot mutation effect prediction and several supervised tasks.

*Transfer.* Disentangled attention between the two tracks is a model change with moderate cost. Low priority for a 10-minute run. Evidence: reported.

### 1.11 E1 (Jain et al., bioRxiv 2025; ICML 2026)

Sources: [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.11.12.688125.full.pdf), [ICML 2026 listing](https://icml.cc/virtual/2026/72071), [GitHub README](https://raw.githubusercontent.com/Profluent-AI/E1/main/README.md).

- Retrieval-augmented encoder: homologous sequences are retrieved at inference and attended with block-causal multi-sequence attention. Single-sequence mode is also supported.
- Sizes 150M, 300M and 600M. Trained on four trillion tokens from the Profluent Protein Atlas v1 (listing summary).
- Motivation: single-sequence models must store all evolutionary context in their weights, so underrepresented families are poorly captured (preprint).

*Transfer.* Not applicable to a single-sequence MLM benchmark. The retrieval step also conflicts with the homology-disjoint test design, because retrieved neighbors for test proteins would be weak by construction. Evidence: reported.

### 1.12 ESM-2 (Lin et al., Science 2023) and Ankh3 (Alsamkary et al., arXiv 2025)

**ESM-2.** Architecture facts from NVIDIA BioNeMo documentation (summary only): the 650M model has 33 layers, 20 heads and hidden size 1280; the 3B model has 36 layers, 40 heads and hidden size 2560; inputs are capped at 1022 residues. Its 15 percent, 80/10/10 corruption is confirmed only as the setup used by [Serrano et al.](https://arxiv.org/html/2406.07249v1), a follow-up study; the ESM-2 paper text was not retrieved. A 650M checkpoint config sets `token_dropout` to true (Hugging Face config, summary only). The rate and mechanism of token dropout were not verified.

**Ankh3.** Source: [arXiv 2505.20052](https://arxiv.org/html/2505.20052v1) (fetched).

- **Objectives:** MLM with masking rates 15, 20 and 30 percent, sampled uniformly at each step, plus sequence completion (the first 50 percent of a sequence is encoded and the rest decoded).
- **Data:** 59,130,945 distinct UniRef50 sequences.
- **Sizes:** Large at 1.88B parameters and XL at 5.73B. Both have 72 layers (48 encoder, 24 decoder).
- **Optimization:** Adafactor, learning rate 1e-2, rsqrt decay with 10K warmup steps, batch size 1024, sequence length 512, about 4M steps (Large) and 5M steps (XL). Hardware is 64 TPUv4.
- **Finding:** the authors attribute the gains to multi-rate masking and completion together. No controlled ablation isolates either component.

*Transfer.* Multi-rate MLM with uniform 15, 20 and 30 percent sampling is the cheapest objective change in this survey. Evidence: reported, not isolated.

### 1.13 METL (Gelman et al., bioRxiv 2024; ICLR 2024 abstract)

Source: [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.03.15.585128.full.pdf). The later journal version was not confirmed.

- Pretraining on Rosetta-simulated structures of millions of variants, with 55 biophysical attributes per structure as targets.
- Relative positional embedding built from 3D distances between residues.
- Reported: generalizes from small training sets and position extrapolation (preprint).

*Transfer.* Out of scope unless Rosetta features are added. The structure-distance positional embedding is a possible alternative to 3Di tokens. Evidence: reported.

### 1.14 DSM and the Synthyra stack (Hallee et al., arXiv 2025; FastPLMs, Protify)

- **DSM:** [arXiv 2506.08293](https://arxiv.org/abs/2506.08293) (abstract only). Built on the ESM2 architecture with a masked forward diffusion process inspired by LLaDA. Reported: generation remains coherent with 90 percent token corruption, and representations match or exceed similarly sized pLMs. Training data, mask schedule, optimizer and token count were not retrieved.
  *Transfer.* A masked-diffusion objective is a per-sequence sampled corruption rate, which is the same idea as section 2's variable mask rate. Evidence: reported.
- **FastPLMs:** [GitHub repository](https://github.com/Synthyra/FastPLMs) (README not fetched; facts come from Hugging Face model cards). The FastESM2 cards list attention backends sdpa (default), kernels_flash, flex and auto. The ESM++ cards describe a reimplementation of ESMC weights with batching and Hugging Face loading. Some ESM++ card revisions name flex attention as default and later revisions name sdpa, so the current default must be checked in the repository. The ESMFold2 card warns that flex attention can diverge numerically from SDPA. The FastESM2-650 card claims over twice the speed of ESM2-650 on an H100 with longer sequences (card claim; benchmark images not read).
  *Transfer.* Use SDPA or varlen flash attention for the 10-minute run, and check loss parity against SDPA before any speed comparison. Evidence: anecdotal.
- **Protify:** [PyPI page](https://pypi.org/project/Protify/) describes a beta tool with a GUI, CLI and YAML experiments that compute embeddings from ESM2 and ESMC through FastPLMs. It is a downstream evaluation tool, not a pretraining recipe.

### 1.15 ProtGPT2 (Ferruz et al., 2022; background only)

Source: [Nature Communications 2022 (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9329459/). Causal, 738M parameters, 36 layers, width 1280, trained on UniRef50 2021_04 without FASTA headers. No 2024 to 2026 successor or ProtTrans update was found. Causal pretraining, so no transfer to MLM.

---

## 2. Masking rate and corruption

### 2.1 Should 15 percent be the rate? (Wettig et al., arXiv 2022; EACL 2023)

Sources: [arXiv 2202.08005](https://arxiv.org/abs/2202.08005) (fetched), [ACL Anthology 2023.eacl-main.217](https://aclanthology.org/2023.eacl-main.217).

- 15 percent is not universally optimal. For BERT-large-sized models, 40 percent masking beat 15 percent on GLUE and SQuAD (exact scores not retrieved).
- Even 80 percent masking kept about 95 percent of fine-tuning performance (abstract).
- Uniform masking needs a higher rate than span or PMI masking.
- Mechanism proposed by the authors: a higher rate corrupts more input, which makes prediction harder, but yields more prediction targets per sequence, which helps optimization.
- The paper also revisits the 80/10/10 corruption split (no protein evidence).

*Transfer.* Run 15 versus 30 percent at equal wall-clock time and select by validation loss. Expected effect: the higher rate gives more targets per step, but a harder task, so the sign on validation loss at 10 minutes is uncertain. Evidence: ablated on BERT-scale models with downstream metrics, not convergence.

### 2.2 Dynamic masking (Liu et al., RoBERTa, 2019)

Source: [arXiv 1907.11692](https://arxiv.org/pdf/1907.11692). The dynamic-versus-static comparison for BERT-base: SQuAD 2.0 F1 78.7 against 78.3, MNLI-m 84.0 against 84.3, SST-2 92.9 against 92.5, medians over five seeds. The authors report the difference as small.

*Transfer.* Resample masks at every batch. Cost is zero. Expected effect: small. Evidence: ablated, small effect.

### 2.3 Multi-rate and variable-rate masking

- **Ankh3:** uniform 15, 20 and 30 percent (section 1.12). Reported, not isolated.
- **ESM3:** masking share drawn from a schedule (section 1.2). Reported.
- **Structure-aware masking, 2026:** Walton, Goel and Aghazadeh propose Bucket Masking, which masks spatially grouped residues. [arXiv 2605.16581](https://arxiv.org/pdf/2605.16581) (summary only; the HTML version returned not found). The summary says the method was fine-tuned on ESM-2 650M with a masking rate sampled from U(0.15, 0.75) per epoch, that the best exploration rate was 0.2, and that the method gains up to 14 percent over random masking on four fitness prediction tasks. The authors attribute the gain to mask placement rather than span size (ablated in their summary).

*Transfer.* Bucket Masking needs residue neighbor sets, which can be computed once per AlphaFold DB structure. Expected effect on MLM loss: unknown. Evidence: reported (downstream fitness, summary only).

### 2.4 Span masking

- **Span masking on 3D chains (Zhao et al., ICML 2024; [arXiv 2402.01481](https://arxiv.org/abs/2402.01481)):** span masking at residue and atom level for structure-based pretraining. The authors report that naively adding atom inputs leaks information that makes residue-level tasks trivial. Summary only for the numbers.
- Wettig et al. (section 2.1) report that span and PMI masking need a lower rate than uniform masking for the same effect.

*Transfer.* Not adopted. Span masking on residues in sequence-only MLM was not found in protein work with an ablation. Evidence: reported (structure-based pretraining, different objective).

### 2.5 Other masking items

- **80/10/10 versus MASK-only:** no protein study that isolates the 10 percent random and 10 percent unchanged positions was found. The benchmark already specifies MASK-only, so no change is suggested without evidence.
- **Token dropout:** only the ESM-2 config flag was found (section 1.12). No protein ablation was found.
- **ModernBERT 30 percent masking:** the original paper's masking rate could not be verified from the retrieved sources. Derivative papers use 30 percent.

---

## 3. Structure-aware pretraining

### 3.1 Structure tokens as input (SaProt, DPLM-2, ProstT5, ProSST)

Covered in sections 1.7 to 1.10. Summary of the evidence:

- SaProt (summary): combined residue and 3Di tokens improve reported downstream tasks. No sequence-only ablation with matched training data was retrieved beyond the README note.
- DPLM-2: the ablation (section 1.7) shows structure data and sequence data both help structure generation, not sequence-only MLM loss.
- ProstT5: span denoising on the 3Di track is part of the recipe.

*Transfer for us.* A combined token changes the target. An auxiliary target leaves the residue loss unchanged (section 3.2).

### 3.2 Structure as an auxiliary target (ISM, structure-aligned pLMs)

- **Implicit Structure Model (ICLR 2025):** a sequence-only model is trained to predict k-means structure tokens derived from a structure encoder. Reported: gains over ESM2 on mutation stability and structure prediction (search summary; the full text was not read; no ablation retrieved).
- **Structure-Aligned Protein Language Model ([arXiv 2505.16896](https://arxiv.org/pdf/2505.16896), 2025):** auxiliary alignment losses to structure. Ablated on contact prediction, fluorescence and metal binding. The authors report that alignment to AlphaFold2 representations underperforms the baseline ESM2 without alignment, and they compare Foldseek tokens with other structure tokenizers. The ESM3 structure token was not compared because of its license.
- **ProteinJEPA (ICML 2026 listing, summary):** latent prediction at masked positions with MLM retained. Gains for a pretrained ESM2, mixed results when pretraining from scratch (6, 8 and 2 wins across three settings, as summarized).

*Transfer.* Auxiliary 3Di prediction keeps the residue loss defined on residues. Expected effect on homology-disjoint test loss: unknown. Structure-alignment losses have a mixed record, so they rank below input-side changes. Evidence: ablated (mixed).

### 3.3 Leakage risk specific to this benchmark (our analysis, not from the literature)

A 3Di state at position i describes the local geometry of residue i relative to its nearest neighbor. That geometry carries information about the residue identity. If the 3Di track is visible at a masked position, the residue is easier to recover, and the MLM loss falls for reasons unrelated to sequence learning. Options: mask both tracks at the same positions, or predict residues only at positions where the 3Di track is also masked. The literature does not discuss this for the joint-token designs above. This must be checked before any 3Di result is interpreted.

### 3.4 Hidden-state probes for 3Di

Not found in the literature for a 3Di probe over hidden states. See section 6.2 for the closest layer-wise results.

---

## 4. Data: clustering, sampling, length, packing, duplication

### 4.1 Sampling bias and taxonomy

- **Ding and Steinhardt (ICLR 2024; [bioRxiv 2024.03.07.584001](https://www.biorxiv.org/content/10.1101/2024.03.07.584001.full.pdf)):** likelihoods from popular pLMs are systematically higher for some species than others, and species identity explains 26 to 69 percent of the variance in likelihood after controlling for protein type (summary). The authors trace this largely to uneven species coverage in the databases.
- Related: [Arcadia Science, 2024](https://research.arcadiascience.com/pub/result-protein-universe-phylogenetic-bias) reports the same bias in a write-up (summary only).

*Transfer.* Weight by AlphaFold DB cluster rather than by taxonomy or raw frequency. Evidence: reported.

### 4.2 Repetition and repeated UniRef data

- **Cheng et al. (NeurIPS 2024 spotlight; [arXiv 2411.02142](https://arxiv.org/abs/2411.02142), fetched abstract):** over 300 models from 3.5M to 10.7B parameters, trained on 5 to 200B unique tokens from a 939M-sequence dataset. Causal models show diminishing returns on UniRef. Masked models overfit when UniRef is repeated. Adding metagenomic sequences avoids the plateau. Code: github.com/cxysteven/ScalingProteinLM (in the abstract).
- **Our arithmetic:** the 10-minute run covers well under one pass (section, "Our arithmetic"), so repetition overfitting is not the binding constraint here. Dedup still matters for homology-diverse batches (section 4.3).

*Transfer.* Do not spend time on repetition controls for a single-pass run. Evidence: ablated in the abstract's scaling study, but at multi-epoch token budgets.

### 4.3 Cluster-balanced sampling

- **NVIDIA BioNeMo ESM-2 replication:** pretraining batches are formed by uniformly sampling each UniRef50 cluster, then taking a random UniRef90 member ([NVIDIA docs](https://docs.nvidia.com/bionemo-recipes/latest/main/datasets/uniprot/), summary only). This is a documented scheme, not an ablation.
- **AlphaFold DB clustering:** the AlphaFold DB documentation describes sequence clustering with MMseqs2 (AFDB50) followed by Foldseek clustering of representatives ([EBI training page](https://www.ebi.ac.uk/training/online/courses/alphafold/accessing-and-predicting-protein-structures-with-alphafold/accessing-predicted-protein-structures-in-the-alphafold-database/using-the-alphafold-database-for-analysis/navigating-the-structural-landscape)). A secondary source reports about 2.30 million Foldseek clusters from about 214 million structures (not verified against the primary paper).
- **UniRef90 versus UniRef50:** one secondary note on ESM-1v reports UniRef90 as most effective for pretraining (summary only, not verified). The ESM C blog uses 70 percent identity clusters (section 1.1). The Ankh3 work uses UniRef50 (section 1.12).

*Transfer.* Uniform sampling over clusters at 0.3 or 0.5 identity before drawing sequences is the cheapest reweighting. Expected effect: better coverage of families and possibly better homology-disjoint test loss. Evidence: anecdotal to reported. No identity-threshold ablation was found (section 7).

### 4.4 Homology-disjoint test sets and leakage

- **Hermann et al. (PMLR v261, 2024; [Beware of Data Leakage from Protein LLM Pretraining](https://proceedings.mlr.press/v261/hermann24a.html), summary only):** a naive cluster split versus a pretraining-aware split changes melting-point benchmark results by an average of 11.1 percent.
- **ICML 2026 "Evaluating out of distribution generalization of protein language models" ([listing](https://icml.cc/virtual/2026/70726), summary only):** overlap between pretraining and test data substantially raises test-set sensitivity while validation perplexity barely moves. Loss can miss generalization problems.
- **Twilight zone:** pLMs degrade at low identity on remote homology tasks ([PMC11344590](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11344590/)). Moldwin, Kabir and Shehu report three state-of-the-art models falling off below 35 percent pairwise identity (summary only).

*Transfer.* The benchmark's 0.3 identity split lies in the regime where pLMs degrade. Validation must also be disjoint at 0.3, otherwise the validation loss will not predict test behavior. Report test loss next to validation loss. Evidence: reported.

### 4.5 Length curricula and cropping

- ESM C uses a two-stage schedule, from context 512 to 2048 (section 1.1). Reported.
- DPLM-2 uses random cropping to 512 and crops low-pLDDT ends (section 1.7). Ablated only as part of the overall recipe.
- ESM-1b caps inputs at 1022 residues. CARP was trained on sequences up to 2048 ([CARP bioRxiv 2022.05.19.492714](https://www.biorxiv.org/content/10.1101/2022.05.19.492714.full.pdf)). About 2.6 percent of UniRef50 representatives were longer than 1022 residues in the March 2020 release (same source).
- Sequence packing and document masking for protein MLM were not found (section 7).

*Transfer.* Crop to 512 for most steps and lengthen near the end, as ESM C does. Our data range is 50 to 2,046 residues, so the crop choice changes what fraction of each protein is seen. Evidence: reported.

### 4.6 Structure quality filters

- DPLM-2 uses pLDDT above 85 for synthetic structures and crops pLDDT below 50 at the ends (section 1.7).

*Transfer.* Apply the same filter to 3Di tracks or down-weight low-pLDDT residues. Expected effect: less noisy structure targets, at the cost of data. Evidence: reported as a recipe choice, not ablated.

---

## 5. Efficiency and scaling

### 5.1 Packing and FlashAttention

- **Çelik and Xie (iScience, 2025; [PMC12481099](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12481099/)):** FlashAttention with sequence packing gives 4 to 9 times faster inference and 3 to 14 times lower memory on ESM models. The reported 6-fold training runtime cut combines activation checkpointing and offload, so it is not a clean measure of unpadding.
- **General LLM packing:** [arXiv 2407.09105](https://arxiv.org/html/2407.09105v6) reports packing with position IDs for variable-length sequences (not protein-specific).

*Transfer.* Use variable-length FlashAttention with cumulative sequence lengths, so that packed proteins do not attend across boundaries, and check this with a unit test. Measure tokens per second. Expected effect: more tokens in 10 minutes. Evidence: reported (inference).

### 5.2 Optimizers

- **Muon (general LLM):** [arXiv 2505.02222](https://arxiv.org/pdf/2505.02222) reports that Muon needs about 10 to 15 percent fewer tokens than AdamW to reach the same loss, with faster wall-clock convergence. LLM domain only.
- **nanoprot-esm2-M (third-party Hugging Face card, 2026):** Muon for matrices, AdamW for embeddings and scalars, weight decay 0.1. 147.6M parameters, 30 layers, hidden size 640, 20 heads, maximum length 512. 3,378 steps at 524,288 residues per step, about 54 minutes on 4 GPUs. Validation masked cross-entropy 3.49 bits per residue over three seeds. No baseline comparison on the card ([model card](https://huggingface.co/yagizdevre/nanoprot-esm2-M)).

*Transfer.* Try Muon for weight matrices with AdamW for embeddings and norms, retuning the learning rate, and compare on validation loss at equal time. Evidence: reported (LLM), anecdotal (pLM).

### 5.3 Scaling laws and small-model budgets

- **Serrano, Ciudad and Molina (arXiv 2406.07249, ICML 2024 workshop):** eight ESM-2-style models from 5M to 650M parameters on UniRef50 (about 64M sequences, 20.8B tokens), trained for one pass with 15 percent BERT-style MLM. Adam with betas (0.9, 0.98) and weight decay 0.01; learning rate 1e-5 (with and without decay); batch sizes 5K to 50K tokens. All runs plateau near training loss 2.455. The authors fit optimal model size scaling as C^0.27 and optimal tokens as C^0.71. A 35M model trained on 6.5B tokens reaches perplexity 11.43, against 10.98 for ESM2-15B and 10.81 for xTrimoPGLM. The 35M run took about one hour on a single H100 (fetched).
  *Note:* the learning rate of 1e-5 is very low, so the plateau may reflect the schedule as much as capacity. The authors' interpretation should be read with that in mind.
- **Cheng et al. (section 4.2):** separate scaling laws for causal and masked objectives, plus a transfer phenomenon from causal to masked training measured in effectively transferred tokens.

*Transfer.* The regime is compute-limited and single-pass. Throughput (section 5.1) and step efficiency (section 5.2) matter more than data quantity for this run. Evidence: ablated in the fits, single study for the plateau.

### 5.4 Efficiency of attention backends

- FastPLMs cards disagree on default backend and warn that flex attention can differ numerically from SDPA (section 1.14). Check parity before using a backend for training.

---

## 6. Evaluation: loss, downstream, and layers

### 6.1 Does lower MLM loss predict downstream performance?

- Within one family and one data distribution, lower held-out perplexity usually tracks better contact, structure and fitness performance. Serrano et al. and Cheng et al. both compare models at matched objective (sections 5.3, 4.2).
- The overlap study (section 4.4) shows validation loss can stay flat while test sensitivity changes.
- No study was found that quantifies the correlation coefficient between MLM loss and downstream results across protein models.

### 6.2 Which layers carry structure?

- **Joeres et al. (arXiv 2608.12090, 2026):** 13 pLMs on 15 downstream tasks from 11 datasets. The last layer rarely gives the best embedding. The best layer depends on task: shallow layers did better on deep mutational scan datasets and deeper layers on diverse natural proteins. For ESM-2 and ProGen2, pretraining loss falls almost monotonically with depth (summary only; the arXiv date is unusual, so check the record before citing).
- **Kumar and Jha (arXiv 2512.00376):** intermediate ESM-2 layers for kinase function. The results were not retrieved.
- **Contact information in attention:** ESM-2 contact prediction reads attention maps through a learned linear projection and sigmoid (summary only). This is a readout over attention, not over hidden states.

*Transfer.* A 3Di linear probe should sweep all 36 layers, not only the last. The Joeres result suggests that the peak may be mid-depth and task-specific. Evidence: reported (one study, not structure-specific).

---

## 7. Not found

- Convergence speed of MLM on proteins at equal wall-clock time, for any masking rate or objective.
- A mask-rate sweep for protein MLM reporting validation loss and downstream results together. The 15 versus 40 percent result (section 2.1) is for BERT on English text.
- An ablation of 3Di tokens for sequence-only test loss, or for homology-disjoint test loss.
- A remote-homology ablation of SaProt's structure tokens. The README mentions only a sequence-only comparison with ESM-2 35M.
- A study varying sequence-identity clustering thresholds in pLM pretraining.
- Sequence packing or document masking for protein MLM. Packing results were found only for ESM inference (section 5.1) and LLMs.
- Token dropout ablations for protein models.
- ProtTrans successors or updated ProtGPT2 releases from 2024 to 2026.
- The masking rate of the original ModernBERT paper, verified from its text.
- The ESM3 tokenizer, codebook size and learning rate (full text not retrieved).
- The full text of AMPLIFY, E1, ProGen3, DSM and the Structure-Aware Masking paper (fetches were rate limited or returned not found; summaries are marked).
- A layer-wise 3Di probe.
- Intermediate-layer results for Kumar and Jha (2026).

---

## 8. Ranked ideas for the 10-minute benchmark

Order is by expected value per unit of implementation cost, with infrastructure first. "Gain" means the expected change in test loss or in loss per wall-clock minute; all values are hypotheses to test with validation loss at equal time.

| Rank | Idea | Source | Evidence | Expected gain | Cost to implement | Risk | Suggested order |
|---|---|---|---|---|---|---|---|
| 1 | Varlen FlashAttention with packed proteins and boundary-checked attention | Çelik and Xie 2025 (PMC12481099) | Reported (inference) | More tokens per 10 minutes; large if the current loader pads heavily | Moderate | Cross-protein attention leak if boundaries are wrong; add a unit test | 1 |
| 2 | Resample masks every batch | RoBERTa, arXiv 1907.11692 | Ablated, small | Small | Trivial | None | 1 |
| 3 | Mask rate varied per sequence (for example 15 to 30 percent, uniform) | Ankh3 (arXiv 2505.20052); ESM3 preprint; Wettig et al. (arXiv 2202.08005) | Reported; Wettig ablated on BERT downstream | Uncertain sign on 10-minute validation loss; more targets per step | Trivial | Harder task may slow early loss; select by time-matched validation | 1 |
| 4 | Uniform sampling over AlphaFold DB clusters at 0.3 or 0.5 identity | NVIDIA ESM-2 documentation; Ding and Steinhardt (bioRxiv 2024.03.07.584001) | Anecdotal to reported | Possibly better homology-disjoint test loss | Low (precomputed clusters) | Reduces effective data diversity per batch if clusters are small; check test | 2 |
| 5 | Two-stage context schedule (512, then 2048 near the end) | ESM C blog (evolutionaryscale.ai) | Reported, not ablated | Throughput early; loss effect unknown | Low | Minor | 2 |
| 6 | Foldseek 3Di as an auxiliary prediction target, residue loss kept as the metric | ISM (ICLR 2025, summary); DPLM-2 (arXiv 2410.13782) | Ablated (DPLM-2, structure generation); reported (ISM) | Unknown on residue test loss; plausible for homology generalization | Moderate (one Foldseek pass over 1.75M proteins) | Leakage if the 3Di track is visible at the target position (section 3.3) | 3 |
| 7 | pLDDT filter and end-cropping of structures for the 3Di track | DPLM-2 (arXiv 2410.13782) | Recipe choice, not ablated | Less noisy structure targets | Low | Less data | 3 |
| 8 | Muon for weight matrices, AdamW for embeddings and norms | Muon, arXiv 2505.02222; nanoprot card | Reported (LLM); anecdotal (pLM) | Claimed 10 to 15 percent token saving in LLMs; unknown for proteins | Low to moderate | Learning-rate retuning and instability | 3 |
| 9 | Bucket Masking (spatially grouped masks) | Walton, Goel and Aghazadeh, arXiv 2605.16581 | Reported (downstream fitness, summary) | Unknown on MLM loss | Moderate (neighbor sets per structure) | Changes the objective; validate on MLM loss first | 4 |
| 10 | Intermediate-layer 3Di linear probe (evaluation only) | Joeres et al., arXiv 2608.12090 | Reported | None to training; better readout | Low | None | 4 |
| 11 | Structure-alignment auxiliary losses (AlphaFold2 alignment) | arXiv 2505.16896 | Ablated, mixed or negative | Probably negative | Moderate | Hurts baseline in reported ablation | Do not prioritize |
| 12 | Warm start from a sequence model with LoRA | DPLM-2 (arXiv 2410.13782) | Ablated (structure generation) | Large in their setting | Not applicable | Benchmark is from scratch | Not applicable |
| 13 | Retrieval over homologs, biophysics pretraining, recursive depth | E1; METL; recAMPLIFY | Reported | Out of scope or unclear for this benchmark | High | Retrieval conflicts with homology-disjoint test; recursion conflicts with the fixed 36-block spec | Not recommended now |

---

## Sources cited

- ESM C blog: https://www.evolutionaryscale.ai/blog/esm-cambrian
- ESM3 preprint: https://www.biorxiv.org/content/10.1101/2024.07.01.600583.full.pdf
- ESM3 blog: https://www.evolutionaryscale.ai/blog/esm3
- AMPLIFY preprint: https://www.biorxiv.org/content/10.1101/2024.09.23.614603.full.pdf
- AMPLIFY repository: https://github.com/chandar-lab/AMPLIFY
- Mila AMPLIFY summary: https://mila.quebec/en/insight/is-bigger-always-better-democratizing-ai-protein-discovery
- recAMPLIFY listing: https://papers.cool/arxiv/2610.26747
- recAMPLIFY model: https://huggingface.co/chandar-lab/recamplify
- ProGen3 preprint: https://www.biorxiv.org/content/10.1101/2025.04.15.649055v1
- ProGen3 poster: https://neurips.cc/virtual/2025/poster/115098
- xTrimoPGLM: https://arxiv.org/abs/2401.06199
- DPLM-2: https://arxiv.org/html/2410.13782v1
- SaProt preprint: https://www.biorxiv.org/content/10.1101/2023.10.01.560349.full.pdf
- SaProt ICLR poster: https://iclr.cc/virtual/2024/poster/19394
- SaProt README mirror: https://www.gitlink.org.cn/NSCCN/SaProt
- ProstT5 preprint: https://www.biorxiv.org/content/10.1101/2023.07.23.550085.full.pdf
- ProstT5 PMC copy: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11616678/
- ProSST abstract: https://proceedings.neurips.cc/paper_files/paper/2024/hash/3ed57b293db0aab7cc30c44f45262348-Abstract.html
- E1 preprint: https://www.biorxiv.org/content/10.1101/2025.11.12.688125.full.pdf
- E1 ICML listing: https://icml.cc/virtual/2026/72071
- E1 README: https://raw.githubusercontent.com/Profluent-AI/E1/main/README.md
- Ankh3: https://arxiv.org/html/2505.20052v1
- METL preprint: https://www.biorxiv.org/content/10.1101/2024.03.15.585128.full.pdf
- DSM: https://arxiv.org/abs/2506.08293
- FastPLMs: https://github.com/Synthyra/FastPLMs
- Protify: https://pypi.org/project/Protify/
- ProtGPT2: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9329459/
- Wettig et al.: https://arxiv.org/abs/2202.08005
- ACL Anthology entry: https://aclanthology.org/2023.eacl-main.217
- RoBERTa: https://arxiv.org/pdf/1907.11692
- Structure-Aware Masking: https://arxiv.org/pdf/2605.16581
- Span mask on 3D chains: https://arxiv.org/abs/2402.01481
- Structure-Aligned PLM: https://arxiv.org/pdf/2505.16896
- Ding and Steinhardt: https://www.biorxiv.org/content/10.1101/2024.03.07.584001.full.pdf
- Arcadia Science write-up: https://research.arcadiascience.com/pub/result-protein-universe-phylogenetic-bias
- Cheng et al.: https://arxiv.org/abs/2411.02142
- Serrano et al.: https://arxiv.org/html/2406.07249v1
- Hermann et al.: https://proceedings.mlr.press/v261/hermann24a.html
- ICML 2026 OOD listing: https://icml.cc/virtual/2026/70726
- Twilight zone study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11344590/
- NVIDIA ESM-2 datasets: https://docs.nvidia.com/bionemo-recipes/latest/main/datasets/uniprot/
- AlphaFold DB structural landscape: https://www.ebi.ac.uk/training/online/courses/alphafold/accessing-and-predicting-protein-structures-with-alphafold/accessing-predicted-protein-structures-in-the-alphafold-database/using-the-alphafold-database-for-analysis/navigating-the-structural-landscape
- CARP: https://www.biorxiv.org/content/10.1101/2022.05.19.492714.full.pdf
- Packing with FlashAttention (general): https://arxiv.org/html/2407.09105v6
- ESM efficiency study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12481099/
- Muon efficiency: https://arxiv.org/pdf/2505.02222
- nanoprot model card: https://huggingface.co/yagizdevre/nanoprot-esm2-M
- Joeres et al.: https://arxiv.org/pdf/2608.12090
- Kumar and Jha: https://arxiv.org/pdf/2512.00376
- Foldseek (Nature Biotechnology 2024): https://link.springer.com/10.1038/s41587-023-01773-0
- ModernBERT: https://arxiv.org/abs/2412.13663
- ESM-2 (Science 2023): https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2.full.pdf
- ISM and ProteinJEPA listings: https://icml.cc/virtual/2026/72047

---

## Section 7 gaps restated for the benchmark owner

No published result was found that tells us, for a 10-minute single-pass MLM on 3Di-annotated AlphaFold DB proteins, which of the ranked ideas reduces test loss. The first two rows (packing and per-batch masks) carry the least risk and can be tested in one short sweep. The structure-target rows need the leakage check in section 3.3 before any result is read.
