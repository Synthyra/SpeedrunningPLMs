# Protein pretraining ideas for the 10-minute speedrun

Date: 2026-10-30

Scope: protein language model (pLM) pretraining findings from 2025 to 2026, read against the fixed benchmark (36 blocks, width 768, MLP 2,048, 15 percent of residues replaced by MASK, 20-way residue loss, 10 minutes on one GH200). Read first: `ideas/README.md` and `ideas/protein_pretraining.md` (2026-10-28), which already cover ESM C, ESM3, AMPLIFY, ProGen3, DPLM, SaProt, E1, Ankh3 and the masking studies. This file adds what those did not cover, and re-ranks the overlap. Abbreviations: MLM (masked language modeling), CE (cross entropy), AFDB (AlphaFold DB), SGM (structure-guided masking, ProteinSage's term), SCL (structural causal learning, ProteinSage's term), JEPA (joint-embedding predictive architecture), BPE (byte-pair encoding).

Every gain below is a hypothesis to test on validation loss at equal wall-clock time. Ranking is by expected gain per implementation hour. A one-seed run resolves about 0.0015 nats (`ideas/README.md`), so most literature effects here are below what one run can see.

## Rule questions to settle before the ranking is used

1. **3Di auxiliary head.** The brief says auxiliary losses are free and a 3Di-prediction head is in the rules. `ideas/README.md` ("Outside the rules") says the objective is fixed at MASK-only MLM and a 3Di target is not a candidate unless Logan opens the objective. This file ranks it as in the rules, following the brief. Logan should settle which holds.
2. **Placement within the 15 percent.** The brief fixes the share of MASK (15 percent of residues), not where the MASKs fall. The catalog reads span masking as fixed. This file treats structure-guided or span placement inside the 15 percent as in the rules, following the brief's wording. Confirm.
3. **Coordinates.** The benchmark ships 3Di strings (`benchmark/data.py`, `train_three_di.u8`), not coordinates or contact maps. Idea 4 needs residue contacts. Without Logan's word to source AFDB coordinates as a new data input, idea 4 is blocked.
4. **Stream check.** Idea 1 depends on how a protein longer than 1,024 residues is handled by the packed window reader (cut into two windows, or kept whole). Check `benchmark/data.py` and the reader before the run.

## What the record already settles (not re-proposed)

Packed windows with a document mask (batch family), the value residual, RoPE with ALiBi-style slopes, NorMuon, one protein per family per pass (013: tie with 012), 512-token windows (009: lose on long proteins), a short-first curriculum (027: invalid at the warmup cap), EMA, U-net skips and gates (all lost). The data reader already exposes family ids and 3Di states.

## Ranked ideas (in the rules first, then out of the rules)

### 1. Whole run at 2,048-residue windows, one compiled shape (in the rules)

- **Source:** EvolutionaryScale, ESM C blog, 2024 (https://www.evolutionaryscale.ai/blog/esm-cambrian). Context 512 for most steps, then 2,048 for the final 500K of 1.5M steps (reported; no ablation). In-house: `ideas/README.md` rows 009 and 028 (the value residual's gain grows with length, 0.025 nats over 500 residues).
- **Evidence:** No protein MLM study found reports 2,048 against 1,024 at equal time. The direction is supported by the record: shorter windows lost on long proteins (009), and the long-protein bin is where the record is weakest. Hypothesis: 0.002 to 0.01 nats on proteins over 1,024, under 0.005 overall. Not measured.
- **What it changes:** 32 windows of 2,048 instead of 64 of 1,024 (the same 65,536 tokens a step). Proteins from 1,025 to 2,046 residues stay whole if the reader keeps them whole.
- **Expected cost:** one compile at the new shape (the record compiles in 106 to 182 s, against a 180 s cap, so pilot the compile first). Step time: our arithmetic, not a source. Attention cost is set by the sum of squared document lengths per window. For documents under 1,024, halving the window count while doubling the window length leaves that total unchanged, so the step-time change comes from long documents only. Activation memory per token is unchanged; the block mask doubles in length. Data preparation: none.
- **Implementation sketch:**
```python
class Window2048(Strategy):  # family "batch", variant "window_2048": one fixed shape for the whole run
    family = "batch"
    def batch_plan(self, stream: StreamReader) -> BatchPlan:  # data-reading hook (new; not in HOOKS yet)
        # 32 windows of 2,048 hold the 65,536 tokens of 64 windows of 1,024
        return stream.pack(window=2048, windows_per_step=32, keep_whole=True)  # 2,046 is the longest protein
    def build(self, model: nn.Module, max_tokens: int) -> None:
        assert max_tokens >= 2048, "the flex block mask is sized to the window"  # fail loudly
```
- **Risk:** A late switch from 1,024 to 2,048 (the ESM C pattern) compiles twice, which the record's 027 showed the cap rejects. Keep one shape from step 0. If long documents dominate the attention work, step time rises and the gain is eaten. Pilot 60 s, then watch the rate.

### 2. A plain 3Di cross-entropy head at half depth (in the rules per the brief; the catalog says outside)

- **Source:** Wang et al., DPLM-2, arXiv 2410.13782 (2024; ICLR 2025): structure and sequence data both help, in a structure-generation ablation (sc-TM at length 500: neither 0.4511; sequence only 0.7673; synthetic structures only 0.8288; both 0.9012). ISM, Implicit Structure Model, ICLR 2025 (summary only, from the 2026 survey; not re-read here): a sequence model trained to predict structure tokens, with reported gains. Negative control: Structure-Aligned pLM, arXiv 2505.16896 (2025): alignment to AlphaFold2 representations underperformed the ESM2 baseline in its ablation.
- **Evidence:** No ablation on residue test loss exists. Sign unknown. The prior from ProteinJEPA (idea 3) says an added objective can cost residue loss at small budgets (eval MLM 2.448 with the auxiliary against 2.443 compute-matched, 8M model).
- **What it changes:** One linear map from a middle block to 20 Foldseek states, trained alongside MLM. The input stays residues only, so 3Di never reaches the embedding path.
- **Expected cost:** the head is 768 x 20 + 20 = 15,380 parameters, about 31 kFLOP a token against about 1.2 GFLOP for the model (6 x 198M), so under 0.01 percent of compute. The real cost is plumbing: `auxiliary_loss(model, states, labels)` receives MLM labels, so the batch must carry `three_di`, and the hook signature changes. Compile: a few seconds.
- **Implementation sketch:**
```python
class ThreeDiHead(Strategy):  # family "auxiliary" (new); a 3Di head at half depth
    family = "auxiliary"
    def build(self, model: nn.Module, max_tokens: int) -> None:
        self.head = nn.Linear(benchmark.WIDTH, 20)  # 768 -> 20 Foldseek states, 15,380 parameters
    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        return [ParameterGroup("three_di", list(self.head.parameters()), "adam", 3e-3)]
    def auxiliary_loss(self, model, states, labels, three_di):  # three_di: (b, l), new argument
        logits = self.head(states[18])  # (b, l, 20); sweep block 9, 18, 27, 36
        valid = three_di != benchmark.NO_STATE  # residues only: CLS, EOS and PAD excluded
        return 0.1 * F.cross_entropy(logits[valid], three_di[valid])  # weight swept over 0.05, 0.1, 0.3
```
- **Risk:** Sign unknown; the head may take capacity from residues. Leakage check (`ideas/protein_pretraining.md` section 3.3) passes only if 3Di never enters the input, which this sketch guarantees. Confirm the rule (question 1) before any GPU time.

### 3. Latent prediction against the student's own last layer, stop-gradient (in the rules)

- **Source:** Ofer, Shahaf and Linial, ProteinJEPA, arXiv 2605.07554 (2026). Controlled study (EMA teacher, 35M and 150M ESM2, 67K steps, batch 1,024, one pass of UniRef50): against compute-matched MLM over 114 task-seed comparisons, median +0.0013 nats, 54 wins, 41 losses, 22 ties (Wilcoxon p 0.035; sign p 0.059). Per-seed sign tests were not all significant (one seed p 0.031). Eval MLM at 8M: 2.448 with the auxiliary against 2.443 compute-matched. Wall-clock study (8 hours, random init): the all-position variant ran 78,818 steps against 134,413 for MLM-only; the masked-position variant's step count is not in the text read.
- **Evidence:** The median gain sits inside one seed's band (0.0015). Expect a tie at best.
- **What it changes:** The controlled recipe's 17 percent step overhead comes from an EMA teacher forward over the full sequence. The wall-clock variant uses the student's own last layer with gradients blocked, so no second forward. Its extra cost is one predictor on the 15 percent masked positions (about 0.1 percent of FLOPs, our arithmetic). The variant needs collapse control: the paper uses SIGReg with 256 projections (weight 0.45), which is not in this repository.
- **Expected cost:** 2 to 3 hours including SIGReg. Compile: unknown; a few seconds more likely.
- **Implementation sketch:**
```python
class LatentPrediction(Strategy):  # family "auxiliary", variant "latent_last" (new)
    family = "auxiliary"
    def build(self, model: nn.Module, max_tokens: int) -> None:
        self.predictor = SwiGLUPredictor(benchmark.WIDTH, 2048)  # one layer, ~4.7M parameters; check Table 1
    def auxiliary_loss(self, model, states, labels, masked):  # masked: (b, l) bool, the MLM positions
        target = norm(states[-1]).detach()  # the student's own last layer, no gradient
        pred = norm(self.predictor(states[18]))  # predict the last layer from mid depth (assumed; check paper)
        cosine = 1 - (pred * target).sum(-1)[masked].mean()  # masked positions only
        return 0.45 * cosine + sigreg(states[-1])  # SIGReg collapse control: a helper to write, 256 projections
```
- **Risk:** Collapse without working SIGReg; a gain inside noise; extra compile. Pilot once on the GPU before a queue.

### 4. Structure-guided placement inside the 15 percent (in the rules; needs contacts)

- **Source:** ProteinSage, bioRxiv 10.64898/2026.03.17.712034 (v2, 2026). 150M model on SageDB-Large, objective MLM then SGM then SCL: Recent 31.1 then 38.7 then 44.2; CASP14 34.7 then 44.1 then 47.5 (contact-map metrics, not MLM loss). Under the fixed 15 percent, a 3 percent SGM plus 12 percent random split beat higher SGM shares. Contact threshold 6 angstrom beat 5 and 7. Bucket Masking, arXiv 2605.16581 (2026): ESM-2 650M, one model per ProteinGym family. Table 1 means, GM span against Bucket (our subtraction): regime +0.055, mutation +0.025, position +0.066, neighborhood +0.004 in Spearman. The authors' control says placement, not span size, drives the gain.
- **Evidence:** Contact and fitness metrics, not MLM loss. ProteinSage's baselines are published checkpoints and reported token counts, not retrained controls. The placement effect is a hypothesis on residue loss.
- **What it changes:** The masked set keeps its 15 percent size but chooses positions by structural neighborhood. The evaluator and the test masks are unchanged, so the test loss stays comparable.
- **Expected cost:** the blocker is data. Contacts need CA coordinates for 1.75M AFDB proteins, which the benchmark does not ship. Downloading and reducing 1.75M structures is a new data input, likely hours of I/O and CPU (estimate, not measured). Per-step cost: none, since placement runs in the data reader.
- **Implementation sketch:**
```python
class StructurePlacement(Strategy):  # family "mask_placement" (new); the 15 percent total stays fixed
    family = "mask_placement"
    def batch_plan(self, stream: StreamReader, contacts: ContactMap) -> BatchPlan:  # data-reading hook (new)
        # contacts[i]: partners j with |i - j| >= 6 and CA-CA distance under 6 angstrom
        sgm = pick_pairs(contacts, share=0.03)  # SGM positions: one endpoint per pair (sampler: check paper)
        rest = sample_disjoint(stream.positions, exclude=sgm, share=0.12)  # the random MLM share
        return stream.mask(sgm | rest)  # total 15 percent of residues
```
- **Risk:** The placement favors contact-rich, often buried residues, so the training distribution drifts from the test loss at random positions. The 12 percent random share limits this, but it is a bias to check against validation at equal time. Blocked until question 3 is answered.

### 5. Pair-trailer structural causal objective, ProteinSage SCL (in the rules; expensive)

- **Source:** ProteinSage, bioRxiv 10.64898/2026.03.17.712034 (v2, 2026), as in idea 4. SCL over MLM at 300 G tokens: CASP14 +10.1, CASP15 +8.4, CAMEO +11.0, Recent +6.6. At 500 G tokens against an MLM model trained for 1.0 T tokens: +3.5, +2.5, +4.9, +0.6.
- **Evidence:** The gains are large at 300 G tokens and shrink at 500 G against a longer MLM baseline. Our run is about 1e8 tokens (`ideas/protein_pretraining.md`, our arithmetic), far below both settings. No number at our budget.
- **What it changes:** Each selected residue pair adds a trailer with source and target slots, predicted from the base sequence with directed attention and no trailer-to-trailer leakage. The base residue loss is unchanged, but the sequence grows, so fewer residues fit in 65,536 tokens a step.
- **Expected cost:** high. It needs idea 4's contacts, a custom attention mask in the flex kernel, and a longer sequence. Estimate 10 or more hours and a measured step-time penalty. Trailer length per pair is not in the visible text.
- **Implementation sketch:**
```python
class PairTrailers(Strategy):  # family "auxiliary" with a data-path change (new)
    family = "auxiliary"
    def batch_plan(self, stream, contacts):  # appends one trailer per selected pair (data-reading hook)
        pairs = pick_pairs(contacts, share=0.03)  # the SGM pairs of idea 4
        return stream.append_trailers(pairs, directed=True, swap_roles_each_step=True)
    def auxiliary_loss(self, model, states, labels, trailer_mask):  # trailer slots only
        return 1.0 * F.cross_entropy(self.head(states)[trailer_mask], trailer_targets)  # weight 1 in the paper
```
- **Risk:** Lowest expected return per hour of the in-rule ideas. Attention leakage between trailers is the main bug source; test it with a unit test. Keep it below idea 4 in the queue.

### 6. Uniform sampling over families or clusters (in the rules; tie in-house)

- **Source:** NVIDIA BioNeMo, ESM-2 pretraining data (uniform over UniRef50 clusters, then a random UniRef90 member; documentation, summary only: https://docs.nvidia.com/bionemo-recipes/latest/main/datasets/uniprot/). Fournier et al., bioRxiv 2024.09.23.614603 (summary only): UniRef50 training hurts when the validation set is not clustered. Ding and Steinhardt, ICLR 2024: species identity explains 26 to 69 percent of the variance in likelihood (summary).
- **Evidence:** In-house, direct: row 013 (one protein per family per pass) against 012: 2.7532 against 2.7531 at val_select, a tie. The record says the budget is not family-limited. Expected gain: about zero. Not worth a GPU slot.
- **What it changes:** Draw a family uniformly, then a member, instead of by raw frequency. Family ids are already in the data store.
- **Expected cost:** about 30 minutes, reader only; no compile or step change.
- **Implementation sketch:**
```python
class FamilyUniform(Strategy):  # family "sampling" (new)
    family = "sampling"
    def batch_plan(self, stream: StreamReader) -> BatchPlan:  # data-reading hook (new)
        families = stream.family_ids  # present in the data store
        return stream.draw(families=uniform_over(families), member=uniform)  # per window until the budget fills
```
- **Risk:** Tie by prior evidence; a repeat would spend a run to confirm a known null. Keep only if the proposers have no other reader change to test.

### 7. Variable or higher mask rate (out of the rules: the 15 percent is fixed)

- **Source:** Ankh3, arXiv 2505.20052 (2025): MLM at 15, 20 and 30 percent sampled uniformly, plus sequence completion; the authors attribute gains to both together (not isolated). FusOn-pLM, Vincoff et al., Nature Communications 2025 (https://www.nature.com/articles/s41467-025-56745-6): cosine schedule from 15 to 40 percent (abstract; sweep numbers not retrieved). Bucket Masking (2026): rate U(0.15, 0.75) per epoch (idea 4's source). Wettig et al., arXiv 2202.08005 (2022; EACL 2023): on BERT-large, 40 percent beat 15 percent on GLUE and SQuAD (exact scores not retrieved; English text, not protein).
- **Evidence:** No isolated protein MLM number. Sign at 10 minutes uncertain: more targets per step, a harder task.
- **What it changes:** Per-window mask share drawn from a range.
- **Expected cost:** trivial to code; a rule change.
- **Implementation sketch:**
```python
class VariableRate(Strategy):  # family "mask_rate" (new); OUT OF RULES: the 15 percent is fixed
    family = "mask_rate"
    def batch_plan(self, stream: StreamReader) -> BatchPlan:
        share = rng.uniform(0.15, 0.30)  # per window; the brief fixes this at 0.15
        return stream.corrupt(share=share, replace_with="MASK")
```
- **Risk:** Breaks the fixed objective and the comparability of the test loss. Do not run without Logan's word to open the objective.

### 8. Diffusion-style corruption at high mask rates (out of the rules)

- **Source:** DSM, Hallee et al., arXiv 2506.08293 (2025; abstract only): ESM2 architecture with a masked forward diffusion process; reported coherent generation at 90 percent corruption and representations at or above similarly sized pLMs. Training schedule, optimizer and token count not retrieved.
- **Evidence:** Reported, not isolated for MLM loss at short budgets. At 90 percent corruption, a 10-minute run would spend most of its gradient on near-empty inputs (our arithmetic).
- **What it changes:** Sampled corruption rates from near 0 to near 1, with a diffusion-style loss weight.
- **Expected cost:** moderate (a new loss weight and schedule); a rule change.
- **Implementation sketch:**
```python
class MaskedDiffusion(Strategy):  # family "objective" (new); OUT OF RULES
    family = "objective"
    def batch_plan(self, stream: StreamReader) -> BatchPlan:
        t = rng.uniform(0, 1)  # per sequence noise level
        return stream.corrupt(share=t, replace_with="MASK")  # loss weight 1 / t is the usual form (check DSM)
```
- **Risk:** Changes the objective and the metric. Only as a separate study.

### 9. Merged subword tokens (out of the rules: the 20-way residue label space is fixed)

- **Source:** Reduced-alphabet tokenization with BPE, Bioinformatics 42(Suppl 1) btag226 (2026): shorter inputs and faster training; no MLM-loss number retrieved. Zenodo vocabulary-selection study (undated in the search result): amino-acid-level models score better on normalized perplexity; minimum description length tracks downstream results best among intrinsic metrics. PETA, arXiv 2310.17415 (2023): per-amino-acid baseline perplexity 7.78; BPE vocabularies tested from 50 to 3,200 (older than the window, included for the number).
- **Evidence:** Both 2025-2026 sources favor per-residue tokens on perplexity or on sequence-level metrics. The rule makes the question moot.
- **What it changes:** Shorter sequences, so more proteins per step, against a different label space. The test loss would no longer be per residue.
- **Expected cost:** moderate; out of rules.
- **Implementation sketch:**
```python
class MergedTokens(Strategy):  # family "tokenizer" (new); OUT OF RULES
    family = "tokenizer"
    def embed(self, model, input_ids, X):  # input_ids: BPE ids over a reduced alphabet
        return self.table[input_ids]  # label space changes from 20 residues; the loss is no longer per residue
```
- **Risk:** The evaluator scores residues; a merged vocabulary breaks the comparison. Not recommended.

### 10. Input-side 3Di joint token, SaProt style (out of the rules unless test proteins carry 3Di)

- **Source:** Su et al., SaProt, bioRxiv 2023.10.01.560349, ICLR 2024 (residue and Foldseek 3Di tokens combined; about 40M pairs; reported downstream gains; the README reports a residue-only variant comparable to ESM-2 35M, summary only).
- **Evidence:** Downstream, not MLM loss. No matched test-loss comparison.
- **What it changes:** The input vocabulary becomes residue-plus-state pairs. The brief says only training proteins carry 3Di, so test inputs would lack the states and the evaluation would change.
- **Expected cost:** moderate. Out of rules unless the test set carries 3Di, which the brief does not say.
- **Implementation sketch:**
```python
class JointToken(Strategy):  # family "embedding" (new); OUT OF RULES unless test has 3Di
    family = "embedding"
    def embed(self, model, input_ids, X):  # input_ids: residue * 20 + state, a 400-token vocabulary
        return self.table[input_ids]  # test inputs need the state track too
```
- **Risk:** Evaluation mismatch and a test-time dependency on 3Di. Not recommended.

### 11. A smaller model at this budget (out of the rules: 36 blocks and width 768 are fixed)

- **Source:** Serrano, Ciudad and Molina, arXiv 2406.07249 (2024; ICML 2024 workshop): one-pass MLM on UniRef50 (20.8B tokens) across 5M to 650M models; fitted optimal model size scales as C^0.27 and optimal tokens as C^0.71. A 35M model trained on 6.5B tokens reaches perplexity 11.43, against 10.98 for ESM2-15B (their numbers). Their learning rate of 1e-5 is low, which the authors' plateau reading may reflect.
- **Evidence:** A fitted exponent from one study at 10^10 tokens. Not tested at 10^8 tokens. As a diagnostic only: at this budget the 198M model may be large for the compute, and depth and width are fixed by the rule.
- **What it changes:** Nothing under the rules. It would require a different benchmark.
- **Expected cost:** not applicable under the rules.
- **Implementation sketch:** not applicable (the benchmark fixes depth and width).
- **Risk:** Recommending it would change the benchmark. Mention only.

## Considered and dropped

- **Per-residue loss weighting by conservation or entropy.** The nearest source (BALM, an antibody language model, year not verified) changes which positions are masked, not how the loss is weighted, and gives no number. A weighted training loss also differs from the unweighted test loss.
- **Retrieval or recursive depth (E1, recAMPLIFY).** E1 retrieves homologs, which conflicts with a homology-disjoint test. recAMPLIFY's recursion conflicts with the fixed 36 blocks.
- **Warm start (DPLM, ESM).** The benchmark is from scratch.
- **ProtGPT3.** Model cards only; causal; no training recipe found.
- **Cramming protein LMs in 24 GPU hours (Frey, bioRxiv 2024.05.14.594108).** Closest single-GPU recipe, but the search returned downstream results, not a validation-loss ablation; not read in full.
- **Long-protein sparse attention (bioRxiv 2025.05.29.656907).** Reports RoPE degradation beyond about 4,000 residues, above our 2,046 maximum; no transfer.
- **Token-based packing (PMC12481099, iScience 2025).** Packing is in the record already. The reported training-time cut (up to 53 percent) combines packing with activation checkpointing and offload, so it is not a clean measure.

## Sources

- ESM C blog, 2024: https://www.evolutionaryscale.ai/blog/esm-cambrian
- ProteinSage, bioRxiv 10.64898/2026.03.17.712034 (v2), 2026: https://www.biorxiv.org/content/10.64898/2026.03.17.712034v2.full
- Structure-Aware Masking (Bucket Masking), arXiv 2605.16581, 2026: https://arxiv.org/html/2605.16581v1
- ProteinJEPA, arXiv 2605.07554 (v2), 2026: https://arxiv.org/html/2605.07554v2
- DPLM-2, arXiv 2410.13782, 2024 (ICLR 2025): https://arxiv.org/html/2410.13782v1
- Structure-Aligned pLM, arXiv 2505.16896, 2025: https://arxiv.org/pdf/2505.16896
- Ankh3, arXiv 2505.20052, 2025: https://arxiv.org/html/2505.20052v1
- FusOn-pLM, Nature Communications 2025: https://www.nature.com/articles/s41467-025-56745-6
- Wettig et al., arXiv 2202.08005, 2022: https://arxiv.org/abs/2202.08005
- DSM, arXiv 2506.08293, 2025: https://arxiv.org/abs/2506.08293
- Reduced alphabets and BPE, Bioinformatics btag226, 2026: https://academic.oup.com/bioinformatics/article/42/Supplement_1/btag226/8726310
- Vocabulary selection, Zenodo 23105495 (undated in search): https://zenodo.org/records/23105495
- PETA, arXiv 2310.17415, 2023: https://arxiv.org/pdf/2310.17415
- SaProt, bioRxiv 2023.10.01.560349, ICLR 2024: https://www.biorxiv.org/content/10.1101/2023.10.01.560349.full.pdf
- Serrano et al., arXiv 2406.07249, 2024: https://arxiv.org/html/2406.07249v1
- Fournier et al., bioRxiv 2024.09.23.614603 (v2; not fetched, rate limited): https://www.biorxiv.org/content/10.1101/2024.09.23.614603v2.full.pdf
- NVIDIA BioNeMo ESM-2 data: https://docs.nvidia.com/bionemo-recipes/latest/main/datasets/uniprot/
- Long-protein sparse attention, bioRxiv 2025.05.29.656907: https://www.biorxiv.org/content/10.1101/2025.05.29.656907v2.full
- Packing and FlashAttention, PMC12481099 (iScience 2025): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12481099/
- Frey, Cramming PLM training, bioRxiv 2024.05.14.594108: https://www.biorxiv.org/content/10.1101/2024.05.14.594108v1.full.pdf

## Caveats

- ProteinJEPA, ProteinSage and Bucket Masking are 2026 preprints. Their numbers are as reported; ProteinSage's comparisons use published checkpoints, not matched retraining.
- Several items come from search summaries, not full texts: ISM, Fournier et al., FusOn-pLM, the Zenodo study, DSM, Ankh3's composition, and the Bioinformatics paper. Each is marked where it is used. Check a number against its link before quoting it outside this file.
- The ranking is a judgment from effect size against implementation cost. None of the in-rule ideas has a published effect on residue test loss at a 10-minute budget. The first pilots should be idea 1 (cheapest, grounded in the record's own length result) and idea 2 (cheap, sign unknown), each on seed 0 against 028 at equal wall-clock time.
