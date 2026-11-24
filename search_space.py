"""The search's catalog: what a genome may vary, the screening switches, and what each choice costs at compile.

A new strategy module registers here: its variants and numbers in `SPACE`, a switch for screening in `SWITCHES`, its
measured or estimated compile seconds in `COMPILE_SECONDS`, and in `NUMBER_VARIANTS` any number that matters only under
some of its variants. `search.py` reads every name from this module; tests and scripts read them as `search.<name>`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


GenomeValue = str | int | float  # a variant, or one numeric field's value


@dataclass(frozen=True)
class Dimension:
    """One searched coordinate of a genome: a family's variant or one numeric field."""

    key: str  # "family.field"
    choices: tuple[GenomeValue, ...] = ()  # a variant's or a categorical number's allowed values; empty for a number
    low: float = 0.0
    high: float = 1.0
    log: bool = False
    integer: bool = False

    @property
    def family(self) -> str:
        return self.key.split(".")[0]

    @property
    def name(self) -> str:
        return self.key.split(".")[1]

    @property
    def categorical(self) -> bool:
        return bool(self.choices)


SPACE: tuple[Dimension, ...] = (
    Dimension("attention.variant", ("standard", "differential", "triangle", "triangle_differential", "paired")),
    Dimension("attention.heads", (12, 6, 8, 4)),
    Dimension("attention.qk_share", (1.0, 0.75, 0.5)),
    Dimension("attention.tie_kv", (0, 1)),
    Dimension("attention.exclusive", (0, 1)),
    Dimension("attention.block_size", (128, 256)),  # flex's kernels need a multiple of 128: 64 fails at compile (auto17_kumo_01, auto18_bayes_rungs_01)
    Dimension("positional.variant", ("none", "alibi")),
    Dimension("positional.slope_scale", low=0.5, high=2.0, log=True),
    Dimension("positional.rope_base", low=1e4, high=1e5, log=True),
    Dimension("positional.rope_fraction", low=0.1, high=1.0, log=True),
    Dimension("positional.nope_every", low=0, high=6, integer=True),
    # Numbers whose 0 is off, as choices with 0 first, so a draw can leave them off.
    Dimension("positional.periodic_scale", (0.0, 0.25, 0.5, 1.0)),
    Dimension("positional.length_bins", (0, 8, 16)),
    Dimension("positional.terminus", (0, 1)),
    Dimension("attention.local_every", (0, 2, 3)),
    Dimension("attention.kv_share", (0, 2, 3)),
    Dimension("attention.key_offset", (0.0, 0.125, 0.25, 0.5)),
    Dimension("attention.score_cap", (0.0, 4.0, 5.0, 6.0, 8.0)),
    Dimension("kernel.variant", ("default", "flash", "cudagraphs")),
    Dimension("head.variant", ("linear", "two_layer")),
    Dimension("head.bias", (0, 1)),
    Dimension("logit_scale.variant", ("none", "ssmax")),
    Dimension("logit_scale.initial_scale", low=0.5, high=2.0, log=True),
    Dimension("protein_context.variant", ("none", "composition", "global_mean", "both")),
    Dimension("protein_context.every", low=3, high=12, integer=True),
    Dimension("canon.variant", ("none", "neighbors")),
    Dimension("canon.reach", (1, 2)),
    Dimension("canon.placement", ("ac", "a", "c")),
    Dimension("canon.every", (1, 2, 3)),
    Dimension("residual.variant", ("prenorm", "ngpt", "perinorm", "ln_scaling", "momentum")),
    Dimension("residual.momentum_beta", low=0.1, high=0.9),
    Dimension("residual.momentum_lookahead", low=0.1, high=0.9),
    Dimension("residual.velocity_norm", (1, 0)),
    Dimension("residual.alpha_init", low=0.02, high=0.2, log=True),
    Dimension("residual.branch_multiplier", low=0.25, high=1.0, log=True),
    Dimension("residual.branch_ramp", low=0.0, high=0.3),
    Dimension("residual.branch_start", low=0.0, high=0.5),
    Dimension("residual.learned_gain", (0.0, 0.1, 0.3, 1.0)),  # 0 is off; the gates' start otherwise
    Dimension("activation.variant", ("relu2", "xielu")),
    Dimension("latent.variant", ("none", "next_layer", "neighbors")),
    Dimension("latent.source_block", low=6, high=30, integer=True),
    Dimension("latent.weight", low=0.01, high=0.5, log=True),
    Dimension("grouping.variant", ("none", "physchem")),
    Dimension("grouping.weight", low=0.02, high=0.5, log=True),
    Dimension("grouping.until", low=0.1, high=0.8),
    Dimension("memory.variant", ("none", "product_keys", "shared_pool")),
    Dimension("memory.block", low=6, high=30, integer=True),
    Dimension("deep_supervision.variant", ("none", "mid_heads")),
    Dimension("deep_supervision.weight", low=0.02, high=0.3, log=True),
    Dimension("deep_supervision.until", low=0.3, high=0.8),
    Dimension("stream.variant", ("single", "hyper", "dense", "attention_residual")),
    Dimension("stream.streams", low=2, high=4, integer=True),
    Dimension("unet.variant", ("none", "mirrored")),
    Dimension("embedding_mix.variant", ("none", "every_block")),
    Dimension("embedding_mix.initial_share", low=0.0, high=0.3),
    Dimension("neighbor.variant", ("none", "pair_table", "bigram_table", "trigram_table")),
    Dimension("loss.variant", ("plain", "entropy_weighted")),
    Dimension("loss.z_weight", (0.0, 1e-4, 1e-3, 1e-2)),
    Dimension("loss.substitution_share", (0.0, 0.05, 0.1, 0.2)),
    Dimension("loss.substitution_until", low=0.2, high=0.8),
    Dimension("value_embeddings.variant", ("none", "ends")),
    Dimension("value_embeddings.blocks_each_end", low=1, high=6, integer=True),
    Dimension("value_residual.variant", ("none", "first_block", "normalized")),
    Dimension("value_residual.initial_share", low=0.2, high=0.8),
    Dimension("value_residual.anchor_norm", (0, 1)),
    Dimension("value_residual.per_head", (0, 1)),
    Dimension("value_residual.qk_anchor", (1.0, 0.9, 0.7, 0.5)),
    Dimension("gates.variant", ("none", "residual_channels", "head_sigmoid")),
    Dimension("softcap.variant", ("none", "tanh")),
    Dimension("softcap.cap", low=10.0, high=60.0, log=True),
    Dimension("optimizer.variant", ("muon", "normuon", "muon_plus", "anvil", "amuse", "adamuon", "muown")),
    Dimension("optimizer.orthogonalizer", ("polar_express", "mud1", "mud2")),
    Dimension("optimizer.polar_steps", (5, 4, 3)),
    Dimension("optimizer.contra_start", (0.0, 0.25, 0.5, 1.0)),  # 0 is off, so a draw can leave it off; above 0 is contra
    Dimension("optimizer.contra_end", (0.0, -0.25, -0.5)),  # below 0 is soft, the post's end of the schedule
    Dimension("optimizer.radial_out", (1.0, 0.75, 0.5, 0.25, 0.0)),
    Dimension("optimizer.radial_in", (1.0, 0.5, 0.0)),
    Dimension("optimizer.variance_scale", (0, 1)),
    Dimension("optimizer.mona_beta", (0.0, 0.975, 0.99)),
    Dimension("optimizer.nesterov", (1, 0)),
    Dimension("optimizer.normuon_beta", low=0.8, high=0.995),
    Dimension("optimizer.qk_lr_scale", low=0.5, high=2.0, log=True),
    Dimension("optimizer.vo_lr_scale", low=0.5, high=2.0, log=True),
    Dimension("optimizer.mlp_lr_scale", low=0.5, high=2.0, log=True),
    Dimension("optimizer.muown_ratio", low=0.05, high=0.5, log=True),
    Dimension("optimizer.plus_axis", ("col_row", "row_col", "row", "col")),
    Dimension("optimizer.anvil_slow_beta", low=0.95, high=0.995),
    Dimension("optimizer.anvil_fast_weight", low=0.2, high=0.7),
    Dimension("optimizer.amuse_beta", low=0.3, high=0.7),
    Dimension("optimizer.amuse_rho", low=0.4, high=1.0),
    Dimension("optimizer.adam_beta1", low=0.7, high=0.95),
    Dimension("optimizer.adam_beta2", low=0.95, high=0.999),
    # Floors halved on 2026-10-31 for the 1-hour track: the LR-against-horizon laws put six times the steps at 0.41 to
    # 0.56 times the 10-minute rates (`2026-10-31_budget_scaling_to_one_hour.md` 1), Muon near 0.016 to 0.022 and the
    # AdamW groups at or just above the old floors. Ceilings raised on 2026-11-03 to the configs' bounds: the top fifth of
    # adaptive runs piled up against them (`embed_lr` at 1e-2, `muon_lr` 63 percent in the top tenth of its range, `head_lr` at
    # 3e-3 on the record), so the search never saw the far side.
    Dimension("optimizer.muon_lr", low=0.01, high=0.12, log=True),
    Dimension("optimizer.embed_lr", low=5e-4, high=3e-2, log=True),
    Dimension("optimizer.head_lr", low=1.5e-4, high=1e-2, log=True),
    Dimension("optimizer.scalar_lr", low=5e-4, high=1e-2, log=True),
    Dimension("optimizer.cautious_weight_decay", low=0.0, high=0.1),
    Dimension("optimizer.adam_beta2_end", low=0.95, high=0.9995),
    Dimension("optimizer.decay_timescale", (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)),  # in total runs, log-spaced
    Dimension("optimizer.fused_polar", (0, 1)),
    Dimension("optimizer.depth_taper", low=-0.5, high=0.5),  # 0, the record's flat rate, is inside
    Dimension("schedule.variant", ("trapezoid", "one_sqrt")),
    Dimension("schedule.warmup_fraction", low=0.0, high=0.05),
    Dimension("schedule.cooldown_fraction", low=0.15, high=0.65),  # floor 0.25 to 0.15 for the 1-hour track (2026-10-31)
    Dimension("schedule.momentum_start", low=0.85, high=0.95),
    Dimension("schedule.momentum", low=0.95, high=0.99),  # the momentum at the end of its warmup ramp
    Dimension("schedule.momentum_cooldown_end", low=0.95, high=0.995),
    Dimension("schedule.muon_late_scale", (1.0, 1.5, 2.0, 3.0)),  # 1 is off
    Dimension("schedule.muon_late_start", low=0.6, high=0.95),
    Dimension("schedule.cooldown_power", low=0.5, high=2.0, log=True),
    Dimension("lookahead.variant", ("none", "periodic")),
    Dimension("lookahead.every", low=3, high=20, log=True, integer=True),
    Dimension("lookahead.alpha", low=0.3, high=0.8),
    Dimension("lookahead.start", low=0.0, high=0.6),
    Dimension("batch.variant", ("packed", "whole")),
    Dimension("batch.window", (1024, 2048)),  # the count follows (`batch.windows_per_step` 0): 65,536 tokens a step
    Dimension("batch.micro_divisor", (1, 2, 4)),
    Dimension("batch.accumulate_early", (1, 2, 4)),
    Dimension("batch.accumulate_late", (1, 2, 4)),
    Dimension("batch.accumulate_switch", low=0.05, high=0.9),
    Dimension("regularization.variant", ("none", "early_dropout")),
    Dimension("regularization.rate", low=0.05, high=0.2, log=True),
    Dimension("regularization.until", low=0.05, high=0.3, log=True),
    Dimension("curriculum.variant", ("none", "short_first", "family_tempered", "length_tempered")),
    Dimension("curriculum.fraction", low=0.2, high=0.6),
    Dimension("curriculum.alpha", low=0.0, high=1.0),
    Dimension("curriculum.beta", low=-0.5, high=0.5),
    Dimension("ema.variant", ("none", "tail", "latest_k", "drift_leap", "two_decay", "plateau_restart")),
    Dimension("ema.start", low=0.4, high=0.9),
    Dimension("ema.decay", low=0.98, high=0.998),
    Dimension("ema.window", low=100, high=1000, log=True, integer=True),  # 10-minute steps, scaled with the run (ema.py)
    Dimension("ema.leap", low=0.05, high=0.5, log=True),
    Dimension("ema.decay_fast", low=0.9, high=0.99),
    Dimension("sam.variant", ("none", "late")),
    Dimension("sam.rho", low=0.01, high=0.2, log=True),
    Dimension("sam.every", low=2, high=8, integer=True),
    Dimension("sam.start", low=0.4, high=0.9),
    Dimension("backprop_free.variant", ("none", "residual_es", "unit_select", "unit_random", "readout_warmstart",
                                        "readout_boost", "unit_route")),
    Dimension("backprop_free.readout_weight", low=0.05, high=1.0, log=True),
    Dimension("backprop_free.switch", low=0.1, high=1.0),
    Dimension("backprop_free.route_k", low=64, high=1024, log=True, integer=True),
    # The shape, free since 2026-11-23 (`docs/decisions/representation/speedrunning_plms/2026-11-23_compute_matched_free_architecture.md`):
    # widths as choices, so each is a multiple of 64 the attention's head counts can divide; the record's first.
    Dimension("shape.blocks", low=12, high=48, integer=True),
    Dimension("shape.hidden", (768, 512, 640, 896, 1024, 1152, 1280, 1536)),
    Dimension("shape.intermediate", (2048, 1024, 1536, 2560, 3072, 4096, 5120, 6144)),
    # The looped core, per-pass routing and the SGLR pool (`2026-11-23_looped_core_families`); a layout past the depth is
    # a genome conflict, which the proposers skip.
    Dimension("loop.variant", ("none", "tied", "untied")),
    Dimension("loop.prelude", low=1, high=8, integer=True),
    Dimension("loop.core", low=2, high=12, integer=True),
    Dimension("loop.passes", low=1, high=6, integer=True),
    Dimension("loop.inject", ("loom", "learned", "none", "read")),
    Dimension("loop.read_decay", (0.0, 0.55)),  # InfiLoop's start; used only by the read
    Dimension("loop.write_gate", (0, 1)),
    Dimension("loop.residual_lambda", (0.0, 0.5)),
    Dimension("loop.sample", (0, 1)),
    Dimension("loop.readout_weight", (0.0, 0.1, 0.3)),
    Dimension("routing.variant", ("none", "per_pass", "shared")),
    Dimension("routing.experts", (8, 4, 16)),
    Dimension("routing.top_k", (2, 1)),
    Dimension("sglr.variant", ("none", "pool")),
    Dimension("sglr.prelude", low=2, high=8, integer=True),
    Dimension("sglr.steps", low=8, high=30, integer=True),
    Dimension("sglr.router", ("state", "program")),
    Dimension("sglr.balance_floor", (0.0, 0.5)),  # SGLR's floor, half the even split's entropy; used only by the state router
)

# The binary switches a screening design may set high; low is the record's choice. Two attention switches combine:
# both high is the composed triangle_differential kernel.
SWITCHES: dict[str, dict[str, dict[str, Any]]] = {
    "unet": {"unet": {"variant": "mirrored"}},
    "embedding_mix": {"embedding_mix": {"variant": "every_block"}},
    "value_embeddings": {"value_embeddings": {"variant": "ends"}},
    "value_residual": {"value_residual": {"variant": "first_block"}},
    "gates": {"gates": {"variant": "residual_channels"}},
    "neighbor": {"neighbor": {"variant": "pair_table"}},
    "ema": {"ema": {"variant": "tail"}},
    "curriculum": {"curriculum": {"variant": "short_first"}},
    "ngpt": {"residual": {"variant": "ngpt"}},
    "differential": {"attention": {"variant": "differential"}},
    "triangle": {"attention": {"variant": "triangle"}},
    "hyper": {"stream": {"variant": "hyper"}},
    "dense": {"stream": {"variant": "dense"}},
    "residual_es": {"backprop_free": {"variant": "residual_es"}},
    "readout_warmstart": {"backprop_free": {"variant": "readout_warmstart"}},
    "unit_select": {"backprop_free": {"variant": "unit_select"}},
    # Switches that change numbers or shapes, not the graph's size, so they add little compile time: queue 3's two
    # near-record changes, the softcap off, plain Muon, and 6 heads of 128.
    "heads_6": {"attention": {"heads": 6}},
    "rope_1e5": {"positional": {"rope_base": 100_000.0}},
    "slopes_x2": {"positional": {"slope_scale": 2.0}},
    "no_softcap": {"softcap": {"variant": "none"}},
    "muon": {"optimizer": {"variant": "muon"}},
    # The 2026-10-30 modules: 2,048-token windows at the record's tokens a step, whole-protein packing, the tempered
    # reading orders at their defaults, SSMax, Peri-LN, LayerNorm Scaling, a halved branch, xIELU, the latent loss,
    # Muon+, the head-wise gate, a quarter of RoPE's pairs, and RoPE off in every fourth block.
    "window_2048": {"batch": {"window": 2048}},
    "whole": {"batch": {"variant": "whole"}},
    "family_tempered": {"curriculum": {"variant": "family_tempered"}},
    "length_tempered": {"curriculum": {"variant": "length_tempered", "beta": 0.5}},
    "ssmax": {"logit_scale": {"variant": "ssmax"}},
    "perinorm": {"residual": {"variant": "perinorm"}},
    "ln_scaling": {"residual": {"variant": "ln_scaling"}},
    "branch_half": {"residual": {"branch_multiplier": 0.5}},
    "xielu": {"activation": {"variant": "xielu"}},
    "latent": {"latent": {"variant": "next_layer"}},
    "muon_plus": {"optimizer": {"variant": "muon_plus"}},
    "head_gate": {"gates": {"variant": "head_sigmoid"}},
    "rope_quarter": {"positional": {"rope_fraction": 0.25}},
    "nope_4": {"positional": {"nope_every": 4}},
    # Round 2: the Dust, noise and Boost hybrids inside backprop, the noise track's Adam beta1, ANVIL and AMUSE, LAWA,
    # the 1-sqrt cooldown and the neighbor heads.
    "readout_boost": {"backprop_free": {"variant": "readout_boost"}},
    "unit_route": {"backprop_free": {"variant": "unit_route"}},
    "branch_ramp": {"residual": {"branch_ramp": 0.1}},
    "anvil": {"optimizer": {"variant": "anvil"}},
    "amuse": {"optimizer": {"variant": "amuse"}},
    "adam_beta1_0.7": {"optimizer": {"adam_beta1": 0.7}},
    "latest_k": {"ema": {"variant": "latest_k"}},
    "one_sqrt": {"schedule": {"variant": "one_sqrt"}},
    "neighbors": {"latent": {"variant": "neighbors"}},
    # Round 3: ESM-2's head, the bidirectional key offset, the helix and strand bias, a binding score cap, the drift
    # leap, the two-decay average, the FlashAttention-4 backend.
    "head_two_layer": {"head": {"variant": "two_layer"}},
    "key_offset": {"attention": {"key_offset": 0.25}},
    "periodic": {"positional": {"periodic_scale": 0.5}},
    "score_cap": {"attention": {"score_cap": 5.0}},
    "drift_leap": {"ema": {"variant": "drift_leap"}},
    "two_decay": {"ema": {"variant": "two_decay"}},
    "flash": {"kernel": {"variant": "flash"}},
    # Round 4: the cooldown's Muon momentum and AdamW beta2, the constant-timescale decay, the selective losses, the
    # context tables, the compiled Polar Express.
    "momentum_0.98": {"schedule": {"momentum_cooldown_end": 0.98}},
    "beta2_0.999": {"optimizer": {"adam_beta2_end": 0.999}},
    "decay_timescale": {"optimizer": {"decay_timescale": 1.0}},
    "entropy_weighted": {"loss": {"variant": "entropy_weighted"}},
    "z_loss": {"loss": {"z_weight": 1e-3}},
    "bigram_table": {"neighbor": {"variant": "bigram_table"}},
    "trigram_table": {"neighbor": {"variant": "trigram_table"}},
    "fused_polar": {"optimizer": {"fused_polar": 1}},
    # Round 5: Attention Residuals, local blocks, CUDA graphs, the plateau restart, early dropout, the length table.
    "attention_residual": {"stream": {"variant": "attention_residual"}},
    "local_3": {"attention": {"local_every": 3}},
    "cudagraphs": {"kernel": {"variant": "cudagraphs"}},
    "plateau_restart": {"ema": {"variant": "plateau_restart"}},
    "early_dropout": {"regularization": {"variant": "early_dropout"}},
    "length_bins": {"positional": {"length_bins": 16}},
    # Round 7, aimed at the 1-hour track: a product-key memory, cross-layer K and V, the residue-group loss, the
    # terminus vectors, Muon's late widening.
    "product_keys": {"memory": {"variant": "product_keys"}},
    "kv_share_2": {"attention": {"kv_share": 2}},
    "grouping": {"grouping": {"variant": "physchem"}},
    "terminus": {"positional": {"terminus": 1}},
    "muon_late_2": {"schedule": {"muon_late_scale": 2.0}},
    # Round 8, for the 1-hour track's late run: SAM in the last 40 percent, AdaMuon, a depth-graded Muon rate, learned
    # branch gains.
    "sam_late": {"sam": {"variant": "late"}},
    "adamuon": {"optimizer": {"variant": "adamuon"}},
    "depth_taper": {"optimizer": {"depth_taper": 0.5}},
    "branch_gains": {"residual": {"learned_gain": 0.3}},
    # Round 8 from the 1-hour capacity scout: three reads of one memory pool, and intermediate heads early in the run.
    "shared_pool": {"memory": {"variant": "shared_pool"}},
    "mid_heads": {"deep_supervision": {"variant": "mid_heads"}},
    # Round 9 (`ideas/2026-11-03_round9_strategies.md`). Throughput: fewer or narrower heads, a narrower QK, tied K and V, MUD,
    # three Polar Express steps. Update rule: Muown, the radial brake and Hyperball's fixed norms, Contra-to-Soft, Muon-VS, MONA,
    # Lookahead, the per-role rates, the cooldown's power. Architecture: the value residual's norm, head grain and QK mix, gated
    # exclusive attention, Canon, the momentum stream, the protein's composition and mean. Targets: BLOSUM62's soft targets, the head's bias.
    "heads_8": {"attention": {"heads": 8}},
    "heads_4": {"attention": {"heads": 4}},
    "qk_75": {"attention": {"qk_share": 0.75}},
    "tie_kv": {"attention": {"tie_kv": 1}},
    "exclusive": {"attention": {"exclusive": 1}},
    "mud1": {"optimizer": {"orthogonalizer": "mud1"}},
    "mud2": {"optimizer": {"orthogonalizer": "mud2"}},
    "polar_3": {"optimizer": {"polar_steps": 3}},
    "muown": {"optimizer": {"variant": "muown"}},
    "radial_brake": {"optimizer": {"radial_out": 0.5}},
    "hyperball": {"optimizer": {"radial_out": 0.0, "radial_in": 0.0}},
    "contra_to_soft": {"optimizer": {"contra_start": 0.5, "contra_end": -0.25}},
    "variance_scale": {"optimizer": {"variance_scale": 1}},
    "mona": {"optimizer": {"mona_beta": 0.99}},
    "lookahead": {"lookahead": {"variant": "periodic"}},
    "mlp_lr_half": {"optimizer": {"mlp_lr_scale": 0.5}},
    "cooldown_power": {"schedule": {"cooldown_power": 1.2}},
    "anchor_norm": {"value_residual": {"anchor_norm": 1}},
    "value_per_head": {"value_residual": {"per_head": 1}},
    "qk_anchor": {"value_residual": {"qk_anchor": 0.7}},
    "value_normalized": {"value_residual": {"variant": "normalized"}},
    "canon": {"canon": {"variant": "neighbors"}},
    "momentum_stream": {"residual": {"variant": "momentum"}},
    "composition": {"protein_context": {"variant": "composition"}},
    "protein_mean": {"protein_context": {"variant": "global_mean"}},
    "substitution_targets": {"loss": {"substitution_share": 0.1}},
    "head_bias": {"head": {"bias": 1}},
    "batch_ramp": {"batch": {"micro_divisor": 2, "accumulate_early": 1, "accumulate_late": 2, "accumulate_switch": 0.25}},  # 32 windows, 64 from a quarter of the clock
    "batch_double_late": {"batch": {"accumulate_early": 1, "accumulate_late": 2, "accumulate_switch": 0.7}},  # 64 windows, 128 from 70 percent
}
BACKPROP_FREE_SWITCHES = ("residual_es", "readout_warmstart", "unit_select", "readout_boost", "unit_route")

# The compile warmup is capped at 180 s and a new graph compiles from scratch: the record's in 106 s on the GH200
# (pilot1_01). Each variant's seconds above it were measured alone on the record (pilot1 and pilot3 at the 180 s rung,
# and 005, 010, 011, 026, 028 by hand); a variant not listed, and every number, added none measurable. The sum predicts a
# genome's warmup; the margin covers compile costs that do not add exactly. The proposers skip a genome predicted over.
WARMUP_CAP = 180.0
RECORD_WARMUP = 106.0
WARMUP_MARGIN = 15.0
# The record's peak training memory (66.4 GiB, lp_record_r180 on enzyme_gh200) and what a genome may be predicted to use
# of the GH200's 96 GB; `search.predicted_peak_gib` scales the first by a genome's shape and batch.
RECORD_PEAK_GIB = 66.4
PEAK_CAP_GIB = 88.0
# The free-shape pilots of 2026-11-23_free_architecture_1h peaked above that scaling by 1.106, 1.070 and 1.109 (79.4,
# 85.8 and 87.6 GiB against 71.8, 80.2 and 79.0), so a prediction times the worst of them must fit the cap. Their
# warmups ran under the prediction (101.4, 181.5 and 134.9 s against 108, 216 and 162), the safe side, so it stays.
PEAK_PREDICTION_ERROR = 1.11
COMPILE_SECONDS: dict[str, float] = {
    "attention.variant=differential": 24.0,
    "attention.variant=triangle": 46.0,
    "attention.variant=triangle_differential": 76.0,
    "attention.variant=paired": 17.0,
    "residual.variant=ngpt": 36.0,
    "stream.variant=hyper": 37.0,
    "stream.variant=dense": 108.0,
    "backprop_free.variant=residual_es": 66.0,
    "backprop_free.variant=unit_select": 5.0,
    "backprop_free.variant=unit_random": 5.0,
    "backprop_free.variant=readout_warmstart": 7.0,
    "value_residual.variant=first_block": 11.0,
    "gates.variant=residual_channels": 12.0,
    "unet.variant=mirrored": 1.0,
    "neighbor.variant=pair_table": 1.0,
    "embedding_mix.variant=every_block": 5.0,
    # Not measured yet, conservative until a 180 s pilot replaces them: 10 s for an elementwise change in every block,
    # 25 s for a new module in every block, 10 s for a first compile at a new window shape.
    "batch.window=2048": 3.0,  # pilot4_01: warmup 125 s against the record genome's 122
    "logit_scale.variant=ssmax": 10.0,  # pilot4_02: 132 s
    "residual.variant=perinorm": 16.0,  # pilot4_03: 138 s
    "residual.variant=ln_scaling": 10.0,
    "activation.variant=xielu": 19.0,  # pilot4_04: 140.6 s
    "latent.variant=next_layer": 1.0,  # pilot4_06: 123.3 s
    "gates.variant=head_sigmoid": 25.0,
    "residual.branch_multiplier!=1.0": 10.0,  # `!=`: any value but the record's adds the hook
    "residual.branch_ramp!=0.0": 10.0,
    # pilot5_03: all 72 readouts compiled in 419.8 s (300 s over the record); now 10 readouts, at every sixth block, the
    # warm start's 5 cost 7 s and the 72 about 4 s each, so 40 s until a pilot measures the 10.
    "backprop_free.variant=readout_boost": 40.0,
    "backprop_free.variant=unit_route": 25.0,  # a top-k and a scatter in every block
    "latent.variant=neighbors": 10.0,
    "head.variant=two_layer": 10.0,
    "attention.key_offset!=0.0": 10.0,  # two shifted copies and a select on the keys in every block
    "attention.score_cap!=0.0": 10.0,  # a tanh inside the flex score function
    "positional.periodic_scale!=0.0": 10.0,  # two cosines inside the flex score function
    "kernel.variant=flash": 30.0,  # the CuTe kernels' JIT; no published compile time, the pilot measures it
    "neighbor.variant=bigram_table": 10.0,  # a gather, a sigmoid and an add after the last block
    "neighbor.variant=trigram_table": 10.0,
    "optimizer.fused_polar=1": 20.0,  # Polar Express compiled at the three bank shapes, outside the model's graph
    # Round 5. Attention Residuals: pilot8_01 alone on confirm1_01 warmed up in 223 s against 114 to 137 s for the other
    # pilot8 runs with fresh graphs, about 100 s of its own (the 666 weighted terms over the 37 values); 110 s keeps it
    # out of any genome that carries another compile cost.
    "stream.variant=attention_residual": 110.0,
    # The local blocks' second flex kernel pair (a mask rule traced into the kernel) and the local mask's build; the
    # catalog's second-mask cost is 20 to 50 s, the triangle's two masks with its other changes 46 s.
    "attention.local_every!=0": 30.0,
    "kernel.variant=cudagraphs": 10.0,  # graph-tree recording of the forward, backward and no-grad forward a shape
    "regularization.variant=early_dropout": 10.0,  # a draw, a compare and a scale on each sublayer output
    "positional.length_bins!=0": 5.0,  # a scatter, a gather and a table add once, at the embedding (pilot8: 119 s warmup)
    # Round 7, estimates: the memory's query matmul, two top-k, the candidate sums and an embedding bag at one block,
    # with its backward scatter (the idea file's 10 to 20 s, at the top); the group head is one more auxiliary graph
    # (next_layer's measured 1 s, so 10 is conservative); the terminus vectors are two masked adds at the embedding, as
    # cheap as the length table. `attention.kv_share` drops matmuls from an unchanged graph and `schedule.muon_late_*`
    # is a float on Muon's groups: no entry.
    "memory.variant=product_keys": 20.0,
    "grouping.variant=physchem": 10.0,
    "positional.terminus!=0": 5.0,
    # Round 8: the learned gains are one more multiply on each sublayer output, `branch_multiplier`'s 10 s. SAM reruns the
    # step's own graph, and AdaMuon and the depth taper live in the eager optimizer: no entry.
    "residual.learned_gain!=0.0": 10.0,
    # Three memory reads: the single read's 20 s plus the scout's 10 s for the extension (the reads share one pool and
    # differ in their queries and keys). The intermediate heads reuse the head's graph twice in the auxiliary loss: the
    # scout's 0 to 5 s, held at the group head's 10 s.
    "memory.variant=shared_pool": 30.0,
    "deep_supervision.variant=mid_heads": 10.0,
    # Round 9. Measured at 180 s on `speedrun2_gh200` against the record's 135.6 s there (`2026-11-03_surrogate_head_to_head`): the
    # gate 159.4 s; the momentum stream 186.8 s, with `ln_scaling`'s 10 s gone from it; the anchor 151.1 s. The composition (127.8 s) and
    # the global mean (134.8 s) add nothing measurable: no entry. The rest are estimates by the kinds above, times 2.2, what the
    # measured ones ran over their first estimates. The value residual's forms count even with it off, a few seconds of margin. No
    # entry, since they are eager or remove work: every `optimizer.*`, `lookahead.*`, `schedule.cooldown_power`,
    # `loss.substitution_*`, `attention.tie_kv`, `attention.heads`.
    "attention.qk_share!=1.0": 22.0,
    "attention.exclusive!=0": 24.0,
    "attention.block_size!=128": 11.0,
    "residual.variant=momentum": 61.0,
    "canon.variant=neighbors": 55.0,
    "canon.reach=2": 33.0,
    "value_residual.variant=normalized": 33.0,
    "value_residual.anchor_norm!=0": 5.0,
    "value_residual.per_head!=0": 11.0,
    "value_residual.qk_anchor!=1.0": 16.0,
    "head.bias!=0": 2.0,
    # 2026-11-23_looped_speedrun's cold compiles on enzyme_gh200 against its record's 125 s: a loop added nothing
    # (108 to 123 s), routing 15 to 21 s, the SGLR pool 45 s.
    "routing.variant!=none": 20.0,
    "sglr.variant=pool": 45.0,
    # No entry, since they run eagerly outside any compiled graph: the loss family (`training_loss`), the cooldown's
    # momentum and beta2 (floats on the param groups), and the constant-timescale decay (a foreach multiply).
}
# A number that matters only under some of its family's variants; the rest follow `dimension_active`'s rule.
NUMBER_VARIANTS: dict[str, tuple[str, ...]] = {
    "positional.slope_scale": ("alibi",),
    "residual.alpha_init": ("ngpt",),
    "residual.branch_multiplier": ("prenorm", "ln_scaling", "momentum"),  # ngpt and perinorm normalize the branch after it
    "residual.momentum_beta": ("momentum",),
    "residual.momentum_lookahead": ("momentum",),
    "optimizer.muown_ratio": ("muown",),
    "optimizer.normuon_beta": ("normuon", "adamuon"),
    "protein_context.every": ("global_mean", "both"),
    "curriculum.fraction": ("short_first",),
    "curriculum.alpha": ("family_tempered",),
    "curriculum.beta": ("length_tempered",),
    "residual.branch_ramp": ("prenorm", "ln_scaling", "momentum"),
    "residual.branch_start": ("prenorm", "ln_scaling", "momentum"),
    "latent.source_block": ("next_layer",),
    "memory.block": ("product_keys",),  # shared_pool reads at 9, 18 and 27
    "stream.streams": ("hyper",),
    "optimizer.cautious_weight_decay": ("muon", "normuon", "muon_plus", "anvil", "adamuon"),  # amuse takes no decay
    "optimizer.anvil_slow_beta": ("anvil",),
    "optimizer.anvil_fast_weight": ("anvil",),
    "optimizer.amuse_beta": ("amuse",),
    "optimizer.amuse_rho": ("amuse",),
    "ema.start": ("tail", "two_decay", "plateau_restart"),
    "ema.decay": ("tail", "two_decay"),
    "ema.window": ("latest_k", "drift_leap", "plateau_restart"),
    "ema.leap": ("drift_leap",),
    "ema.decay_fast": ("two_decay",),
    "backprop_free.readout_weight": ("readout_boost",),
    "backprop_free.switch": ("readout_boost",),
    "backprop_free.route_k": ("unit_route",),
}
