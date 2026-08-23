"""Device model for ``device_class: fws_cim`` — the single source of truth.

Every fixed-weight-stationary compute-in-memory (FWS-CIM) timing law lives in
:class:`CimDeviceModel`. Consumers (the per-op GEMM branch in
``base_timing.get_gemm_time`` and the FWS spatial report in
``inference_timing.calc_time``) call these methods; no law is duplicated at a
consumer.

Laws (validated numerically against the OPTIMA reference model)
---------------------------------------------------------------

Analog weight-GEMM law
    ``vec_cycles = adc_mux * slice_cycles``;
    ``vec_latency = vec_cycles / f_analog``.
    A weight GEMM's time is **independent of K and N** — every array a weight
    matrix maps to fires in parallel — so ``T = M_tokens * vec_latency`` with
    ``M_tokens`` the token count of the call as received (tp-sharded K/N never
    changes T). ``shots_per_output`` affects energy only, never time.

Array counting
    A K x N weight matrix occupies
    ``arrays(K, N) = ceil(K / rows) * ceil(N / (cols_adc * adc_mux))``
    analog arrays. This reproduces OPTIMA's per-layer counts (QKV -> 3,
    O-proj -> 1, FFN1/FFN2 -> ceil(I/H) each for the validation geometries;
    the FFN1 pair doubles under SwiGLU/gated MLP).

Analog energy / area
    Per stage per layer: ``E = M_tokens * energy_per_vec_pj * shots_per_output
    * arrays``. Area = arrays * ``area_mm2_per_array``. NOTE (OPTIMA
    equivalence): the recorded OPTIMA "CTT area" counts transformer-stack
    arrays only — endpoint (patch-embed / classifier-head) arrays are extra —
    hence the split into :meth:`CimDeviceModel.stack_area_mm2` and
    :meth:`CimDeviceModel.total_area_mm2`.

Systolic-array (SA) attention law (``cim.fabric.model: sa``)
    ``cycles(M, N, K; R, C) = ceil(M/R) * ceil(N/C) * (K + R + C - 2) - 1``
    — bit-exact against OPTIMA's recorded ScaleSim outputs (os dataflow, CALC
    bandwidth, generous SRAM). Attention uses OPTIMA's **folded-K**
    convention: heads are folded into K to model dual-buffered
    fully-pipelined arrays (per-head fill/drain is NOT priced):

    - ``heads_chip = ceil(kv_heads / tp)`` when tp-sharded (tp >= 2), else
      ``kv_heads``; ``heads_per_replica = ceil(heads_chip / replicas)``.
    - The law is evaluated at the CALL DIMS of the attention GEMM as it
      arrives (score orientation ``(m, k, n)`` = (query rows, head_dim,
      context)); the concurrent dual run is derived from the same call:
      ``run_a = sa_cycles(m, n, k * heads_per_replica * streams)`` and
      ``run_b = sa_cycles(m, k, n * heads_per_replica * streams)``.
      ``streams`` is the number of independent batch streams (B) whose
      attention the stage prices: streams fold into the contraction dim
      exactly as heads do (back-to-back dual-buffered runs; one fill/drain
      penalty for the whole batch). ``streams = 1`` reproduces the pass-1
      law bit-exactly. A wavefront that carries B streams (the LLM
      prefill/decode report convention) MUST price streams = B — the
      fabric is one fixed R x C x num_arrays sidecar, so B streams cost
      ~B x one stream's cycles; anything less would exceed the arithmetic
      peak of the declared silicon.
      For MHA prefill (m=S, k=head_dim, n=S, streams=1) this reduces
      EXACTLY to the pass-1 shapes: QK^T M=S, N=S, K=head_dim*h_rep and
      PV M=S, N=head_dim, K=S*h_rep. The max() is symmetric under k<->n,
      so score and output calls of the same stage return the same folded
      total.
    - QK and PV occupy the fabric's arrays **concurrently** (assumes
      ``fabric.num_arrays >= 2``):
      ``total_cycles = max(QK, PV) + fill_drain_penalty`` (config, default
      3*R — a hand-picked surrogate inherited from OPTIMA);
      ``T = total_cycles / f_fabric``.

GQA / decode folding semantics (our stated generalization; OPTIMA refuses GQA)
    The folded-K call-dims law prices GQA prefill (score call
    m = S * shared_heads with shared_heads = num_heads // kv_heads, k =
    head_dim, n = S) and decode (score: m = shared_heads, k = head_dim,
    n = context; output: m = shared_heads, k = context, n = head_dim —
    the llm_util decode descriptors) with no special cases: kv heads are
    folded into K via heads_per_replica exactly as in MHA prefill, and the
    query rows of the call carry the shared-head multiplicity.

Softmax-lanes law
    ``cycles = pipeline_depth + ceil(tokens_q * heads_chip / lanes) - 1``
    with ``lanes = softmax_lanes * replicas`` and ``tokens_q`` the
    B-scaled query rows of the stage (prefill: streams * S — at the
    recorded B=1 points this is the pass-1 form S * heads_chip unchanged;
    decode: tokens_q = B * shared_heads at the step's context). The
    report's stage-2 time is ``max(T_sa, T_softmax)``.

Decode laws + KV stories (pass 2)
    Decode weight ops (``decode_qkv_proj_f`` etc.) arrive per-stream with
    m=1 and the B streams outside the call (the caller never multiplies
    non-attention time by the descriptor batch), so ``price_gemm`` prices
    them at ``M = B`` (analog law over the B decode streams). MoE bucket
    ops (``*_hot`` / ``*_cold`` / ``*_shared``) already arrive with
    token-true M and are priced as received. Decode attention takes the
    call-dims law above at the step's context with ``streams = B`` (the
    B streams' independent attention problems fold into the contraction).
    The decode stage-2 law is
    ``S2_decode = max(T_sa, T_softmax, kv_read_bytes / kv_bw)`` with the
    SHARDED GQA kv-bytes form (mirrors memory_estimation's per-device law:
    ``2 * B * ceil(kv_heads/tp) * head_dim * context * kv_precision``) and
    ``kv_bw`` from the KV story: ``cim_sram`` reads KV from the fabric
    activation SRAM (the DRAM-stub tier bandwidth), ``cim_dram`` from the
    optional ``cim.kv_dram`` block. ``hbm_only`` is invalid — the device
    has no HBM. The aggregate decode figure ``B / period`` is a fabric
    ceiling (full pipeline occupancy); the sustained law
    (:meth:`CimDeviceModel.decode_sustained_throughput`) caps the resident
    wavefronts by the KV stream capacity AND the result by the two
    physical rate ceilings:
    ``min(min(ceil(step_latency/period), floor(kv_max_streams/B)) * B /
    step_latency, B / period, kv_bw / kv_bytes_per_token)``. The
    ``B / period`` cap exists because the bottleneck stage completes at
    most one wavefront of B per period; the bandwidth cap exists because
    every resident wavefront's kv-reading S2 stages draw on the ONE
    declared KV tier concurrently, so the aggregate token rate can never
    imply more than the tier's bandwidth.

MoE laws (experts-per-chip, pass 2)
    Arrays for a MoE layer = attention arrays (QKV + O; K/N law) + routed
    experts ``E * [arrays(H, fold*I_moe) + arrays(I_moe, H)]`` + shared
    experts ``n_shared * same`` + router ``arrays(H, E)``, with
    ``fold = 2`` for gated MLPs (the FUSED descriptor convention: N=2*I
    arrives already fused — never double again; gatedness comes from the
    model's ``uses_gated_mlp`` source, not from model_type=='vit_dinov3').
    FWS routed-FFN stage time (all expert arrays coexist and fire in
    parallel): ``T = tokens_hot * vec_latency`` with ``tokens_hot =
    ceil(tokens_owner * top_k * alpha / E)`` (alpha = the one-hot
    expert_imbalance_factor, tokens_owner = B*S; cp == 1 enforced).
    Shared experts run on their own arrays concurrently over ALL owner
    tokens: ``T_shared = tokens_owner * vec_latency``. MoE FFN stage time
    = max(routed, shared). Dispatch/combine are boundary transfers over
    the ep link with ``bytes = tokens_owner * top_k * hidden * act_bytes``
    each way; ``cim.chip.moe_expert_parallel`` = k spreads each MoE
    layer's routed experts over k chips (divides per-chip expert arrays
    and dispatch/combine time by k, multiplies chip count).
    PER-OP PRICING NOTE: the per-op (sequential) figure deliberately does
    NOT cancel the caller's expert serialization — price_gemm prices the
    per-expert call and the caller's outer multiplier models serialized
    experts. The FWS spatial report computes the parallel-expert law above
    directly and is authoritative.

Endpoints (model-shaped, pass 2)
    ViT models keep patch_embed (analog, M=seq, chip 0) and vit_head
    (analog, M=B, last chip). LLM models get lm_head instead
    (linear_softmax: arrays = ceil(H/rows) * ceil(vocab/(cols_adc*mux)),
    stage M = B*S prefill / B decode, last chip). The LLM embedding lookup
    is NOT a GEMM (a memory roofline priced on the stub) — it is a report
    note only; ``disable_embedding_unembedding`` removes it for
    pure-transformer studies.

Digital-helper absorption (assumption, not a computation)
    OPTIMA sizes its LN / GELU / adder lanes so they never bound a stage (it
    raises if one would). Pass 1 adopts the same contract: an analog stage's
    time is exactly the analog law; helpers are absorbed. Native per-op
    pointwise pricing (roofline against the stub hierarchy) stays untouched —
    it only feeds the sequential (non-FWS) total.

Macro resource model (QIF P2: cards, tiles, slicing, the per-macro pool)
    DEVICE CARDS. ``cim.analog`` and ``cim.fabric`` are the parameter halves
    of two cards: an ANALOG MACRO card and a SHARED DIGITAL CHIPLET card. The
    card adds the structural knobs — ``bits_per_cell`` / ``weight_bits``,
    the slicing arrangement, ``bank_depth`` (allocation granularity in mux
    slots), ``stack_3d_height`` (a FOOTPRINT divisor only), and a validity
    menu of admitted (bits_per_cell, weight_bits, mux) points that the tool
    REFUSES to leave rather than interpolate. A config with no ``cim.cards``
    block synthesizes cards that wrap these very objects with every knob
    inert, so shipped YAMLs price identically. The SA-attention and
    softmax-lane laws above are the digital chiplet card's laws (D13).

    TILES. A tile is one weight sub-matrix resident in one macro, carrying an
    owner (model, layer, op, expert, shard), a K/N range, a bit-slice index,
    and a site (macro id, row range, column-set ids). A column set is one mux
    slot — ``cols_adc`` stored columns, one ADC pass — and is the smallest
    allocatable unit. The array census IS the tile enumerator: at the shipped
    defaults (no slicing, one whole-macro bank) the enumerator returns exactly
    ``arrays(K, N)`` tiles on ``arrays(K, N)`` macros. Resident tiles must fit
    their macro's stored column sets; overflow, an out-of-range set, and a
    doubly-claimed set are :class:`MacroCapacityError`.

    ACTIVE-COLUMN-SET PRICING (ADJ-4). An op pays
    ``M_tokens * active_column_sets * slice_cycles / f_analog``, with
    ``active_column_sets`` the WORST macro's set count (macros holding one
    op's tiles convert concurrently). One owner filling every mux slot
    activates ``mux`` sets and reproduces ``M * vec_latency`` bit-identically
    — the pass-1 charge is the full-occupancy special case, and every OPTIMA
    parity configuration is full-occupancy (tested, not assumed). Analog
    energy reads the same way: ``energy_per_vec_pj`` is a whole macro's
    per-vector energy, so an op pays the fraction of stored columns it holds,
    and full occupancy returns ``M * E_vec * shots * macros``.

    BIT SLICING (D11). ``n_s = ceil(weight_bits / bits_per_cell)``; 1 unless a
    card opts in. The column-set arrangement multiplies stored-column demand
    by ``n_s`` and keeps one slice group's reduction local to a macro; the
    chained-macro arrangement gives each slice its own macros and sends
    ``n_s - 1`` partial words per output over the p2p law to a sink. Either
    way P2 emits plain-data :class:`ReductionOpDescriptor` records for the DAG
    builder and prices them with the shift-add law: ``(n_s - 1)`` adds per
    output, tree depth ``ceil(log2(n_s))``, pipelined at
    ``cycles = depth + ceil(results / lanes) - 1``.

    PER-MACRO DIGITAL POOL (D12). Sized so it never blocks: enough lanes to
    consume the macro's peak result rate ``cols_adc * f_analog /
    slice_cycles``, holding ``lanes * (n_s - 1)`` adders. The number is
    DERIVED and REPORTED, never a constraint. Area and energy come from card
    knobs that default to 0 — an undeclared cost reports zero rather than an
    invented number. Crossing ``POOL_DISCLOSURE_AREA_SHARE`` (1/5 of the
    served macro footprint, ADJ-4) prints a loud note and changes nothing.

    None of this is wired into the closed-form spatial report, which is
    legacy-frozen (A1, ADJ-8).

Attention outer-multiplier compensation (N_mult)
    Callers price ONE head-shape then multiply outside: SINGLE (tp=1)
    multiplies by ``B * kv_heads``; TENSOR / TENSOR_SEQUENCE (tp >= 2)
    multiply by ``B * kv_heads / tp``. The SA law computes the folded
    per-chip TOTAL ``T_chip``, so :meth:`CimDeviceModel.price_gemm` returns
    ``T_chip / N_mult`` and the caller's multiplication restores ``T_chip``.
    Because the attention-score and attention-output ops are folded into ONE
    concurrent fabric run, both ops return the same ``T_chip / N_mult``; the
    per-op *sequential* sum therefore counts the folded stage total once per
    act-GEMM op (twice per layer). The FWS spatial report — the authoritative
    output — uses the stage laws directly and counts it once.

``cim.fabric.model: gpu_native``
    Attention act x act GEMMs are NOT intercepted: ``price_gemm`` returns
    ``None`` for them and the native tile/roofline machinery (priced against
    the stub tech_param, which must then describe the sidecar) continues.

OPTIMA block-latency equivalence (recon note for validation)
    The recorded OPTIMA runs sum a sixth pipeline stage equal to exactly one
    analog stage time (their trailing peripherals stage), so OPTIMA's
    ``block_latency`` = :meth:`CimDeviceModel.block_latency` (S1..S5) plus
    ``analog_gemm_time(seq)``. Rapid-LLM lists endpoint stages separately
    instead; validation against the recorded numbers must add that one analog
    stage time.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import llm_util


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


# Op-name role classification (names verified from the ViT trace in
# train_timing.py: qkv_projection_f/b, attention_score_f/b,
# attention_output_f/b, output_projection_f/b, ffn_f/b, ffn1_f/b, ffn2_f/b,
# vit_patch_embed_f/b, vit_head_f/b, embedding_f/b, linear_softmax_f/b —
# and from the decode trace in inference_timing.py: decode_qkv_proj_f,
# decode_attention_score_f, decode_attention_output_f,
# decode_output_projection_f, decode_ffn1_f, decode_ffn2_f. A leading
# 'decode_' prefix is stripped before matching (see _split_decode_prefix).
# Backward twins (*_b) ARE priced during inference and then discarded; they
# get the same law on whatever dims arrive — never crash, never log per-call.
_ACT_GEMM_PREFIXES = ("attention_score", "attention_output")
_WEIGHT_GEMM_PREFIXES = (
    "vit_patch_embed",
    "qkv_proj",     # covers qkv_proj_* (decode) and qkv_projection_*
    "output_projection",
    "ffn",          # covers ffn_*, ffn1_*, ffn2_* incl. MoE buckets *_hot/_cold/_shared/_uniform
    "vit_head",
    "embedding",
    "linear",       # covers linear_softmax_*
    "router",       # MoE router (weight; arrays = ceil(H/rows)*ceil(E/(cols*mux)))
    "moe",          # moe_dispatch / moe_combine: pure-comm OperationTimings that
                    # never reach get_gemm_time — no pricing needed, never warn.
)

# MoE compute-bucket suffixes (train_timing get_moe_ffn_f bucket naming).
# Bucket ops arrive with token-true M (already includes the B streams), so
# the decode M = B scaling below must never apply to them. The balanced
# ("uniform") bucket carries the _uniform suffix for exactly this reason:
# its per-expert M is token-true too, and without the suffix a balanced
# decode MoE FFN op would be B-scaled twice (the plain decode_ffn*_f names
# are reserved for per-stream dense decode ops, which DO need M = B).
_MOE_BUCKET_SUFFIXES = ("_hot", "_cold", "_shared", "_uniform")


@dataclass(frozen=True)
class CimModelParams:
    """Model parameters the device laws and stage helpers need.

    Duck-typed sources (a parsed ``LLMConfig`` or a ``TimeCalculation``)
    are adapted via :meth:`from_model`.
    """

    hidden_dim: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    kv_heads: int
    head_dim: int
    seq_len: int
    batch_size: int
    #: True when the model uses a gated (SwiGLU-style) MLP pair. Derived from
    #: the model's uses_gated_mlp source (llm_util), NOT model_type checks.
    gated_mlp: bool
    patch_dim: int      # 0 disables the patch-embed endpoint stage
    num_classes: int    # 0 disables the classifier-head endpoint stage
    # --- MoE / vocab fields (pass 2; safe dense defaults) -------------------
    num_experts: int = 1
    top_k: int = 1
    moe_intermediate_size: int = 0     # 0 => use intermediate_size
    n_shared_experts: int = 0
    #: One-hot expert imbalance contract (MOE_ONE_HOT_EXPERT_MODEL.md):
    #: the hot expert carries factor alpha of a balanced share.
    expert_imbalance_factor: float = 1.0
    #: Per-layer MoE mask (True = MoE layer). Empty => all layers dense.
    moe_layer_mask: Tuple[bool, ...] = ()
    vocab_size: int = 0                # 0 disables the lm_head endpoint (LLM)
    disable_embedding_unembedding: bool = False

    @classmethod
    def from_model(cls, model) -> "CimModelParams":
        """Adapt any object carrying the model dims (LLMConfig or tc)."""
        hidden_dim = int(getattr(model, "hidden_dim"))
        num_heads = int(getattr(model, "num_heads"))
        head_dim = getattr(model, "head_dim", None)
        if head_dim is None:
            head_dim = hidden_dim // num_heads
        kv_heads = getattr(model, "kv_heads", None)
        if kv_heads is None:
            kv_heads = getattr(getattr(model, "attention", None), "kv_heads", None)
        if kv_heads is None:
            kv_heads = num_heads
        batch_size = getattr(model, "batch_size", None)
        if batch_size is None:
            gbs = int(getattr(model, "global_batch_size", 1) or 1)
            gas = int(getattr(model, "gradient_accumulation_steps", 1) or 1)
            batch_size = max(1, gbs // max(1, gas))
        # LLMConfig spells these num_experts / top_k; TimeCalculation spells
        # them moe_num_experts / moe_top_k. Duck-type both.
        num_experts = getattr(model, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(model, "moe_num_experts", 1)
        top_k = getattr(model, "top_k", None)
        if top_k is None:
            top_k = getattr(model, "moe_top_k", 1)
        mask_raw = getattr(model, "moe_layer_mask", None) or ()
        disable_embed = getattr(model, "disable_embedding_unembedding", None)
        if disable_embed is None:
            # TimeCalculation does not mirror this flag; read its model.
            disable_embed = getattr(
                getattr(model, "model", None), "disable_embedding_unembedding", False
            )
        # model_type: TimeCalculation only mirrors it AFTER the cim model is
        # built (train_timing sets self.model_type post-super().__init__), so
        # fall through to the tc's parsed model config; gatedness must not
        # silently degrade to the swiglu_mlp attr alone on the run path.
        model_type = getattr(model, "model_type", None)
        if model_type is None:
            model_type = getattr(getattr(model, "model", None), "model_type", "")
        return cls(
            hidden_dim=hidden_dim,
            intermediate_size=int(getattr(model, "intermediate_size")),
            num_layers=int(getattr(model, "num_layers")),
            num_heads=num_heads,
            kv_heads=int(kv_heads),
            head_dim=int(head_dim),
            seq_len=int(getattr(model, "seq_len")),
            batch_size=int(batch_size),
            gated_mlp=llm_util.uses_gated_mlp(
                model_type,
                swiglu_mlp=bool(getattr(model, "swiglu_mlp", False)),
            ),
            patch_dim=int(getattr(model, "patch_dim", 0) or 0),
            num_classes=int(getattr(model, "num_classes", 0) or 0),
            num_experts=max(1, int(num_experts or 1)),
            top_k=max(1, int(top_k or 1)),
            moe_intermediate_size=int(getattr(model, "moe_intermediate_size", 0) or 0),
            n_shared_experts=int(getattr(model, "n_shared_experts", 0) or 0),
            expert_imbalance_factor=float(getattr(model, "expert_imbalance_factor", 1.0) or 1.0),
            moe_layer_mask=tuple(bool(x) for x in mask_raw),
            vocab_size=int(getattr(model, "vocab_size", 0) or 0),
            disable_embedding_unembedding=bool(disable_embed),
        )

    # --- derived views ------------------------------------------------------

    @property
    def shared_heads(self) -> int:
        """Query heads sharing one KV head (llm_util: num_heads // kv_heads)."""
        return max(1, self.num_heads // max(1, self.kv_heads))

    @property
    def ffn1_fold(self) -> int:
        """FFN1 output-dim fold: the fused gated descriptor arrives with N=2*I."""
        return 2 if self.gated_mlp else 1

    @property
    def moe_intermediate(self) -> int:
        """Per-class intermediate size for MoE layers (dense size when unset)."""
        return self.moe_intermediate_size if self.moe_intermediate_size > 0 else self.intermediate_size

    @property
    def layer_mask(self) -> Tuple[bool, ...]:
        """MoE mask normalized to num_layers entries (padded with dense)."""
        mask = tuple(self.moe_layer_mask[: self.num_layers])
        if len(mask) < self.num_layers:
            mask = mask + (False,) * (self.num_layers - len(mask))
        return mask

    @property
    def num_moe_layers(self) -> int:
        return sum(self.layer_mask)

    @property
    def use_moe(self) -> bool:
        return self.num_experts > 1 and self.num_moe_layers > 0

    @property
    def is_vit_shaped(self) -> bool:
        """ViT-shaped endpoints (patch embed / classifier head) present."""
        return self.patch_dim > 0 or self.num_classes > 0

    @property
    def lm_head_enabled(self) -> bool:
        """LLM unembedding endpoint present (linear_softmax on the last chip)."""
        return (
            not self.is_vit_shaped
            and self.vocab_size > 0
            and not self.disable_embedding_unembedding
        )


@dataclass(frozen=True)
class AttentionTiming:
    """Folded-K attention result for one chip (see module docstring).

    qk/pv are named in SCORE orientation: for a call (m, k, n),
    qk_cycles = sa(m, n, k*h_rep) and pv_cycles = sa(m, k, n*h_rep).
    The total (max of the concurrent pair) is symmetric under k<->n, so an
    output-orientation call yields the same total with the labels swapped.
    """

    qk_cycles: int
    pv_cycles: int
    total_cycles: int       # max(qk, pv) + fill_drain_penalty
    sa_time_s: float        # total_cycles / f_fabric == T_chip
    softmax_cycles: int
    softmax_time_s: float
    stage_time_s: float     # max(sa_time_s, softmax_time_s) — report stage 2
    heads_chip: int
    heads_per_replica: int


@dataclass(frozen=True)
class DecodeS2Timing:
    """Decode stage-2 law: max(T_sa, T_softmax, kv_read_bytes / kv_bw)."""

    attention: AttentionTiming
    kv_read_bytes: float
    kv_read_time_s: float
    stage_time_s: float
    bound: str              # "sa" | "softmax" | "kv_read"


@dataclass(frozen=True)
class DecodeSustainedThroughput:
    """Reconciliation of the decode fabric ceiling with the KV stream cap.

    See :meth:`CimDeviceModel.decode_sustained_throughput` for the law.
    """

    wavefronts_full: int    # ceil(step latency / period): wavefronts that fill the pipeline
    wavefronts_kv: int      # floor(kv_max_streams / B): wavefronts the KV capacity holds
    tokens_per_s: float     # min(wavefronts_full, wavefronts_kv) * B / step latency
    limiting_factor: str    # "fabric" | "kv_capacity" | "infeasible"


# ---------------------------------------------------------------------------
# The macro resource model (QIF P2): cards, tiles, sites, sliced reductions
#
# A3: the macro is the atomic resource and the TILE is the allocation unit.
# The pass-1 array census counts the same units; a tile adds the identity the
# census never had — a named owner (D21) and a site.
# ---------------------------------------------------------------------------


class MacroCapacityError(ValueError):
    """Resident tiles claim more (or overlapping) column sets than a macro stores.

    The named hard error of the capacity check: a macro's stored columns are
    finite, so an allocation that overflows one is refused, never absorbed.
    """


class CardValidityError(ValueError):
    """An unlisted (bits_per_cell, weight_bits, mux) point was requested.

    The card's validity menu says what the device admits. The tool refuses an
    unlisted point instead of interpolating between listed ones.
    """


@dataclass(frozen=True)
class TileOwner:
    """Who owns a tile: model, layer, op, expert, shard (D21)."""

    model: str = ""
    layer: int = 0
    op: str = ""
    expert: int = -1        # -1 = not an expert tile
    shard: int = 0

    @property
    def label(self) -> str:
        expert = "" if self.expert < 0 else f".e{self.expert}"
        return f"{self.model}.L{self.layer}.{self.op}{expert}.s{self.shard}"


@dataclass(frozen=True)
class TileSite:
    """Where a tile sits: macro id, row range, column-set (mux-slot) ids."""

    macro_id: int
    row_start: int
    row_end: int                      # exclusive
    column_sets: Tuple[int, ...]

    @property
    def num_column_sets(self) -> int:
        return len(self.column_sets)


@dataclass(frozen=True)
class Tile:
    """One weight sub-matrix resident in one macro, with a name and a site.

    ``k_start``/``k_end`` and ``n_start``/``n_end`` are half-open ranges of
    the owner's weight matrix; ``slice_index`` names the bit slice (0 when the
    card declares no slicing).
    """

    owner: TileOwner
    k_start: int
    k_end: int
    n_start: int
    n_end: int
    slice_index: int
    site: TileSite

    @property
    def rows_used(self) -> int:
        return self.k_end - self.k_start

    @property
    def logical_columns(self) -> int:
        return self.n_end - self.n_start

    @property
    def group_key(self) -> Tuple[object, int, int]:
        """Slice-group key: the tiles that reduce into one output block."""
        return (self.owner, self.k_start, self.n_start)


@dataclass(frozen=True)
class TiledOpCost:
    """What one op on one owner's tiles costs (P2.3).

    ``active_column_sets`` is the charge: the ADC/mux path serializes, one
    stored column set converts at a time, and macros fire in parallel — so the
    op pays the WORST macro's active-set count, not the full mux depth.
    """

    time_s: float
    active_column_sets: int
    macros: int
    total_active_column_sets: int
    energy_pj: float


@dataclass(frozen=True)
class ReductionOpDescriptor:
    """Plain data for one shift-and-add reduction the DAG builder will consume.

    Emitted, never executed here: P2 says what the op is and what it costs;
    the DAG builder (P3/P4) turns it into a node. ``transport_partials`` is
    the number of partial words that must cross a link before the tree runs
    (0 for the column-set arrangement, n_slices-1 for chained macros).
    """

    kind: str                          # "shift_add_tree"
    arrangement: str                   # "column_sets" | "chained_macros"
    owner: TileOwner
    operand_tiles: Tuple[Tile, ...]
    output_lanes: int                  # logical output columns the tree produces
    n_slices: int
    adds_per_output: int               # n_slices - 1
    depth: int                         # ceil(log2(n_slices))
    site_macro_id: int                 # the pool that hosts the tree (sink when chained)
    local: bool                        # every operand slice sits in site_macro_id
    transport_partials: int            # partial words per output that cross a link


@dataclass(frozen=True)
class ReductionCost:
    """Priced shift-and-add tree: pipelined, one result per cycle after fill."""

    adds: float
    cycles: int
    time_s: float
    energy_pj: float
    transport_bytes: float


@dataclass(frozen=True)
class DigitalPoolSizing:
    """Derived per-macro digital pool (D12, P2.5) — REPORTED, never a constraint.

    Sized so it never blocks: the pool consumes the macro's peak result rate
    (``cols_adc * f_analog / slice_cycles`` results per second). ``disclose``
    fires at the ADJ-4 threshold — a pool whose derived area reaches 1/5 of
    the macro footprint it serves gets a loud note and nothing else.
    """

    result_rate_per_s: float
    pool_clock_hz: float
    lanes: int
    adders: int
    area_mm2: float
    macro_footprint_mm2: float
    area_share: float
    disclose: bool
    note: Optional[str]


#: ADJ-4: a per-macro pool this large stops being absorbed silently. It is a
#: DISCLOSURE threshold — crossing it prints a note and changes no number.
POOL_DISCLOSURE_AREA_SHARE = 0.2


def check_allocation_capacity(column_sets_per_macro: int, assignments) -> None:
    """Capacity-check user-written tile assignments (D10).

    A macro stores ``mux`` column sets, so range plus uniqueness IS the
    capacity: no set outside 0..mux-1, and no set claimed twice. Lives at
    module scope because `config` runs it while parsing `cim.allocation`,
    before any device model exists.
    """
    mux = int(column_sets_per_macro)
    claimed = {}
    for entry in assignments:
        label = TileOwner(
            model=getattr(entry, "model", ""),
            layer=getattr(entry, "layer", 0),
            op=getattr(entry, "op", ""),
            expert=getattr(entry, "expert", -1),
            shard=getattr(entry, "shard", 0),
        ).label
        for column_set in entry.column_sets:
            if column_set < 0 or column_set >= mux:
                raise MacroCapacityError(
                    f"macro {entry.macro}: {label} claims column set {column_set}, but the "
                    f"card stores {mux} column sets (mux slots) per macro."
                )
            key = (entry.macro, column_set)
            if key in claimed:
                raise MacroCapacityError(
                    f"macro {entry.macro}: column set {column_set} is claimed by both "
                    f"{claimed[key]} and {label}. One column set holds one tile."
                )
            claimed[key] = label


class CimDeviceModel:
    """Closed-form FWS-CIM device laws (see module docstring for each law)."""

    def __init__(self, hw_config, model_params):
        cim = getattr(hw_config, "cim_config", None)
        if cim is None:
            raise ValueError(
                "CimDeviceModel requires hw_config.cim_config (a parsed 'cim:' block); "
                "got None. Set device_class: fws_cim with a cim block."
            )
        self.cim = cim
        # The active device cards own the parameter halves the laws read
        # (P2.1). A config with no `cim.cards` block synthesizes cards that
        # wrap these very objects, so this binding moves no number.
        cards = getattr(cim, "cards", None)
        self.analog = cim.analog if cards is None else cards.analog_card.params
        self.fabric = cim.fabric if cards is None else cards.digital_card.fabric
        self.chip = cim.chip
        if isinstance(model_params, CimModelParams):
            self.params = model_params
        else:
            self.params = CimModelParams.from_model(model_params)
        self._warned_unknown_ops = set()
        if self.fabric.model == "sa" and int(self.fabric.num_arrays) < 2:
            print(
                "[WARNING]: cim.fabric.num_arrays < 2 — the folded attention law "
                "assumes the QK and PV runs occupy separate arrays concurrently; "
                "timing will be optimistic."
            )

    # ------------------------------------------------------------------
    # Analog weight-GEMM law
    # ------------------------------------------------------------------

    @property
    def vec_cycles(self) -> int:
        """Analog cycles to evaluate one input vector: adc_mux * slice_cycles."""
        return int(self.analog.adc_mux) * int(self.analog.slice_cycles)

    @property
    def vec_latency_s(self) -> float:
        """Seconds per input vector: vec_cycles / f_analog."""
        return self.vec_cycles / (float(self.analog.analog_clock_mhz) * 1e6)

    def analog_gemm_time(self, m_tokens: float) -> float:
        """Weight-GEMM time: T = M_tokens * vec_latency (K/N-independent)."""
        return float(m_tokens) * self.vec_latency_s

    def arrays(self, k: int, n: int) -> int:
        """Arrays a K x N weight matrix occupies."""
        return _ceil_div(k, self.analog.rows) * _ceil_div(n, self.analog.cols)

    def analog_stage_energy_pj(self, m_tokens: float, num_arrays: int) -> float:
        """Analog energy per stage per layer: M * E_vec * shots * arrays (pJ)."""
        return (
            float(m_tokens)
            * float(self.analog.energy_per_vec_pj)
            * int(self.analog.shots_per_output)
            * int(num_arrays)
        )

    # ------------------------------------------------------------------
    # Digital fabric (attention sidecar)
    # ------------------------------------------------------------------

    @property
    def f_fabric_hz(self) -> float:
        return float(self.fabric.clock_ghz) * 1e9

    def sa_cycles(self, m: int, n: int, k: int) -> int:
        """Systolic-array closed form: ceil(M/R)*ceil(N/C)*(K+R+C-2) - 1."""
        r = int(self.fabric.rows)
        c = int(self.fabric.cols)
        return _ceil_div(m, r) * _ceil_div(n, c) * (int(k) + r + c - 2) - 1

    def heads_chip(self, kv_heads: Optional[int] = None, tp: int = 1) -> int:
        """KV heads resident on one chip: ceil(kv_heads/tp) when tp >= 2."""
        kv = int(self.params.kv_heads if kv_heads is None else kv_heads)
        tp = max(1, int(tp))
        return _ceil_div(kv, tp) if tp >= 2 else kv

    def softmax_cycles(self, tokens_q: int, heads_chip: int) -> int:
        """Softmax-lanes law: pipeline_depth + ceil(tokens_q*heads_chip/lanes) - 1.

        tokens_q is the query-row count of the score call (prefill: S — the
        pass-1 form is unchanged; decode: B * shared_heads).
        """
        lanes = int(self.fabric.softmax_lanes) * int(self.fabric.replicas)
        return (
            int(self.fabric.softmax_pipeline_depth)
            + _ceil_div(int(tokens_q) * int(heads_chip), lanes)
            - 1
        )

    def attention_call_timing(
        self,
        m: int,
        k: int,
        n: int,
        kv_heads: Optional[int] = None,
        tp: int = 1,
        softmax_tokens: Optional[int] = None,
        streams: int = 1,
    ) -> AttentionTiming:
        """Folded-K attention at CALL DIMS (m, k, n) on one chip.

        Heads fold into K exactly as in pass 1, and ``streams`` (the B
        independent batch streams the stage prices) folds into K the same
        way: the concurrent pair is run_a = sa(m, n, k*h_rep*streams) and
        run_b = sa(m, k, n*h_rep*streams), total = max + fill_drain (one
        penalty for the whole batch). For a score-orientation call
        (m, head_dim, context) run_a is QK^T and run_b is PV; MHA prefill
        (m=S, k=d, n=S, streams=1) reproduces the pass-1 shapes exactly.
        softmax_tokens defaults to streams * m (the B-scaled query rows);
        decode passes B * shared_heads explicitly.
        """
        h_chip = self.heads_chip(kv_heads, tp)
        h_rep = _ceil_div(h_chip, self.fabric.replicas)
        s_fold = max(1, int(streams))
        qk = self.sa_cycles(m, n, int(k) * h_rep * s_fold)
        pv = self.sa_cycles(m, k, int(n) * h_rep * s_fold)
        total = max(qk, pv) + int(self.fabric.fill_drain_penalty_cycles)
        sa_time = total / self.f_fabric_hz
        sm_tokens = int(s_fold * m if softmax_tokens is None else softmax_tokens)
        sm_cycles = self.softmax_cycles(sm_tokens, h_chip)
        sm_time = sm_cycles / self.f_fabric_hz
        return AttentionTiming(
            qk_cycles=qk,
            pv_cycles=pv,
            total_cycles=total,
            sa_time_s=sa_time,
            softmax_cycles=sm_cycles,
            softmax_time_s=sm_time,
            stage_time_s=max(sa_time, sm_time),
            heads_chip=h_chip,
            heads_per_replica=h_rep,
        )

    def attention_timing(
        self,
        seq_len: Optional[int] = None,
        head_dim: Optional[int] = None,
        kv_heads: Optional[int] = None,
        tp: int = 1,
        streams: int = 1,
    ) -> AttentionTiming:
        """MHA-prefill view of the call-dims law (m=S, k=head_dim, n=S)."""
        s = int(self.params.seq_len if seq_len is None else seq_len)
        d = int(self.params.head_dim if head_dim is None else head_dim)
        return self.attention_call_timing(
            m=s, k=d, n=s, kv_heads=kv_heads, tp=tp, streams=streams
        )

    def decode_attention_timing(
        self,
        context: int,
        batch_size: Optional[int] = None,
        kv_heads: Optional[int] = None,
        tp: int = 1,
    ) -> AttentionTiming:
        """Decode attention at the step's context (score call orientation).

        score m = shared_heads, k = head_dim, n = context (llm_util decode
        descriptors), streams = B: the B streams' independent attention
        problems fold into the contraction like heads (the wavefront is one
        batch of B streams, so the SA prices all B). Softmax elements are
        B-scaled the same way: tokens_q = B * shared_heads at the step's
        context.
        """
        p = self.params
        b = int(p.batch_size if batch_size is None else batch_size)
        return self.attention_call_timing(
            m=p.shared_heads,
            k=p.head_dim,
            n=int(context),
            kv_heads=kv_heads,
            tp=tp,
            softmax_tokens=b * p.shared_heads,
            streams=b,
        )

    # ------------------------------------------------------------------
    # Per-op pricing (consumer: base_timing.get_gemm_time early branch)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_decode_prefix(name: str) -> Tuple[str, bool]:
        """Strip a leading 'decode_' prefix; return (base_name, is_decode)."""
        n = str(name or "").lower()
        if n.startswith("decode_"):
            return n[len("decode_"):], True
        return n, False

    def _classify_op(self, name: str) -> str:
        n, _ = self._split_decode_prefix(name)
        for prefix in _ACT_GEMM_PREFIXES:
            if n.startswith(prefix):
                return "act"
        for prefix in _WEIGHT_GEMM_PREFIXES:
            if n.startswith(prefix):
                return "weight"
        return "unknown"

    @staticmethod
    def _attention_n_mult(tc) -> float:
        """Caller's outer multiplier for one attention op (see module docstring).

        Mirrors train_timing: SINGLE (tp=1) callers multiply the per-shape time
        by batch = B*kv_heads; TENSOR / TENSOR_SEQUENCE (tp >= 2) callers
        multiply by batch * (1/tp) = B*kv_heads/tp. Context parallelism (cp>1)
        prices attention differently and stays rejected for fws_cim.
        """
        tp = max(1, int(getattr(tc, "tp", 1) or 1))
        cp = max(1, int(getattr(tc, "cp", 1) or 1))
        if cp > 1:
            raise ValueError(
                "device_class: fws_cim does not support context parallelism "
                f"(got parallelism.cp = {cp}); set cp: 1."
            )
        n_mult = float(int(tc.batch_size) * int(tc.kv_heads))
        if tp >= 2:
            n_mult /= tp
        return n_mult

    def price_gemm(self, name: str, dim1: int, dim2: int, dim3: int, tc) -> Optional[float]:
        """Price one GEMM call by op-name role; None means fall through.

        Returns the kernel time in seconds WITHOUT launch overhead (the
        get_gemm_time branch adds ``self.O`` unless disable_overhead), or
        ``None`` for attention act-GEMMs under ``fabric.model: gpu_native``
        (the native tile/roofline path continues).

        ``tc`` needs only: ``batch_size``, ``kv_heads``, ``tp``, ``cp``
        (all set by TimeCalculation.__init__). Attention is priced at the
        CALL DIMS (dim1, dim2, dim3) with heads AND the B batch streams
        folded into K (the caller's n_mult multiply then restores the one
        B-folded fabric run per op); non-decode
        weight GEMMs use only ``dim1`` (M as received). Decode weight ops
        arrive per-stream (m=1, B streams outside the call, and the caller
        never multiplies non-attention time by the descriptor batch), so
        they are priced at M = B * dim1 — EXCEPT MoE bucket ops
        (*_hot/_cold/_shared/_uniform), whose M is already token-true.
        """
        base, is_decode = self._split_decode_prefix(name)
        role = self._classify_op(name)
        if role == "act":
            if self.fabric.model == "gpu_native":
                return None
            att = self.attention_call_timing(
                m=int(dim1),
                k=int(dim2),
                n=int(dim3),
                kv_heads=int(tc.kv_heads),
                tp=max(1, int(getattr(tc, "tp", 1) or 1)),
                streams=max(1, int(getattr(tc, "batch_size", 1) or 1)),
            )
            return att.sa_time_s / self._attention_n_mult(tc)
        if role == "unknown":
            warn_base = base
            for suffix in ("_f", "_b"):
                if warn_base.endswith(suffix):
                    warn_base = warn_base[: -len(suffix)]
                    break
            if warn_base not in self._warned_unknown_ops:
                self._warned_unknown_ops.add(warn_base)
                print(
                    f"[WARNING]: fws_cim: unknown GEMM op name '{name}' — pricing "
                    "with the analog weight-GEMM law (T = M * vec_latency)."
                )
        m_tokens = int(dim1)
        if is_decode and not base.endswith(_MOE_BUCKET_SUFFIXES):
            m_tokens *= max(1, int(getattr(tc, "batch_size", 1) or 1))
        return self.analog_gemm_time(m_tokens)

    # ------------------------------------------------------------------
    # Stage-level helpers (consumer: the FWS spatial report, DESIGN 4.3)
    # ------------------------------------------------------------------

    def per_layer_stage_arrays(self) -> "OrderedDict[str, int]":
        """Analog arrays per DENSE transformer layer, by stage.

        FFN1 uses the FUSED gated descriptor (N = 2*I when gated) — the
        pass-2 refinement of the pass-1 "x2" note; identical for every
        pass-1 validation geometry (none is gated at an I that straddles a
        column boundary).
        """
        return OrderedDict(
            (stage, self.arrays(k, n)) for stage, (k, n) in self.per_layer_stage_shapes().items()
        )

    def per_layer_stage_shapes(self) -> "OrderedDict[str, Tuple[int, int]]":
        """The (K, N) weight-matrix shape each dense stage occupies.

        The shapes the census counts and the tile enumerator tiles (P2.2);
        exported so a mapper can name a tile's owner op without restating a
        model shape.
        """
        p = self.params
        qkv_n = (p.num_heads + 2 * p.kv_heads) * p.head_dim  # 3H for MHA
        return OrderedDict(
            (
                ("qkv", (p.hidden_dim, qkv_n)),
                ("o_proj", (p.num_heads * p.head_dim, p.hidden_dim)),
                ("ffn1", (p.hidden_dim, p.ffn1_fold * p.intermediate_size)),
                ("ffn2", (p.intermediate_size, p.hidden_dim)),
            )
        )

    def arrays_per_layer(self) -> int:
        return sum(self.per_layer_stage_arrays().values())

    # --- MoE layer class (experts-per-chip) --------------------------------

    def router_arrays(self) -> int:
        """Router weight arrays: ceil(H/rows) * ceil(E/(cols_adc*mux))."""
        p = self.params
        return self.arrays(p.hidden_dim, p.num_experts)

    def moe_expert_ffn_arrays(self) -> int:
        """Arrays ONE expert's FFN pair occupies (fused gated descriptor)."""
        p = self.params
        i_moe = p.moe_intermediate
        return self.arrays(p.hidden_dim, p.ffn1_fold * i_moe) + self.arrays(i_moe, p.hidden_dim)

    def moe_layer_stage_arrays(self) -> "OrderedDict[str, int]":
        """Analog arrays per MoE transformer layer, by stage.

        Attention arrays match the dense class; routed experts contribute
        E expert-FFN pairs, shared experts n_shared pairs, plus the router.
        """
        p = self.params
        dense = self.per_layer_stage_arrays()
        i_moe = p.moe_intermediate
        ffn1 = self.arrays(p.hidden_dim, p.ffn1_fold * i_moe)
        ffn2 = self.arrays(i_moe, p.hidden_dim)
        return OrderedDict(
            (
                ("qkv", dense["qkv"]),
                ("o_proj", dense["o_proj"]),
                ("router", self.router_arrays()),
                ("ffn1_routed", p.num_experts * ffn1),
                ("ffn2_routed", p.num_experts * ffn2),
                ("ffn1_shared", p.n_shared_experts * ffn1),
                ("ffn2_shared", p.n_shared_experts * ffn2),
            )
        )

    def arrays_per_moe_layer(self) -> int:
        return sum(self.moe_layer_stage_arrays().values())

    def moe_routed_arrays(self) -> int:
        """Routed-expert arrays of one MoE layer (the spreadable portion)."""
        stages = self.moe_layer_stage_arrays()
        return stages["ffn1_routed"] + stages["ffn2_routed"]

    def layer_class_mask(self) -> Tuple[bool, ...]:
        """Per-layer class (True = MoE), all-dense when the model has no MoE."""
        p = self.params
        if not p.use_moe:
            return (False,) * p.num_layers
        return p.layer_mask

    def arrays_for_layer(self, layer_idx: int) -> int:
        """Total analog arrays of one layer, by its class."""
        if self.layer_class_mask()[layer_idx]:
            return self.arrays_per_moe_layer()
        return self.arrays_per_layer()

    def endpoint_arrays(self) -> "OrderedDict[str, int]":
        """Arrays for the model-shaped endpoint stages.

        ViT: patch_embed + vit_head (0 when a stage is absent). LLM:
        lm_head (linear_softmax) unless disable_embedding_unembedding.
        The LLM embedding lookup is not a GEMM (report note only).
        """
        p = self.params
        if p.is_vit_shaped:
            patch = self.arrays(p.patch_dim, p.hidden_dim) if p.patch_dim > 0 else 0
            head = self.arrays(p.hidden_dim, p.num_classes) if p.num_classes > 0 else 0
            return OrderedDict((("patch_embed", patch), ("vit_head", head)))
        if p.lm_head_enabled:
            return OrderedDict((("lm_head", self.arrays(p.hidden_dim, p.vocab_size)),))
        return OrderedDict()

    def transformer_stack_arrays(self) -> int:
        mask = self.layer_class_mask()
        num_moe = sum(mask)
        num_dense = len(mask) - num_moe
        return num_dense * self.arrays_per_layer() + num_moe * self.arrays_per_moe_layer()

    def total_arrays(self) -> int:
        return self.transformer_stack_arrays() + sum(self.endpoint_arrays().values())

    def prefill_attention_timing(
        self, seq_len: Optional[int] = None, tp: int = 1, streams: int = 1
    ) -> AttentionTiming:
        """Prefill attention at the score call dims (GQA-aware).

        score m = S * shared_heads, k = head_dim, n = S (llm_util prefill
        descriptors); MHA (shared_heads == 1, streams=1) reduces to the
        pass-1 shapes. ``streams`` is the number of batch streams the
        wavefront carries (an LLM prefill report passes streams = B, so
        S2 prices the same B streams whose tokens S1/S3/S4/S5 price).
        """
        p = self.params
        s = int(p.seq_len if seq_len is None else seq_len)
        return self.attention_call_timing(
            m=s * p.shared_heads, k=p.head_dim, n=s, tp=tp, streams=streams
        )

    def layer_stage_times(
        self,
        seq_len: Optional[int] = None,
        tp: int = 1,
        tokens: Optional[int] = None,
        streams: int = 1,
    ) -> "OrderedDict[str, float]":
        """Per-layer DENSE stage times S1..S5 (seconds); helpers are absorbed.

        S1/S3/S4/S5 are the analog law at ``tokens`` (default: seq_len —
        the pass-1 form; an LLM prefill report passes tokens = B*S);
        S2 = max(T_sa, T_softmax) at the prefill score call dims with
        ``streams`` batch streams (an LLM prefill report passes streams =
        B so every stage prices the same wavefront of B streams).
        """
        s = int(self.params.seq_len if seq_len is None else seq_len)
        t_analog = self.analog_gemm_time(s if tokens is None else tokens)
        att = self.prefill_attention_timing(seq_len=s, tp=tp, streams=streams)
        return OrderedDict(
            (
                ("S1_qkv", t_analog),
                ("S2_attention", att.stage_time_s),
                ("S3_o_proj", t_analog),
                ("S4_ffn1", t_analog),
                ("S5_ffn2", t_analog),
            )
        )

    # --- MoE stage-time laws (experts-per-chip) ----------------------------

    def moe_tokens_hot(self, tokens_owner: int) -> int:
        """Hot-expert token load: ceil(tokens_owner * top_k * alpha / E)."""
        p = self.params
        return int(
            math.ceil(
                float(tokens_owner) * float(p.top_k) * float(p.expert_imbalance_factor)
                / float(p.num_experts)
            )
        )

    def moe_routed_ffn_time(self, tokens_owner: int) -> float:
        """Parallel-expert routed FFN stage: T = tokens_hot * vec_latency.

        All E expert arrays coexist and fire in parallel; the hot expert
        (one-hot imbalance contract, factor alpha) binds the stage.
        """
        return self.moe_tokens_hot(tokens_owner) * self.vec_latency_s

    def moe_shared_ffn_time(self, tokens_owner: int) -> float:
        """Shared experts run concurrently over ALL owner tokens."""
        if self.params.n_shared_experts <= 0:
            return 0.0
        return float(tokens_owner) * self.vec_latency_s

    def moe_ffn_stage_time(self, tokens_owner: int) -> float:
        """MoE FFN stage = max(routed, shared).

        Identity: E=1, top_k=1, alpha=1, n_shared=0 reduces exactly to the
        dense FFN stage time analog_gemm_time(tokens_owner).
        """
        return max(
            self.moe_routed_ffn_time(tokens_owner),
            self.moe_shared_ffn_time(tokens_owner),
        )

    def moe_layer_stage_times(
        self,
        seq_len: Optional[int] = None,
        tp: int = 1,
        tokens_owner: Optional[int] = None,
        streams: int = 1,
    ) -> "OrderedDict[str, float]":
        """Per-layer MoE stage times (seconds); helpers are absorbed.

        tokens_owner defaults to seq_len (B folded in by the caller for
        B > 1 — the pass-1 report convention); ``streams`` carries the
        same B to the attention stage. Dispatch/combine are boundary
        transfers, not stages (moe_dispatch_time).
        """
        s = int(self.params.seq_len if seq_len is None else seq_len)
        t_owner = int(s if tokens_owner is None else tokens_owner)
        t_analog = self.analog_gemm_time(t_owner)
        att = self.prefill_attention_timing(seq_len=s, tp=tp, streams=streams)
        t_ffn = self.moe_ffn_stage_time(t_owner)
        return OrderedDict(
            (
                ("S1_qkv", t_analog),
                ("S2_attention", att.stage_time_s),
                ("S3_o_proj", t_analog),
                ("S4_router", t_analog),
                ("S5_ffn1_moe", t_ffn),
                ("S6_ffn2_moe", t_ffn),
            )
        )

    def block_latency(self, seq_len: Optional[int] = None, tp: int = 1) -> float:
        """One-layer latency: sum(S1..S5).

        NOTE: OPTIMA's recorded block latency additionally sums one analog
        stage time (its trailing peripherals stage); compare against
        ``block_latency(seq) + analog_gemm_time(seq)``.
        """
        return sum(self.layer_stage_times(seq_len, tp).values())

    def endpoint_stage_times(
        self,
        seq_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        decode: bool = False,
    ) -> "OrderedDict[str, float]":
        """Model-shaped endpoint stages (analog law).

        ViT: patch_embed at M=seq, vit_head at M=B (pass-1 form). LLM:
        lm_head at M = B*S prefill / M = B decode, on the last chip.
        """
        p = self.params
        s = int(p.seq_len if seq_len is None else seq_len)
        b = int(p.batch_size if batch_size is None else batch_size)
        times = OrderedDict()
        if p.patch_dim > 0:
            times["patch_embed"] = self.analog_gemm_time(s)
        if p.num_classes > 0:
            times["vit_head"] = self.analog_gemm_time(b)
        if p.lm_head_enabled:
            times["lm_head"] = self.analog_gemm_time(b if decode else b * s)
        return times

    @property
    def embedding_note(self) -> Optional[str]:
        """Report note for the LLM embedding lookup (not a GEMM; stub-priced)."""
        p = self.params
        if p.is_vit_shaped or p.vocab_size <= 0 or p.disable_embedding_unembedding:
            return None
        return (
            "embedding lookup is a memory roofline priced on the stub hierarchy "
            "(not an analog GEMM); set model_param.disable_embedding_unembedding "
            "for pure-transformer studies."
        )

    def all_stage_times(
        self,
        seq_len: Optional[int] = None,
        tp: int = 1,
        batch_size: Optional[int] = None,
        tokens: Optional[int] = None,
        streams: int = 1,
    ) -> "OrderedDict[str, float]":
        """Stages of every ACTIVE layer class followed by the endpoint stages.

        Dense stages (S1..S5) appear when the model has at least one dense
        layer; MoE stages appear with a ``moe_`` key prefix when it has at
        least one MoE layer. ``tokens`` is the analog-stage token count
        (default: seq_len — the pass-1 convention; an LLM report passes
        tokens = B*S) and ``streams`` the attention-stage batch fold (an
        LLM report passes streams = B). All-dense models at streams=1
        reproduce the pass-1 table exactly.
        """
        mask = self.layer_class_mask()
        num_moe = sum(mask)
        times = OrderedDict()
        if len(mask) - num_moe > 0:
            times.update(
                self.layer_stage_times(seq_len, tp, tokens=tokens, streams=streams)
            )
        if num_moe > 0:
            moe_times = self.moe_layer_stage_times(
                seq_len, tp, tokens_owner=tokens, streams=streams
            )
            for stage, t in moe_times.items():
                times[f"moe_{stage}"] = t
        times.update(self.endpoint_stage_times(seq_len, batch_size))
        return times

    def pipeline_period(
        self,
        seq_len: Optional[int] = None,
        tp: int = 1,
        batch_size: Optional[int] = None,
        tokens: Optional[int] = None,
        streams: int = 1,
    ) -> Tuple[float, str]:
        """(period_s, bottleneck stage name): period = max over ALL stages
        of every layer class plus the endpoint stages."""
        times = self.all_stage_times(seq_len, tp, batch_size, tokens=tokens, streams=streams)
        bottleneck = max(times, key=times.get)
        return times[bottleneck], bottleneck

    # ------------------------------------------------------------------
    # Decode + KV stories (inference.kvcache_type: cim_sram | cim_dram)
    # ------------------------------------------------------------------

    def kv_bytes_per_stream_layer(
        self, context: int, kv_precision_bytes: float, tp: int = 1
    ) -> float:
        """KV bytes ONE stream holds for ONE layer on ONE device (sharded).

        Mirrors memory_estimation's per-device GQA law: 2 (K and V) *
        ceil(kv_heads / tp) * head_dim * context * kv_precision_bytes.
        """
        p = self.params
        kv_heads_per_tp = math.ceil(int(p.kv_heads) / max(1, int(tp)))
        return (
            2.0
            * float(kv_heads_per_tp)
            * float(p.head_dim)
            * float(context)
            * float(kv_precision_bytes)
        )

    def kv_bytes_per_stream(
        self, context: int, kv_precision_bytes: float, tp: int = 1
    ) -> float:
        """KV bytes one stream holds across ALL layers on one device."""
        return self.params.num_layers * self.kv_bytes_per_stream_layer(
            context, kv_precision_bytes, tp
        )

    def kv_read_bytes(
        self,
        context: int,
        kv_precision_bytes: float,
        batch_size: Optional[int] = None,
        tp: int = 1,
    ) -> float:
        """KV bytes ONE decode step reads per layer (all B streams, sharded)."""
        b = int(self.params.batch_size if batch_size is None else batch_size)
        return float(b) * self.kv_bytes_per_stream_layer(context, kv_precision_bytes, tp)

    def kv_max_streams(
        self,
        kv_capacity_bytes: float,
        context: int,
        kv_precision_bytes: float,
        tp: int = 1,
    ) -> int:
        """Streams the KV capacity can hold at the given context."""
        per_stream = self.kv_bytes_per_stream(context, kv_precision_bytes, tp)
        if per_stream <= 0:
            return 0
        return int(float(kv_capacity_bytes) // per_stream)

    def kv_max_context(
        self,
        kv_capacity_bytes: float,
        kv_precision_bytes: float,
        batch_size: Optional[int] = None,
        tp: int = 1,
    ) -> int:
        """Longest context the KV capacity can hold at the configured B."""
        b = int(self.params.batch_size if batch_size is None else batch_size)
        per_token = float(b) * self.kv_bytes_per_stream(1, kv_precision_bytes, tp)
        if per_token <= 0:
            return 0
        return int(float(kv_capacity_bytes) // per_token)

    def kv_story_bandwidth(
        self, kvcache_type: str, sram_bandwidth_bytes_per_s: Optional[float] = None
    ) -> float:
        """KV read bandwidth for the configured story.

        cim_sram: KV lives in the fabric activation SRAM (the DRAM-stub
        tier); the caller passes that tier's bandwidth. cim_dram: the
        optional cim.kv_dram block's bandwidth. hbm_only is invalid — the
        device has no HBM (config validation rejects it; defense in depth).
        """
        story = str(kvcache_type or "").strip().lower()
        if story == "cim_sram":
            if sram_bandwidth_bytes_per_s is None:
                raise ValueError(
                    "kv_story_bandwidth('cim_sram') needs the activation-SRAM "
                    "(DRAM-stub tier) bandwidth from the stub hierarchy."
                )
            return float(sram_bandwidth_bytes_per_s)
        if story == "cim_dram":
            if self.cim.kv_dram is None:
                raise ValueError(
                    "inference.kvcache_type: cim_dram requires the cim.kv_dram "
                    "block (capacity_bytes / bandwidth_bytes_per_s)."
                )
            return float(self.cim.kv_dram.bandwidth_bytes_per_s)
        raise ValueError(
            f"fws_cim has no KV story {kvcache_type!r}: the device has no HBM; "
            "use inference.kvcache_type: cim_sram or cim_dram."
        )

    def kv_story_capacity(
        self, kvcache_type: str, sram_capacity_bytes: Optional[float] = None
    ) -> float:
        """KV capacity (bytes) for the configured story.

        cim_sram: KV shares the fabric activation SRAM (the DRAM-stub
        tier); the caller passes that tier's size. cim_dram: the optional
        cim.kv_dram block's capacity. hbm_only is invalid — the device has
        no HBM (config validation rejects it; defense in depth).
        """
        story = str(kvcache_type or "").strip().lower()
        if story == "cim_sram":
            if sram_capacity_bytes is None:
                raise ValueError(
                    "kv_story_capacity('cim_sram') needs the activation-SRAM "
                    "(DRAM-stub tier) capacity from the stub hierarchy."
                )
            return float(sram_capacity_bytes)
        if story == "cim_dram":
            if self.cim.kv_dram is None:
                raise ValueError(
                    "inference.kvcache_type: cim_dram requires the cim.kv_dram "
                    "block (capacity_bytes / bandwidth_bytes_per_s)."
                )
            return float(self.cim.kv_dram.capacity_bytes)
        raise ValueError(
            f"fws_cim has no KV story {kvcache_type!r}: the device has no HBM; "
            "use inference.kvcache_type: cim_sram or cim_dram."
        )

    def decode_s2_timing(
        self,
        context: int,
        kv_bandwidth_bytes_per_s: float,
        kv_precision_bytes: float,
        batch_size: Optional[int] = None,
        tp: int = 1,
    ) -> DecodeS2Timing:
        """Decode stage 2: max(T_sa, T_softmax, kv_read_bytes / kv_bw)."""
        b = int(self.params.batch_size if batch_size is None else batch_size)
        att = self.decode_attention_timing(context, batch_size=b, tp=tp)
        kv_bytes = self.kv_read_bytes(context, kv_precision_bytes, batch_size=b, tp=tp)
        bw = float(kv_bandwidth_bytes_per_s)
        kv_time = kv_bytes / bw if bw > 0 else float("inf")
        parts = (
            ("sa", att.sa_time_s),
            ("softmax", att.softmax_time_s),
            ("kv_read", kv_time),
        )
        bound, stage_time = max(parts, key=lambda item: item[1])
        return DecodeS2Timing(
            attention=att,
            kv_read_bytes=kv_bytes,
            kv_read_time_s=kv_time,
            stage_time_s=stage_time,
            bound=bound,
        )

    # --- decode stage tables (DIRECT LAW EVALUATION; the report consumer
    # evaluates these at chosen contexts — never through the discarded
    # per-step decode temp instances) --------------------------------------

    def decode_layer_stage_times(
        self,
        context: int,
        kv_bandwidth_bytes_per_s: float,
        kv_precision_bytes: float,
        batch_size: Optional[int] = None,
        tp: int = 1,
    ) -> "OrderedDict[str, float]":
        """Per-layer DENSE decode stage times at one step's context.

        Weight stages price the analog law at M = B (the B decode
        streams); S2 is the decode law max(T_sa, T_softmax, kv_read/kv_bw).
        """
        b = int(self.params.batch_size if batch_size is None else batch_size)
        t_analog = self.analog_gemm_time(b)
        s2 = self.decode_s2_timing(
            context, kv_bandwidth_bytes_per_s, kv_precision_bytes, batch_size=b, tp=tp
        )
        return OrderedDict(
            (
                ("S1_qkv", t_analog),
                ("S2_attention", s2.stage_time_s),
                ("S3_o_proj", t_analog),
                ("S4_ffn1", t_analog),
                ("S5_ffn2", t_analog),
            )
        )

    def decode_moe_layer_stage_times(
        self,
        context: int,
        kv_bandwidth_bytes_per_s: float,
        kv_precision_bytes: float,
        batch_size: Optional[int] = None,
        tp: int = 1,
    ) -> "OrderedDict[str, float]":
        """Per-layer MoE decode stage times (tokens_owner = B at one step)."""
        b = int(self.params.batch_size if batch_size is None else batch_size)
        t_analog = self.analog_gemm_time(b)
        s2 = self.decode_s2_timing(
            context, kv_bandwidth_bytes_per_s, kv_precision_bytes, batch_size=b, tp=tp
        )
        t_ffn = self.moe_ffn_stage_time(b)
        return OrderedDict(
            (
                ("S1_qkv", t_analog),
                ("S2_attention", s2.stage_time_s),
                ("S3_o_proj", t_analog),
                ("S4_router", t_analog),
                ("S5_ffn1_moe", t_ffn),
                ("S6_ffn2_moe", t_ffn),
            )
        )

    def decode_all_stage_times(
        self,
        context: int,
        kv_bandwidth_bytes_per_s: float,
        kv_precision_bytes: float,
        batch_size: Optional[int] = None,
        tp: int = 1,
    ) -> "OrderedDict[str, float]":
        """Decode stages of every ACTIVE layer class plus the lm_head
        endpoint (M = B), keyed like :meth:`all_stage_times`."""
        mask = self.layer_class_mask()
        num_moe = sum(mask)
        times = OrderedDict()
        if len(mask) - num_moe > 0:
            times.update(
                self.decode_layer_stage_times(
                    context, kv_bandwidth_bytes_per_s, kv_precision_bytes, batch_size, tp
                )
            )
        if num_moe > 0:
            moe_times = self.decode_moe_layer_stage_times(
                context, kv_bandwidth_bytes_per_s, kv_precision_bytes, batch_size, tp
            )
            for stage, t in moe_times.items():
                times[f"moe_{stage}"] = t
        times.update(self.endpoint_stage_times(batch_size=batch_size, decode=True))
        return times

    def decode_pipeline_period(
        self,
        context: int,
        kv_bandwidth_bytes_per_s: float,
        kv_precision_bytes: float,
        batch_size: Optional[int] = None,
        tp: int = 1,
    ) -> Tuple[float, str]:
        """(period_s, bottleneck stage name) for one decode step's context."""
        times = self.decode_all_stage_times(
            context, kv_bandwidth_bytes_per_s, kv_precision_bytes, batch_size, tp
        )
        bottleneck = max(times, key=times.get)
        return times[bottleneck], bottleneck

    def decode_sustained_throughput(
        self,
        step_latency_s: float,
        period_s: float,
        kv_max_streams: int,
        batch_size: Optional[int] = None,
        kv_read_bandwidth_bytes_per_s: Optional[float] = None,
        kv_bytes_per_token: Optional[float] = None,
    ) -> DecodeSustainedThroughput:
        """Sustained aggregate decode throughput: resident wavefronts,
        capped by the fabric ceiling and the KV tier bandwidth.

        The report's aggregate figure ``B / period`` is a FABRIC CEILING:
        it assumes full spatial-pipeline occupancy, with each wavefront one
        batch of B decode streams occupying one stage. Filling the pipeline
        needs

            wavefronts_full = ceil(step_latency / period)

        wavefronts in flight, but the KV capacity at the same context holds
        at most ``kv_max_streams`` streams (:meth:`kv_max_streams`), i.e.

            wavefronts_kv = floor(kv_max_streams / B)   (minimum 0)

        wavefronts. Each resident wavefront completes one token per stream
        every step latency, and no schedule can beat the two physical rate
        ceilings — the bottleneck stage passes at most one wavefront of B
        per period, and the ONE declared KV tier serves every resident
        wavefront's kv reads concurrently — so the sustained aggregate
        rate is

            min(min(wavefronts_full, wavefronts_kv) * B / step_latency,
                B / period,
                kv_bw / kv_bytes_per_token).

        The bandwidth cap applies only when the caller provides BOTH
        ``kv_read_bandwidth_bytes_per_s`` (the tier bandwidth) and
        ``kv_bytes_per_token`` (per-device KV bytes one generated token
        reads across all layers, :meth:`kv_bytes_per_stream` at the same
        context).

        limiting_factor: "fabric" when ``B / period`` binds (the pipeline
        fills), "kv_capacity" when the resident-wavefront count binds,
        "kv_bandwidth" when the tier bandwidth binds below both, and
        "infeasible" (0 tok/s) when the KV capacity cannot hold even one
        wavefront of B streams. ``step_latency_s`` and ``period_s`` are
        the single-stream decode step latency and the decode pipeline
        period evaluated at the SAME context (normally the final one).
        """
        b = int(self.params.batch_size if batch_size is None else batch_size)
        if b <= 0 or float(step_latency_s) <= 0 or float(period_s) <= 0:
            raise ValueError(
                "decode_sustained_throughput needs batch_size, step_latency_s, "
                f"and period_s all > 0 (got B={b}, step_latency_s={step_latency_s}, "
                f"period_s={period_s})."
            )
        w_full = int(math.ceil(float(step_latency_s) / float(period_s)))
        w_kv = max(0, int(kv_max_streams) // b)
        resident = min(w_full, w_kv)
        if w_kv == 0:
            tokens_per_s = 0.0
            limiting = "infeasible"
        else:
            tokens_per_s = resident * b / float(step_latency_s)
            ceiling = b / float(period_s)
            if tokens_per_s >= ceiling:
                # A pipeline with period P sustains at most B / P.
                tokens_per_s = ceiling
                limiting = "fabric"
            elif w_kv < w_full:
                limiting = "kv_capacity"
            else:
                limiting = "fabric"
            if (
                kv_read_bandwidth_bytes_per_s is not None
                and kv_bytes_per_token is not None
                and float(kv_bytes_per_token) > 0
            ):
                bw_cap = float(kv_read_bandwidth_bytes_per_s) / float(kv_bytes_per_token)
                if bw_cap < tokens_per_s:
                    tokens_per_s = bw_cap
                    limiting = "kv_bandwidth"
        return DecodeSustainedThroughput(
            wavefronts_full=w_full,
            wavefronts_kv=w_kv,
            tokens_per_s=tokens_per_s,
            limiting_factor=limiting,
        )

    # ------------------------------------------------------------------
    # Boundary transfers and MoE dispatch/combine (chip-to-chip links)
    # ------------------------------------------------------------------

    def boundary_bytes(self, tokens: int, act_bytes: float) -> float:
        """Activation bytes crossing a chip boundary: tokens * hidden * act."""
        return float(tokens) * float(self.params.hidden_dim) * float(act_bytes)

    @staticmethod
    def p2p_time_s(size_bytes: float, bandwidth_bytes_per_s: float, latency_s: float) -> float:
        """Analytical point-to-point law (mirrors base_timing:381-386)."""
        if size_bytes <= 0:
            return 0.0
        if bandwidth_bytes_per_s == 0:
            return float("inf")
        return float(size_bytes) / float(bandwidth_bytes_per_s) + float(latency_s)

    @property
    def moe_expert_parallel(self) -> int:
        """Chips each MoE layer's routed experts spread over (>= 1)."""
        return max(1, int(getattr(self.chip, "moe_expert_parallel", 1) or 1))

    def moe_dispatch_bytes(self, tokens_owner: int, act_bytes: float) -> float:
        """Balanced A2A sizing, EACH WAY: tokens_owner * top_k * H * act."""
        p = self.params
        return float(tokens_owner) * float(p.top_k) * float(p.hidden_dim) * float(act_bytes)

    def moe_dispatch_time(
        self,
        tokens_owner: int,
        act_bytes: float,
        bandwidth_bytes_per_s: float,
        latency_s: float,
    ) -> float:
        """One-way dispatch (== combine) time on the ep link.

        moe_expert_parallel spreads the routed experts over k chips with
        parallel links, dividing the transfer time by k.
        """
        t = self.p2p_time_s(
            self.moe_dispatch_bytes(tokens_owner, act_bytes),
            bandwidth_bytes_per_s,
            latency_s,
        )
        return t / self.moe_expert_parallel

    def moe_expert_pool(self) -> Tuple[int, int]:
        """(num expert chips, routed arrays per expert chip) for k > 1.

        With moe_expert_parallel k > 1 each MoE layer's routed experts
        leave the layer chip and spread over k dedicated chips; (0, 0)
        when k == 1 or the model has no MoE layers.
        """
        k = self.moe_expert_parallel
        p = self.params
        if k <= 1 or not p.use_moe:
            return (0, 0)
        return (p.num_moe_layers * k, _ceil_div(self.moe_routed_arrays(), k))

    # ------------------------------------------------------------------
    # Area and (partial) energy
    # ------------------------------------------------------------------

    def stack_area_mm2(self) -> float:
        """Transformer-stack analog area (== OPTIMA's recorded "CTT area");
        0 when area reporting is disabled (area_mm2_per_array == 0)."""
        return self.transformer_stack_arrays() * float(self.analog.area_mm2_per_array)

    def total_area_mm2(self) -> float:
        """Stack + endpoint ANALOG area; 0 when area reporting is disabled.

        This is the OPTIMA parity accounting (its recorded "CTT area") and
        counts analog macro silicon only. The shared digital chiplet is a
        separate accounting — see :meth:`system_area_mm2` (D21: one accounting
        per metric, and never two totals under one name).
        """
        return self.total_arrays() * float(self.analog.area_mm2_per_array)

    def shared_digital_area_mm2(self) -> float:
        """Declared silicon of the shared digital chiplet card (D13); 0 by default."""
        return float(self.digital_card.area_mm2)

    def system_area_mm2(self) -> float:
        """Analog macro area plus the shared digital chiplet's declared area."""
        return self.total_area_mm2() + self.shared_digital_area_mm2()

    def layer_stage_energy_pj(
        self, seq_len: Optional[int] = None
    ) -> "OrderedDict[str, float]":
        """Analog energy per layer by stage (pJ); attention has no analog part."""
        s = int(self.params.seq_len if seq_len is None else seq_len)
        return OrderedDict(
            (stage, self.analog_stage_energy_pj(s, n_arrays))
            for stage, n_arrays in self.per_layer_stage_arrays().items()
        )

    def moe_layer_stage_energy_pj(
        self, tokens_owner: Optional[int] = None
    ) -> "OrderedDict[str, float]":
        """Analog energy per MoE layer by stage (pJ).

        Routed FFN energy counts each dispatched token once on its own
        expert's arrays (total vec-evals = tokens_owner * top_k per FFN
        stage regardless of imbalance); shared experts see all owner
        tokens. Attention has no analog part.
        """
        p = self.params
        t_owner = int(p.seq_len if tokens_owner is None else tokens_owner)
        stages = self.moe_layer_stage_arrays()
        ffn1 = self.arrays(p.hidden_dim, p.ffn1_fold * p.moe_intermediate)
        ffn2 = self.arrays(p.moe_intermediate, p.hidden_dim)
        routed_tokens = t_owner * p.top_k
        energy = OrderedDict(
            (
                ("qkv", self.analog_stage_energy_pj(t_owner, stages["qkv"])),
                ("o_proj", self.analog_stage_energy_pj(t_owner, stages["o_proj"])),
                ("router", self.analog_stage_energy_pj(t_owner, stages["router"])),
                # per-expert arrays, tokens summed over experts:
                ("ffn1_routed", self.analog_stage_energy_pj(routed_tokens, ffn1)),
                ("ffn2_routed", self.analog_stage_energy_pj(routed_tokens, ffn2)),
                ("ffn1_shared", self.analog_stage_energy_pj(t_owner, stages["ffn1_shared"])),
                ("ffn2_shared", self.analog_stage_energy_pj(t_owner, stages["ffn2_shared"])),
            )
        )
        return energy

    def endpoint_energy_pj(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> "OrderedDict[str, float]":
        """Analog energy of the model-shaped endpoint stages (pJ; prefill M)."""
        p = self.params
        s = int(p.seq_len if seq_len is None else seq_len)
        b = int(p.batch_size if batch_size is None else batch_size)
        ep_arrays = self.endpoint_arrays()
        energy = OrderedDict()
        if p.patch_dim > 0:
            energy["patch_embed"] = self.analog_stage_energy_pj(s, ep_arrays["patch_embed"])
        if p.num_classes > 0:
            energy["vit_head"] = self.analog_stage_energy_pj(b, ep_arrays["vit_head"])
        if p.lm_head_enabled:
            energy["lm_head"] = self.analog_stage_energy_pj(b * s, ep_arrays["lm_head"])
        return energy

    def kv_dram_energy_pj(self, kv_bytes: float) -> float:
        """cim_dram KV traffic energy: bytes * 8 * energy_per_bit_pj (pJ)."""
        kv_dram = self.cim.kv_dram
        if kv_dram is None:
            return 0.0
        return float(kv_bytes) * 8.0 * float(kv_dram.energy_per_bit_pj)

    def transformer_stack_energy_pj(self, seq_len: Optional[int] = None) -> float:
        """Analog energy of all transformer layers for one forward pass (pJ).

        Mask-aware: dense layers use the dense stage energies, MoE layers
        the MoE stage energies (tokens_owner = seq_len; a B > 1 caller
        folds B into seq_len).
        """
        mask = self.layer_class_mask()
        num_moe = sum(mask)
        num_dense = len(mask) - num_moe
        dense_pj = sum(self.layer_stage_energy_pj(seq_len).values())
        total = num_dense * dense_pj
        if num_moe:
            total += num_moe * sum(self.moe_layer_stage_energy_pj(seq_len).values())
        return total

    # ------------------------------------------------------------------
    # Chip placement and capacity (mapping: cim.chip.layers_per_chip)
    # ------------------------------------------------------------------

    @property
    def layers_per_chip_is_auto(self) -> bool:
        """True when cim.chip.layers_per_chip is the 'auto' derivation."""
        return isinstance(self.chip.layers_per_chip, str)

    def _placed_layer_arrays(self, layer_idx: int) -> int:
        """Arrays layer ``layer_idx`` puts on ITS chip.

        Layer classes come from the MoE mask; with moe_expert_parallel > 1
        the routed-expert arrays leave the layer chip for the expert pool
        (moe_expert_pool) and are not counted here.
        """
        if self.layer_class_mask()[layer_idx]:
            arrays = self.arrays_per_moe_layer()
            if self.moe_expert_parallel > 1:
                arrays -= self.moe_routed_arrays()
            return arrays
        return self.arrays_per_layer()

    def derive_auto_layers_per_chip(self) -> Tuple[int, ...]:
        """Greedy capacity-first packing for cim.chip.layers_per_chip: 'auto'.

        Layer order is preserved; each chip takes layers until the next
        layer would overflow ``arrays_per_chip``. Endpoint arrays are
        pinned: the leading endpoint (patch_embed) occupies chip 0 and the
        trailing endpoints (vit_head / lm_head) must fit on the last chip
        next to its layers. The derived split is an OUTPUT (reported by the
        FWS spatial report).
        """
        capacity = int(self.chip.arrays_per_chip)
        if capacity <= 0:
            raise ValueError(
                "cim.chip.layers_per_chip: 'auto' derives the per-chip layer "
                "split from the array capacity and requires "
                f"cim.chip.arrays_per_chip > 0 (got {capacity})."
            )
        endpoint = self.endpoint_arrays()
        head_arrays = endpoint.get("patch_embed", 0)
        tail_arrays = endpoint.get("vit_head", 0) + endpoint.get("lm_head", 0)
        num_layers = self.params.num_layers
        counts = []
        current_layers = 0
        current_usage = head_arrays  # chip 0 carries the leading endpoint
        for layer_idx in range(num_layers):
            need = self._placed_layer_arrays(layer_idx)
            tail = tail_arrays if layer_idx == num_layers - 1 else 0
            if current_layers > 0 and current_usage + need + tail > capacity:
                counts.append(current_layers)
                current_layers = 0
                current_usage = 0
            if current_usage + need + tail > capacity:
                layer_class = "MoE" if self.layer_class_mask()[layer_idx] else "dense"
                raise ValueError(
                    "cim.chip.layers_per_chip: 'auto' cannot place layer "
                    f"{layer_idx} ({layer_class}): it needs {need} analog arrays"
                    + (f" (+{tail} trailing endpoint arrays)" if tail else "")
                    + (f" (+{current_usage} leading endpoint arrays)" if current_layers == 0 and current_usage else "")
                    + f" on one chip but cim.chip.arrays_per_chip = {capacity}. "
                    "Raise arrays_per_chip (or moe_expert_parallel for MoE layers)."
                )
            current_usage += need
            current_layers += 1
        counts.append(current_layers)
        return tuple(counts)

    def chip_layer_counts(self) -> Tuple[int, ...]:
        """Layers on each chip. An int spec fills chips uniformly (last chip
        takes the remainder); an explicit list must sum to num_layers;
        'auto' derives the split greedily (derive_auto_layers_per_chip)."""
        spec = self.chip.layers_per_chip
        num_layers = self.params.num_layers
        if isinstance(spec, str):
            # config guarantees the only string literal is "auto".
            return self.derive_auto_layers_per_chip()
        if isinstance(spec, tuple):
            if sum(spec) != num_layers:
                raise ValueError(
                    f"cim.chip.layers_per_chip {list(spec)} sums to {sum(spec)} "
                    f"but the model has num_layers = {num_layers}."
                )
            return spec
        per_chip = int(spec)
        num_chips = _ceil_div(num_layers, per_chip)
        counts = [per_chip] * (num_chips - 1)
        counts.append(num_layers - per_chip * (num_chips - 1))
        return tuple(counts)

    def chip_array_usage(self) -> Tuple[int, ...]:
        """Analog arrays needed per LAYER chip.

        Layer classes come from the MoE mask (layer order preserved); with
        moe_expert_parallel > 1 the routed-expert arrays leave the layer
        chip for the expert pool (moe_expert_pool). Endpoint arrays land
        on chip 0 (patch_embed) and the last chip (vit_head / lm_head).
        """
        counts = self.chip_layer_counts()
        endpoint = self.endpoint_arrays()
        usage = []
        layer_idx = 0
        for count in counts:
            chip_total = 0
            for _ in range(count):
                chip_total += self._placed_layer_arrays(layer_idx)
                layer_idx += 1
            usage.append(chip_total)
        usage[0] += endpoint.get("patch_embed", 0)
        usage[-1] += endpoint.get("vit_head", 0) + endpoint.get("lm_head", 0)
        return tuple(usage)

    def validate_capacity(self) -> None:
        """Hard error when a chip needs more arrays than arrays_per_chip > 0.

        Checks the layer chips and, under moe_expert_parallel > 1, the
        dedicated expert-pool chips.
        """
        capacity = int(self.chip.arrays_per_chip)
        if capacity <= 0:
            return
        for chip_idx, needed in enumerate(self.chip_array_usage()):
            if needed > capacity:
                raise ValueError(
                    f"fws_cim capacity overflow: chip {chip_idx} needs {needed} "
                    f"analog arrays but cim.chip.arrays_per_chip = {capacity}. "
                    "Reduce layers_per_chip or raise arrays_per_chip."
                )
        num_expert_chips, arrays_each = self.moe_expert_pool()
        if num_expert_chips > 0 and arrays_each > capacity:
            raise ValueError(
                f"fws_cim capacity overflow: each MoE expert chip needs "
                f"{arrays_each} analog arrays but cim.chip.arrays_per_chip = "
                f"{capacity}. Raise cim.chip.moe_expert_parallel or "
                "arrays_per_chip."
            )

    # ------------------------------------------------------------------
    # Device cards (P2.1, D14 / ADJ-4)
    # ------------------------------------------------------------------

    @property
    def card(self):
        """The active analog-macro card (`cim.analog` promoted, plus knobs)."""
        cards = getattr(self.cim, "cards", None)
        if cards is None:
            import config as _config

            return _config.CIMAnalogCardConfig.synthesized(self.analog)
        return cards.analog_card

    @property
    def digital_card(self):
        """The active shared-digital-chiplet card.

        The SA-attention and softmax-lane laws above ARE this card's laws
        (D13); the card wraps the very ``cim.fabric`` object they read, so
        promoting the block to a card moves no number.
        """
        cards = getattr(self.cim, "cards", None)
        if cards is None:
            import config as _config

            return _config.CIMDigitalChipletCardConfig.synthesized(self.fabric)
        return cards.digital_card

    @property
    def n_slices(self) -> int:
        """Bit slices per weight word: ceil(weight_bits / bits_per_cell) (D11).

        1 unless the card declares ``bits_per_cell < weight_bits`` — slicing
        is per-card opt-in (ADJ-4), so every shipped card returns 1.
        """
        return int(self.card.n_slices)

    def check_card_point(
        self,
        bits_per_cell: Optional[int] = None,
        weight_bits: Optional[int] = None,
        mux: Optional[int] = None,
    ) -> None:
        """Refuse an operating point the card's validity menu does not list.

        A card with an empty menu declares none, so nothing is refused.
        """
        card = self.card
        bpc = card.bits_per_cell if bits_per_cell is None else int(bits_per_cell)
        wbits = card.weight_bits if weight_bits is None else int(weight_bits)
        mux_pt = card.column_sets_per_macro if mux is None else int(mux)
        if not card.admits(bpc, wbits, mux_pt):
            raise CardValidityError(
                f"card '{card.name}' ({card.device}) does not admit "
                f"(bits_per_cell={bpc}, weight_bits={wbits}, mux={mux_pt}). "
                f"Admitted points: {[p.as_tuple() for p in card.validity]}. "
                "The menu is the card's own statement of what the device supports; "
                "the tool refuses an unlisted point instead of interpolating one."
            )

    def macro_footprint_mm2(self) -> float:
        """Package footprint of one macro: silicon area / 3D stack height.

        Stack height divides the FOOTPRINT only — never silicon area, never
        energy.
        """
        return float(self.analog.area_mm2_per_array) / int(self.card.stack_3d_height)

    # ------------------------------------------------------------------
    # Tiles: the allocation unit (P2.2, A3 / D10)
    # ------------------------------------------------------------------

    @property
    def allocation(self):
        """The parsed `cim.allocation` block, or None (dedicated per matrix)."""
        return getattr(self.cim, "allocation", None)

    @property
    def column_sets_per_tile(self) -> int:
        """Mux slots one tile claims: `cim.allocation` wins, else the card's bank.

        The default is the whole macro, which is today's dedicated-per-matrix
        behavior — one owner, every mux slot.
        """
        allocation = self.allocation
        if allocation is not None and allocation.column_sets_per_tile is not None:
            return int(allocation.column_sets_per_tile)
        return int(self.card.allocation_granularity)

    def enumerate_tiles(
        self,
        k: int,
        n: int,
        owner: TileOwner,
        *,
        n_slices: Optional[int] = None,
        column_sets_per_tile: Optional[int] = None,
        first_macro_id: int = 0,
        arrangement: Optional[str] = None,
    ) -> Tuple[Tile, ...]:
        """The array census, promoted: tiles with owners and sites.

        Row blocks are ``ceil(K / rows)``; each row block needs
        ``ceil(N / cols_adc) * n_slices`` column sets, packed ``bank`` sets to
        a tile and ``mux`` sets to a macro. At the defaults (no slicing, whole
        -macro banks) the tile count and the macro count are both exactly
        ``arrays(K, N) = ceil(K/rows) * ceil(N/(cols_adc*mux))``.

        Ordering follows the arrangement: ``column_sets`` walks the slices of
        one output block back to back so their reduction stays local to a
        macro; ``chained_macros`` walks one whole slice before the next, so a
        slice owns its macros and the partials travel (D11).
        """
        card = self.card
        rows = int(card.params.rows)
        cols_adc = int(card.stored_columns_per_set)
        mux = int(card.column_sets_per_macro)
        n_s = self.n_slices if n_slices is None else max(1, int(n_slices))
        bank = self.column_sets_per_tile if column_sets_per_tile is None else int(column_sets_per_tile)
        if bank < 1 or mux % bank != 0:
            raise ValueError(
                f"column_sets_per_tile = {bank} must be >= 1 and divide the card's mux "
                f"= {mux}: the mux slot is the smallest allocatable unit (ADJ-4)."
            )
        arrangement = (arrangement or card.slicing).strip().lower()
        cols_per_tile = cols_adc * bank
        tiles_per_macro = mux // bank
        n_row_blocks = _ceil_div(k, rows)
        n_col_blocks = _ceil_div(n, cols_per_tile)
        if arrangement == "chained_macros":
            order = [
                (rb, cb, s)
                for rb in range(n_row_blocks)
                for s in range(n_s)
                for cb in range(n_col_blocks)
            ]
            # Linear packing: a slice owns its own macros and the partials
            # travel, which is what the arrangement means.
            placement = [(index // tiles_per_macro, index % tiles_per_macro) for index in range(len(order))]
        else:
            order = [
                (rb, cb, s)
                for rb in range(n_row_blocks)
                for cb in range(n_col_blocks)
                for s in range(n_s)
            ]
            # Locality is the whole point of the column-set arrangement (D11),
            # so a slice group is placed as a unit: whole groups per macro,
            # never straddling one. Padding the tail of a macro is the price.
            groups_per_macro = tiles_per_macro // n_s
            if groups_per_macro < 1:
                raise ValueError(
                    f"the column_sets arrangement cannot keep a slice group local: "
                    f"n_slices = {n_s} needs {n_s} of the macro's {tiles_per_macro} tile "
                    f"slots (mux {mux} / column_sets_per_tile {bank}). Use a smaller "
                    "column_sets_per_tile, or the chained_macros arrangement, which "
                    "prices the partial transport instead (D11)."
                )
            placement = []
            for index in range(len(order)):
                group, slice_in_group = divmod(index, n_s)
                macro_offset, group_in_macro = divmod(group, groups_per_macro)
                placement.append((macro_offset, group_in_macro * n_s + slice_in_group))
        tiles = []
        for (rb, cb, s), (macro_offset, slot) in zip(order, placement):
            tiles.append(
                Tile(
                    owner=owner,
                    k_start=rb * rows,
                    k_end=min(int(k), (rb + 1) * rows),
                    n_start=cb * cols_per_tile,
                    n_end=min(int(n), (cb + 1) * cols_per_tile),
                    slice_index=s,
                    site=TileSite(
                        macro_id=first_macro_id + macro_offset,
                        row_start=rb * rows,
                        row_end=min(int(k), (rb + 1) * rows),
                        column_sets=tuple(range(slot * bank, (slot + 1) * bank)),
                    ),
                )
            )
        return tuple(tiles)

    def tile_macro_count(
        self,
        k: int,
        n: int,
        *,
        n_slices: Optional[int] = None,
        column_sets_per_tile: Optional[int] = None,
    ) -> int:
        """Macros a K x N weight matrix occupies once tiled.

        Equals :meth:`arrays` exactly at the defaults (no slicing, whole-macro
        banks) — the degenerate identity the census law becomes.
        """
        tiles = self.enumerate_tiles(
            k,
            n,
            TileOwner(),
            n_slices=n_slices,
            column_sets_per_tile=column_sets_per_tile,
        )
        return len({tile.site.macro_id for tile in tiles})

    @staticmethod
    def macro_occupancy(tiles) -> "OrderedDict[int, Tuple[Tile, ...]]":
        """Resident tiles grouped by macro id, in first-seen order."""
        by_macro: "OrderedDict[int, list]" = OrderedDict()
        for tile in tiles:
            by_macro.setdefault(tile.site.macro_id, []).append(tile)
        return OrderedDict((macro, tuple(items)) for macro, items in by_macro.items())

    def _check_sites(self, pairs) -> None:
        """Capacity check over (owner, site) pairs — the named hard error.

        A macro stores ``mux`` column sets, so range plus uniqueness IS the
        capacity: no set outside 0..mux-1, and no set claimed twice.
        """
        mux = int(self.card.column_sets_per_macro)
        claimed = {}
        for owner, site in pairs:
            for column_set in site.column_sets:
                if column_set < 0 or column_set >= mux:
                    raise MacroCapacityError(
                        f"macro {site.macro_id}: {owner.label} claims column set "
                        f"{column_set}, but the card stores {mux} column sets (mux slots) "
                        "per macro."
                    )
                key = (site.macro_id, column_set)
                if key in claimed:
                    raise MacroCapacityError(
                        f"macro {site.macro_id}: column set {column_set} is claimed by both "
                        f"{claimed[key]} and {owner.label}. One column set holds one tile."
                    )
                claimed[key] = owner.label

    def validate_macro_capacity(self, tiles) -> None:
        """Resident tiles must fit their macros' stored columns (A3).

        Raises :class:`MacroCapacityError` on an out-of-range column set, a
        column set claimed twice, or a macro over its stored-column count.
        """
        self._check_sites((tile.owner, tile.site) for tile in tiles)

    def allocation_sites(self) -> Tuple[Tuple[TileOwner, TileSite], ...]:
        """(owner, site) pairs the user wrote in `cim.allocation.assignments`."""
        allocation = self.allocation
        if allocation is None:
            return ()
        card = self.card
        rows = int(card.params.rows)
        return tuple(
            (
                TileOwner(
                    model=entry.model,
                    layer=entry.layer,
                    op=entry.op,
                    expert=entry.expert,
                    shard=entry.shard,
                ),
                TileSite(
                    macro_id=entry.macro,
                    row_start=0,
                    row_end=rows,
                    column_sets=entry.column_sets,
                ),
            )
            for entry in allocation.assignments
        )

    def validate_allocation(self) -> None:
        """Capacity-check the user's `cim.allocation` block (a no-op when absent)."""
        self._check_sites(self.allocation_sites())

    # ------------------------------------------------------------------
    # Active-column-set pricing (P2.3, ADJ-4)
    # ------------------------------------------------------------------

    def analog_op_time(self, m_tokens: float, active_column_sets: int) -> float:
        """T = M_tokens * (active_column_sets * slice_cycles + switches) / f_analog.

        The op pays for the column sets it ACTIVATES, not the full mux depth.
        One owner filling every mux slot activates ``mux`` sets and pays
        ``mux * slice_cycles`` — the pass-1 charge, exactly.

        ``switches`` is ``(active_column_sets - 1) * card.bank_switch_cycles``.
        Every shipped card declares 0, so no shipped number moves and the
        full-occupancy identity is untouched — but the zero is now a DECLARED
        card figure rather than the silent omission AUDIT finding 3 records.
        """
        # Grouped exactly as vec_cycles / vec_latency_s are, so the
        # full-occupancy case is bit-identical to analog_gemm_time, not
        # merely close: floating-point association is part of the identity.
        cycles = int(active_column_sets) * int(self.analog.slice_cycles)
        switch_cycles = int(self.card.bank_switch_cycles)
        if switch_cycles:
            cycles += max(0, int(active_column_sets) - 1) * switch_cycles
        return float(m_tokens) * (cycles / (float(self.analog.analog_clock_mhz) * 1e6))

    @staticmethod
    def active_column_sets(tiles) -> int:
        """The op's charge: the worst macro's active column sets.

        Every macro holding the op's tiles fires concurrently (the K/N-free
        law), so the op waits on the macro that must convert the most sets.
        """
        by_macro = CimDeviceModel.macro_occupancy(tiles)
        if not by_macro:
            return 0
        return max(
            sum(tile.site.num_column_sets for tile in items) for items in by_macro.values()
        )

    def price_tiled_op(self, m_tokens: float, tiles) -> TiledOpCost:
        """Duration and analog energy of one op over its resident tiles.

        Energy is the pass-1 law read per column set: ``energy_per_vec_pj`` is
        a whole macro's per-vector energy (it already includes the mux), so an
        op that activates a fraction of the stored columns pays that fraction.
        A full-occupancy op pays ``M * E_vec * shots * macros`` — pass 1.
        """
        by_macro = self.macro_occupancy(tiles)
        charge = self.active_column_sets(tiles)
        total_sets = sum(tile.site.num_column_sets for tile in tiles)
        mux = int(self.card.column_sets_per_macro)
        energy = (
            float(m_tokens)
            * float(self.analog.energy_per_vec_pj)
            * int(self.analog.shots_per_output)
            * total_sets
            / mux
        )
        return TiledOpCost(
            time_s=self.analog_op_time(m_tokens, charge),
            active_column_sets=charge,
            macros=len(by_macro),
            total_active_column_sets=total_sets,
            energy_pj=energy,
        )

    # ------------------------------------------------------------------
    # Bit slicing, priced (P2.4, D11)
    # ------------------------------------------------------------------

    def reduction_descriptors(
        self, tiles, *, arrangement: Optional[str] = None
    ) -> Tuple[ReductionOpDescriptor, ...]:
        """Shift-and-add descriptors the DAG builder consumes (plain data).

        One descriptor per slice group — the tiles that hold the slices of one
        output block. Returns () when the card declares no slicing, which is
        the degenerate identity: no slices, no reduction ops, no cost.
        """
        arrangement = (arrangement or self.card.slicing).strip().lower()
        groups: "OrderedDict[object, list]" = OrderedDict()
        for tile in tiles:
            groups.setdefault(tile.group_key, []).append(tile)
        descriptors = []
        for members in groups.values():
            members = sorted(members, key=lambda t: t.slice_index)
            n_s = len({tile.slice_index for tile in members})
            if n_s <= 1:
                continue
            macros = {tile.site.macro_id for tile in members}
            sink = members[0].site.macro_id
            local = len(macros) == 1
            descriptors.append(
                ReductionOpDescriptor(
                    kind="shift_add_tree",
                    arrangement=arrangement,
                    owner=members[0].owner,
                    operand_tiles=tuple(members),
                    output_lanes=members[0].logical_columns,
                    n_slices=n_s,
                    adds_per_output=n_s - 1,
                    depth=(n_s - 1).bit_length(),
                    site_macro_id=sink,
                    local=local,
                    transport_partials=0 if local else n_s - 1,
                )
            )
        return tuple(descriptors)

    def price_reduction(
        self,
        descriptor: ReductionOpDescriptor,
        m_tokens: float,
        *,
        act_bytes: float = 0.0,
        pool: Optional[DigitalPoolSizing] = None,
    ) -> ReductionCost:
        """Cost of one shift-and-add tree: ~(n_s-1) adds per output, depth ~log2(n_s).

        Pipelined, the tree returns one result per lane per cycle after fill:
        ``cycles = depth + ceil(results / lanes) - 1`` (the softmax-lanes
        shape). Energy and area come from the card's pool knobs, which are 0
        until a card declares them — the term then reports zero rather than an
        invented number. Chained partials ride the p2p law: the returned
        ``transport_bytes`` is what the caller hands :meth:`p2p_time_s`.
        """
        pool = self.digital_pool_sizing(n_slices=descriptor.n_slices) if pool is None else pool
        results = float(m_tokens) * int(descriptor.output_lanes)
        adds = results * int(descriptor.adds_per_output)
        cycles = int(descriptor.depth) + _ceil_div(math.ceil(results), pool.lanes) - 1
        return ReductionCost(
            adds=adds,
            cycles=cycles,
            time_s=cycles / pool.pool_clock_hz,
            energy_pj=adds * float(self.card.pool_energy_per_add_pj),
            transport_bytes=results * int(descriptor.transport_partials) * float(act_bytes),
        )

    # ------------------------------------------------------------------
    # Per-macro digital pool sizing (P2.5, D12)
    # ------------------------------------------------------------------

    def macro_result_rate_per_s(self) -> float:
        """Peak ADC results one macro emits: cols_adc * f_analog / slice_cycles."""
        return (
            int(self.card.stored_columns_per_set)
            * float(self.analog.analog_clock_mhz)
            * 1e6
            / int(self.analog.slice_cycles)
        )

    def digital_pool_sizing(
        self, n_slices: Optional[int] = None
    ) -> DigitalPoolSizing:
        """Derive the per-macro pool from the card and its resident tiles.

        Sized never to block: enough lanes to consume the macro's peak result
        rate. The number is REPORTED, never a constraint (D12). Crossing the
        ADJ-4 disclosure share prints a note and changes nothing.
        """
        card = self.card
        n_s = self.n_slices if n_slices is None else max(1, int(n_slices))
        pool_clock_hz = (
            float(card.pool_clock_ghz) * 1e9 if card.pool_clock_ghz > 0 else self.f_fabric_hz
        )
        rate = self.macro_result_rate_per_s()
        lanes = max(1, math.ceil(rate / pool_clock_hz))
        adders = lanes * (n_s - 1)
        area = adders * float(card.pool_area_mm2_per_adder)
        footprint = self.macro_footprint_mm2()
        share = area / footprint if footprint > 0 else 0.0
        disclose = share >= POOL_DISCLOSURE_AREA_SHARE
        note = None
        if disclose:
            note = (
                f"per-macro digital pool: derived area {area:.6g} mm2 is "
                f"{share * 100:.1f}% of the {footprint:.6g} mm2 macro footprint it serves "
                f"(disclosure threshold {POOL_DISCLOSURE_AREA_SHARE * 100:.0f}%). "
                "The pool is still absorbed and still costs no time here (D12); this note "
                "is the disclosure, not a constraint."
            )
        return DigitalPoolSizing(
            result_rate_per_s=rate,
            pool_clock_hz=pool_clock_hz,
            lanes=lanes,
            adders=adders,
            area_mm2=area,
            macro_footprint_mm2=footprint,
            area_share=share,
            disclose=disclose,
            note=note,
        )

    def report_digital_pool(self, sizing: Optional[DigitalPoolSizing] = None) -> str:
        """The one-line derived pool report D12 requires (P2.5).

        D12 says the pool is SIZED so it never blocks the pipeline and the
        derived sizing is REPORTED. It is reported on stdout and nowhere else:
        the closed-form report file is frozen (ADJ-8), and the pool constrains
        no number in it, so a report FIELD would claim an accounting the
        legacy path does not have.
        """
        sizing = self.digital_pool_sizing() if sizing is None else sizing
        line = (
            f"[FWS-CIM] per-macro digital pool (derived, D12): {sizing.lanes} shift-add "
            f"lanes, {sizing.adders} adders at {sizing.pool_clock_hz / 1e9:.4g} GHz, "
            f"consuming {sizing.result_rate_per_s:.6g} ADC results/s; area "
            f"{sizing.area_mm2:.6g} mm2 = {sizing.area_share * 100:.1f}% of the "
            f"{sizing.macro_footprint_mm2:.6g} mm2 macro footprint it serves. "
            "The pool constrains nothing (D12); this is the derived sizing, reported."
        )
        digital_area = self.shared_digital_area_mm2()
        if digital_area > 0:
            line += (
                f" Shared digital chiplet card '{self.digital_card.name}' declares "
                f"{digital_area:.6g} mm2, counted in system_area_mm2 and never in the "
                "analog area total."
            )
        return line

    def disclose_digital_pool(self, sizing: Optional[DigitalPoolSizing] = None) -> Optional[str]:
        """Print the pool note when the ADJ-4 share is crossed; return it either way."""
        sizing = self.digital_pool_sizing() if sizing is None else sizing
        if sizing.disclose:
            print(f"[NOTE]: {sizing.note}")
        return sizing.note
