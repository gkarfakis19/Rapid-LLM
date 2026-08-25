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
    - ARRAY GROUPS AND FOLD CONCURRENCY (ADJ-10). ``num_arrays`` is now a
      COUNT OF CONCURRENT FOLDS, not a flag. The fabric's arrays are
      partitioned into a QK group of ``a_qk`` and a PV group of ``a_pv``
      (``a_qk + a_pv = num_arrays``, each at least 1); the
      ``folds = heads_per_replica * streams`` back-to-back runs of a call
      are dealt round-robin across that group, so an array in the QK group
      carries ``ceil(folds / a_qk)`` of them and the folded contraction dim
      it sees is ``k * ceil(folds / a_qk)`` instead of ``k * folds``:

        ``QK = sa_cycles(m, n, k * ceil(folds / a_qk))``
        ``PV = sa_cycles(m, k, n * ceil(folds / a_pv))``

      At ``num_arrays = 2`` the split is (1, 1) and both expressions
      reproduce the pass-1 law BIT FOR BIT, which is why every OPTIMA parity
      configuration is untouched by this generalization. The partition is
      chosen per call by :func:`fold_group_split` (see its docstring for the
      objective and the tie-break). CONCURRENCY SATURATES AT ``folds``: a
      group wider than the fold count leaves arrays with nothing to carry,
      so ``num_arrays = 2 * folds`` is the width past which no further copy
      of the measured block shortens the call, and the residue
      ``sa_cycles(m, n, k)`` is the DECLARED ``rows x cols`` geometry's own
      floor. WHAT IS NOT MODELLED, BY NAME: splitting ONE fold's N across
      arrays (the ``ceil(n / cols)`` column passes) would keep shrinking the
      call past that floor, and it is not modelled because it is a claim
      about the fabric's dataflow — how the K matrix is broadcast and how
      the column tiles are merged — that no recorded ScaleSim reference in
      this repo covers. Inventing it would be inventing geometry (ADJ-4), so
      the floor stands and ADJ-10's derivation NAMES it instead.

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

Digital op laws (QIF P2.6) — the shared chiplet's vector/scan engine
    The chiplet card gains ENGINE CAPABILITY knobs (``vector_lanes``,
    ``vector_clock_ghz``, ``vector_pipeline_depth``, ``state_bytes_per_cycle``)
    and the laws for the op kinds P1 admitted are timed on them:
    ``cycles = pipeline_depth + ceil(ops / lanes) - 1``. One lane retires ONE
    scalar operation per cycle, so no law can exceed ``lanes * clock`` ops/s.
    ``vector_lanes`` has NO default and refuses by name
    (:class:`EngineCapabilityError`) — ADJ-4, no invented numbers.

    - Mamba-2 SSD: :func:`ssm_recurrent_scan_work` (per-token recurrence, work
      counts ported term-for-term from OPTIMA ``stage_m3_scan``) and
      :func:`ssd_chunked_scan_work` (chunked; UNVALIDATED, OPTIMA has no
      chunked reference — it prices prefill and decode alike).
    - Mamba-1 selective scan: the unchunked subset, same law, one gate term.
    - Delta rule / DeltaNet / RWKV-7: :func:`delta_rule_work`. UNVALIDATED.
      RG-LRU (:func:`rg_lru_work`) is its strict subset at ``d_k = 1``.
    - Short depthwise conv: NOT a chiplet op. ADJ-3 puts it on the per-macro
      pool, absorbed into :meth:`CimDeviceModel.digital_pool_sizing`.
    - Sliding-window attention: the SA law at ``n = min(context, window)`` —
      a thin wrapper, never a second attention accounting.
    - MLA (D6): the SA law at the MLA call dims with ``kv_heads = 1`` (the
      latent is one shared rank, so tp buys nothing), plus
      :meth:`CimDeviceModel.mla_kv_replication`, the replicated-weight area.

    STANCE: OPTIMA SIZES an engine to a reuse target and its digital load can
    only cost area, never time (AUDIT finding 6). We TIME declared work on a
    declared engine. Only the work-count arithmetic is ported, never sizing.

OPTIMA block-latency equivalence (recon note for validation)
    The recorded OPTIMA runs sum a sixth pipeline stage equal to exactly one
    analog stage time (their trailing peripherals stage), so OPTIMA's
    ``block_latency`` = :meth:`CimDeviceModel.block_latency` (S1..S5) plus
    ``analog_gemm_time(seq)``. Rapid-LLM lists endpoint stages separately
    instead; validation against the recorded numbers must add that one analog
    stage time.
"""

import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import yaml

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
    #: Shared-expert FFN width (`model_param.ffn_dims.shared_expert`); 0 falls
    #: back to the routed-expert width. llm_util's parameter census already
    #: reads this field, so the array census must read the same one or the two
    #: accountings of one model disagree (D21).
    shared_intermediate_size: int = 0
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
            shared_intermediate_size=int(
                dict(getattr(model, "ffn_dims", {}) or {}).get("shared_expert", 0) or 0
            ),
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
    def shared_expert_intermediate(self) -> int:
        """Shared-expert FFN width; the routed width when the model declares none.

        Granite-4.0-H-Tiny is the case that needs it: 64 routed experts at 512
        beside ONE always-on shared MLP at 1024. Falling back to the routed
        width is the degenerate identity every existing MoE config takes, so
        those array counts do not move.
        """
        return (
            self.shared_intermediate_size
            if self.shared_intermediate_size > 0
            else self.moe_intermediate
        )

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
    #: ADJ-10's fold concurrency, as it was actually spent on this call.
    #: ``folds`` = heads_per_replica * streams (the back-to-back runs folded
    #: into K); ``qk_arrays``/``pv_arrays`` are the two array groups the
    #: fabric's ``num_arrays`` was partitioned into, and
    #: ``qk_folds_per_array``/``pv_folds_per_array`` are the ceil-divisions
    #: that multiply the contraction dim. All five default to the pass-1
    #: values, so a hand-built AttentionTiming still reads as the old law.
    folds: int = 1
    qk_arrays: int = 1
    pv_arrays: int = 1
    qk_folds_per_array: int = 1
    pv_folds_per_array: int = 1
    #: The softmax width the call was priced on: ``softmax_lanes * replicas``,
    #: with ``softmax_lanes`` DERIVED under ADJ-10 when a derivation is
    #: installed and declared otherwise.
    softmax_width: int = 1
    #: The CALL DIMS this timing was taken at, carried so ADJ-10's derivation
    #: can re-price the same call at a candidate width. A duration cannot be
    #: re-priced; dims can.
    call_m: int = 0
    call_k: int = 0
    call_n: int = 0
    softmax_tokens: int = 0


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
    """Where a tile sits: macro id, row range, column-set (mux-slot) ids.

    **Row convention, pinned by the mapping consumer (QIF P3.2).** A site's
    ``[row_start, row_end)`` is the half-open K range of the OWNER'S WEIGHT
    MATRIX that this site holds — the range :meth:`CimDeviceModel.enumerate_tiles`
    emits, so ``site.row_start == tile.k_start`` and ``site.row_end ==
    tile.k_end``, and the span never exceeds the card's rows.
    ``fws_mapping.validate_tile_row_convention`` enforces it on every tile the
    mapper places.

    One producer does NOT carry a K range and cannot:
    :meth:`CimDeviceModel.allocation_sites` reads `cim.allocation` entries,
    which declare a macro and a set of mux slots and nothing about rows, so it
    fills ``(0, rows)`` as a placeholder. That output feeds
    :meth:`CimDeviceModel._check_sites`, which reads column sets ONLY and never
    looks at the row range. The mapper therefore never consumes it as one: a
    user allocation reaches a tile through the enumerator, which supplies the
    rows, and the config supplies only the macro and the column sets.
    """

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
    #: --- short depthwise conv absorbed into the pool (ADJ-3, P2.6 4) ------
    #: 0 everywhere means no conv was declared and every figure below is 0,
    #: which reproduces the P2.5 sizing exactly.
    conv_kernel: int = 0
    conv_channels: int = 0
    conv_ops_per_result: int = 0
    conv_column_duty: float = 0.0
    conv_ops_per_s: float = 0.0
    conv_lanes: int = 0
    conv_area_mm2: float = 0.0

    @property
    def total_lanes(self) -> int:
        """Every lane the pool carries: shift-add lanes plus conv lanes."""
        return int(self.lanes) + int(self.conv_lanes)


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


# ---------------------------------------------------------------------------
# Dense packing (QIF P7.2): Invariant W in code
#
# INVARIANT W (D27, George 2026-08-24, HARD): weight space is NEVER wasted.
# Every bank of every macro holds real weights — a "spatial" mapping does not
# idle banks, it fills them with other projections, layers or tensors. Waste
# therefore exists ONLY at dimension-mismatch remainders, and it is a REPORTED
# first-class metric rather than a rounding nobody sees.
#
# The consequence is the GLOBAL CELL FLOOR: analog silicon is determined by the
# model, not chosen by the mapper. ``cell_floor_macros`` is that floor,
# ``macro_count`` is what the packing actually reached, and the difference is
# the waste, itemized — never a single number that hides which of the two
# mismatches produced it.
#
# The bank, not the macro, is the packing unit: a bank is
# ``column_sets_per_tile`` mux slots (the card's declared allocation
# granularity, ADJ-4), so a card that admits only whole-macro allocation packs
# whole macros and a card that admits single mux slots packs single slots. The
# packer never allocates finer than the card admits.
# ---------------------------------------------------------------------------

#: The ONE walk the dense packer admits (D26). K-INNER means the K blocks of
#: one output block are CONSECUTIVE in the bank stream, so a tensor's K stack
#: lands in one macro whenever it fits, and — because the stack's banks fire
#: back to back — one accumulator per macro is live at a time. Every other walk
#: is on the DOA register: it buys nothing and costs a partial-sum transport.
CANONICAL_WALK = "k_inner"

#: The DOA register, in code (D26). "Machinery for a dead fold is a conformance
#: violation, not initiative", so these names exist HERE, as refusals, and
#: nowhere else in the tree as capability.
DEAD_FOLDS: "OrderedDict[str, str]" = OrderedDict(
    (
        (
            "non_canonical_walk",
            "P7 admits the K-INNER walk only (cim_timing.CANONICAL_WALK). An N-inner "
            "or slice-major walk scatters one output block's K partials across macros, "
            "which buys no weight space and costs a partial-sum transport per block. "
            "It is on the DOA register (D26) and no code path implements it.",
        ),
        (
            "slicing_x_folding",
            "Bit slicing x folding is on the DOA register (D26). Slicing already owns a "
            "placement law (P2.4: a slice group is local to one macro, or the chained "
            "arrangement prices the partials), and folding the same tensor again on top "
            "of it composes two placement laws whose interaction nothing validates. A "
            "card that slices packs through enumerate_tiles, not through dense_pack.",
        ),
        (
            "replication",
            "Replication is on the DOA register (D26) and contradicts Invariant W "
            "(D27): a second copy of a weight spends cells that hold no new weight, "
            "which is the one thing D27 forbids. D27 alone is the reason it is dead — "
            "the older serving argument (\"it buys throughput only in a regime D15 does "
            "not serve\") no longer applies, because D29 supersedes D15 for fws_cim and "
            "makes cross-STAGE bank sharing cost time every beat, which is exactly the "
            "case replication could relieve. The refusal stands on the weight-space "
            "invariant, not on the serving regime.",
        ),
        (
            "prefill_folding",
            "DECODE ONLY (D25), and D25 alone is the reason: P7 prices no prefill op at "
            "all, so a prefill mapping has nothing to pack. The original rationale "
            "leaned on P7's Fact 3 (decode sequentiality makes co-residency free), and "
            "that paragraph is RETIRED — under D29 the pipeline is always full and "
            "cross-stage sharing contends every beat. The refusal is unchanged; its "
            "reason is now the decode-only scope and nothing else.",
        ),
    )
)


class DeadFoldError(ValueError):
    """A DOA-register fold was asked for by name (D26).

    The register is BINDING: the refusal is the entire implementation, and the
    error names which entry was hit and why it is dead.
    """


def refuse_dead_fold(name: str, *, context: str = "") -> None:
    """Raise :class:`DeadFoldError` for a registered dead fold (D26).

    ``name`` must be a key of :data:`DEAD_FOLDS`; asking to refuse something
    that is not on the register is itself an error, because the register is the
    only place a fold may be declared dead.
    """
    if name not in DEAD_FOLDS:
        raise ValueError(
            f"{name!r} is not on the DOA register (D26). The register is "
            f"cim_timing.DEAD_FOLDS = {tuple(DEAD_FOLDS)}; a fold is dead there or it "
            "is not dead."
        )
    where = f"{context}: " if context else ""
    raise DeadFoldError(f"{where}{name}. {DEAD_FOLDS[name]}")


@dataclass(frozen=True)
class PackRequest:
    """One weight matrix offered to the dense packer: an owner and a (K, N).

    ``group`` is a LOCALITY hint and nothing else — the packer keeps a group's
    tensors adjacent in the bank stream (a layer's tensors stay together, which
    is what keeps them on one chip), and it never reorders across groups.
    """

    owner: TileOwner
    k: int
    n: int
    group: str = ""


@dataclass(frozen=True)
class MacroFill:
    """What one macro actually holds after dense packing.

    ``real_cells`` is weights; ``committed_cells`` is every cell the macro
    has. Invariant W is the statement that the two are equal up to the
    dimension-mismatch remainder, which is exactly ``committed - real``.
    """

    macro_id: int
    banks_used: int
    banks_total: int
    real_cells: int
    committed_cells: int
    owners: Tuple[TileOwner, ...]

    @property
    def occupancy(self) -> float:
        """Real cells over the macro's cells — 1.0 is a perfectly packed macro."""
        if self.committed_cells <= 0:
            return 0.0
        return self.real_cells / self.committed_cells


@dataclass(frozen=True)
class KStack:
    """The K blocks of ONE output block of one tensor, in walk order.

    ``depth`` is how many K blocks reduce into the block: ``depth == 1`` needs
    no accumulator at all. ``local`` is the whole point of the K-inner walk —
    every block in one macro, so the partials never leave it.
    """

    owner: TileOwner
    n_start: int
    n_end: int
    depth: int
    macro_ids: Tuple[int, ...]
    sink_macro_id: int

    @property
    def local(self) -> bool:
        return len(set(self.macro_ids)) == 1

    @property
    def width(self) -> int:
        """Logical output columns this stack produces (the accumulator's lanes)."""
        return int(self.n_end) - int(self.n_start)


@dataclass(frozen=True)
class DensePacking:
    """A dense packing and its accounting (Invariant W, D27).

    ONE waste accounting, itemized into the two mismatches that produce it:

      * ``remainder_cells`` — cells inside a committed bank that the tensor's
        own dimensions do not reach (a K or N remainder).
      * ``tail_cells`` — banks of the last macro that the stream did not fill.

    ``waste_cells = remainder_cells + tail_cells`` and
    ``waste_pct = 100 * waste_cells / committed_cells``. There is no third
    number and no rounding anywhere: ``real + remainder + tail == committed``.
    """

    tiles: Tuple[Tile, ...]
    macro_fills: Tuple[MacroFill, ...]
    k_stacks: Tuple[KStack, ...]
    walk: str
    first_macro_id: int
    macro_count: int
    banks_used: int
    banks_per_macro: int
    column_sets_per_bank: int
    cells_per_bank: int
    cells_per_macro: int
    real_cells: int
    block_cells: int
    committed_cells: int
    remainder_cells: int
    tail_cells: int
    cell_floor_macros: int
    disclosures: Tuple[str, ...] = ()

    @property
    def waste_cells(self) -> int:
        return int(self.remainder_cells) + int(self.tail_cells)

    @property
    def waste_pct(self) -> float:
        """The reported metric (D27). 0.0 when nothing is committed."""
        if self.committed_cells <= 0:
            return 0.0
        return 100.0 * self.waste_cells / self.committed_cells

    @property
    def floor_delta_macros(self) -> int:
        """Macros this packing costs ABOVE the global cell floor.

        Never negative: the floor is a lower bound the packing cannot beat.
        """
        return int(self.macro_count) - int(self.cell_floor_macros)

    @property
    def local_k_stacks(self) -> Tuple[KStack, ...]:
        return tuple(stack for stack in self.k_stacks if stack.depth > 1 and stack.local)

    @property
    def spread_k_stacks(self) -> Tuple[KStack, ...]:
        return tuple(stack for stack in self.k_stacks if stack.depth > 1 and not stack.local)

    def summary(self) -> "OrderedDict[str, object]":
        """The packing's numbers, flat, for a report or an atlas."""
        return OrderedDict(
            (
                ("walk", self.walk),
                ("macros", int(self.macro_count)),
                ("cell_floor_macros", int(self.cell_floor_macros)),
                ("floor_delta_macros", int(self.floor_delta_macros)),
                ("banks_used", int(self.banks_used)),
                ("banks_per_macro", int(self.banks_per_macro)),
                ("real_cells", int(self.real_cells)),
                ("committed_cells", int(self.committed_cells)),
                ("remainder_cells", int(self.remainder_cells)),
                ("tail_cells", int(self.tail_cells)),
                ("waste_cells", int(self.waste_cells)),
                ("waste_pct", float(self.waste_pct)),
                ("k_stacks", len(self.k_stacks)),
                ("local_k_stacks", len(self.local_k_stacks)),
                ("spread_k_stacks", len(self.spread_k_stacks)),
            )
        )


@dataclass(frozen=True)
class AccumulatorCost:
    """What ONE K stack's accumulation costs beyond the analog pass walk (P7.2).

    THE LAW. A K stack of depth ``d`` fires ``d`` banks back to back (K-inner),
    each pass emitting ``m * width`` results into one accumulator. Add ``j``
    retires while pass ``j + 1`` is converting, so adds 1..d-2 are HIDDEN by
    construction; only the LAST add has no pass behind it to hide under, and it
    is the tail this law charges. A stall appears only if the pool cannot drain
    a pass before the next one lands — which the P2.5 pool sizing forbids by
    construction (``lanes * pool_clock >= cols_adc * f_analog / slice_cycles``),
    so ``stall_cycles`` is 0 for every shipped card and the zero is DERIVED,
    not assumed.

    The accumulation is modelled BATCH-PER-PASS — a pass's results are added as
    one vector through the pool's lanes — which is the same stance
    :meth:`price_reduction` takes for the shift-add tree, so one shape has one
    modelling stance rather than two. It is not a safety margin (D28): a
    per-result streaming accumulator would retire the tail in one add latency
    instead of one drain, and the difference is stated here rather than padded
    into a number.

    ONE ACCOUNTING (D21). A SPREAD stack (its K blocks on different macros) is
    already priced end to end by the row-block partial-sum law — P3 emits the
    transfer and the pool rowsum, P4 prices them with
    :meth:`CimDeviceModel.price_reduction`. This law therefore charges a spread
    stack NOTHING and reports ``priced_here = False`` plus the transport bytes
    for reconciliation. The new content here is the LOCAL stack, which the DAG
    absorbs into the analog op at zero today.
    """

    stack: KStack
    depth: int
    width: int
    lanes: int
    partial_adds: float
    hidden_adds: float
    drain_cycles: int
    stall_s: float
    time_s: float
    energy_pj: float
    transport_partials: int
    transport_bytes: float
    priced_here: bool
    law: str
    disclosures: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PackingAccumulatorCost:
    """Every K stack of a packing, charged the way the machine pays for them.

    TIME is the WORST MACRO's accumulator tail, exactly as
    :meth:`CimDeviceModel.active_column_sets` charges the worst macro's passes:
    macros fire concurrently, so an op waits on the slowest one, and the tails
    of stacks that share a macro add up because that macro walks them in
    sequence.

    AREA counts ONE accumulator per macro that hosts a local stack, not one per
    stack: under the K-inner walk a stack's banks are consecutive, so its
    partial is consumed before the next stack's first pass and exactly one
    accumulator is live per macro at a time. That is what the canonical walk
    buys, stated as the number it changes.
    """

    stacks: Tuple[AccumulatorCost, ...]
    time_s: float
    energy_pj: float
    area_mm2: float
    accumulators: int
    adders_per_accumulator: int
    transport_bytes: float
    local_stacks: int
    spread_stacks: int
    disclosures: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Digital op laws (QIF P2.6): the shared digital chiplet's VECTOR/SCAN engine
#
# STANCE (the inversion this whole section exists to state): OPTIMA SIZES an
# engine to hit a reuse target — `create_s3_scan_collection` derives unit
# counts from a microcycle budget and `get_execution_cycles` then returns that
# budget, ignoring its own arguments. Digital load can only cost area there,
# never time (AUDIT finding 6). WE do the opposite: a law DECLARES the work of
# a call and TIMES it on a DECLARED engine — `cycles(work; lanes) / clock`. So
# the ported content is the WORK-COUNT arithmetic (state dims, groups, heads,
# chunking), never the sizing logic.
#
# One lane retires ONE scalar arithmetic operation (a multiply OR an add) per
# vector cycle. Every law is therefore bounded by `lanes * clock` ops/s BY
# CONSTRUCTION (see :meth:`CimDeviceModel.price_vector_work`); the arithmetic
# -peak test that pins this is the DESIGN2 section-8 erratum precedent.
# ---------------------------------------------------------------------------


class EngineCapabilityError(ValueError):
    """The digital chiplet card declares no engine the requested law needs.

    ADJ-4, no invented numbers: the systolic array is a MATMUL engine and the
    softmax lanes are a softmax pipeline, so neither can stand in for a scan /
    vector engine. A card that does not declare ``vector_lanes`` gets this
    refusal instead of a number nobody can defend.
    """


#: Law provenance labels (D21: what is a reference and what is not).
LAW_OPTIMA_M3 = "optima_m3_reference"
"""Work counts reproduce OPTIMA `stages_mamba.stage_m3_scan` term for term."""
LAW_OPTIMA_M3_SUBSET = "optima_m3_subset"
"""The same term structure with a strictly smaller declared work set."""
LAW_UNVALIDATED = "unvalidated"
"""Derived from first principles. NO numeric reference exists anywhere."""


@dataclass(frozen=True)
class VectorWork:
    """The DECLARED scalar arithmetic of one digital op call, itemized.

    ``terms`` is ``(name, muls, adds)`` per algorithm step, so every operation
    charged has exactly one named source (D21: one accounting per metric).
    ``state_bytes`` is the recurrent-state traffic (reads + writes) the call
    moves — reported always, priced only when the card declares a state
    bandwidth.
    """

    law: str
    validated: str
    terms: Tuple[Tuple[str, float, float], ...]
    state_bytes: float
    detail: Tuple[Tuple[str, float], ...] = ()

    @property
    def muls(self) -> float:
        return math.fsum(term[1] for term in self.terms)

    @property
    def adds(self) -> float:
        return math.fsum(term[2] for term in self.terms)

    @property
    def ops(self) -> float:
        """Scalar operations charged: muls + adds (one lane-cycle each)."""
        return self.muls + self.adds

    def term(self, name: str) -> Tuple[float, float]:
        """(muls, adds) of one named term; KeyError-free lookup is not wanted."""
        for term_name, muls, adds in self.terms:
            if term_name == name:
                return (muls, adds)
        raise KeyError(f"{self.law}: no work term named {name!r}")


@dataclass(frozen=True)
class DigitalOpCost:
    """One digital op priced on the declared vector engine.

    ``time_s`` is ``max(arith_time_s, state_time_s)`` — the engine cannot
    retire arithmetic faster than its lanes, nor state faster than its
    declared state bandwidth. ``state_time_s`` is 0.0 when the card declares
    no ``state_bytes_per_cycle``; that relaxation is named in ``disclosures``,
    never silent.
    """

    law: str
    validated: str
    work: VectorWork
    lanes: int
    clock_hz: float
    pipeline_depth: int
    arith_cycles: int
    arith_time_s: float
    state_time_s: float
    time_s: float
    ops_per_s: float
    disclosures: Tuple[str, ...]


# --- Mamba2 SSD / Mamba1 selective scan ------------------------------------


def ssm_recurrent_scan_work(
    tokens: float,
    *,
    d_inner: int,
    d_state: int,
    n_groups: int = 1,
    n_heads: int = 0,
    variant: str = "mamba2",
    act_bytes: float = 1.0,
) -> VectorWork:
    """Per-token recurrent SSM state update — the UNCHUNKED law (P2.6 1+2).

    This is one law with two gate spellings, because Mamba-1 selective scan IS
    the unchunked subset of Mamba-2 SSD:

    ======================  ==========================================
    term                    work per token (over the whole mixer)
    ======================  ==========================================
    ``state_decay``         ``d_inner * d_state`` muls  (S *= exp(dt A))
    ``input_gate``          mamba2: ``n_heads * d_state`` muls (dt_h * B_g,
                            one dt per head against the group's B);
                            mamba1: ``d_inner`` muls (dt_i * x_i, no head
                            structure — B is not per-channel)
    ``outer_write``         ``d_inner * d_state`` muls  (rank-1 (dtB) (x) x)
    ``state_combine``       ``d_inner * d_state`` adds
    ``readout_mul``         ``d_inner * d_state`` muls  (y = S . C)
    ``readout_reduce``      ``d_inner * d_state`` adds
    ======================  ==========================================

    OPTIMA EQUIVALENCE (D21 — this is the whole reference, stated exactly).
    ``create_s3_scan_collection`` in naive (``dot_prod_trick=False``) mode
    censuses, per group per MICROCYCLE, ``decay_muls = outer_muls =
    combine_adds = readout_muls = readout_adds = L' * U`` and ``beta_muls =
    g_h * U``, with ``L = d_inner/G``, ``g_h = n_heads/G``, ``L' =
    ceil(L/T_L)``, ``U = ceil(N/T_N)`` and ``T_L * T_N ~ reuse`` microcycles
    per token. Multiplying its per-microcycle census by its own ``reuse``:

    * the decay / outer / combine / readout terms equal OURS times
      ``(L'*T_L/L) * (U*T_N/N)`` — pure TILING CEIL, 1.0 when the tiles
      divide. This is a term-for-term port. At the one RECORDED point
      (``compiler_mamba_giant_22nm.md``, mux 16, digital clock 0.8 GHz read
      off ``configs/digital_tile_config.yaml``) the tiles do divide: reuse
      256, T_L 32, T_N 8, L' 16, U 16, so the ceil is EXACTLY 1.0 and the
      port carries no residual at all.
    * ``beta_muls * reuse = g_h * N * T_L`` — ``T_L`` times ours, because
      OPTIMA recomputes ``dt * B`` inside every channel tile. That is an
      artifact of ITS tiling of a SIZED engine, not per-token work, so we do
      not charge it. Named here rather than quietly dropped.
    * OPTIMA additionally instantiates ``y_acc_units = L' * U`` accumulators,
      which its OWN ``adds_per_group_mc`` does not count. It is a hardware
      census term, not a work term; we do not charge it either.

    The trick (``dot_prod_trick=True``, OPTIMA's default) replaces the
    streaming readout with projected-state math. It is a different ALGORITHM,
    not a different accounting of this one, so it is not ported: this law is
    the streaming form, and the streaming form is what the terms say.
    """
    variant = str(variant or "mamba2").strip().lower()
    if variant not in ("mamba1", "mamba2"):
        raise ValueError(
            f"ssm_recurrent_scan_work: variant must be 'mamba1' or 'mamba2' (got {variant!r})"
        )
    tokens = float(tokens)
    d_inner = int(d_inner)
    d_state = int(d_state)
    n_groups = int(n_groups)
    n_heads = int(n_heads)
    if tokens < 0:
        raise ValueError("ssm_recurrent_scan_work: tokens must be >= 0")
    if d_inner < 1 or d_state < 1:
        raise ValueError("ssm_recurrent_scan_work: d_inner and d_state must be >= 1")
    if n_groups < 1 or d_inner % n_groups != 0:
        raise ValueError(
            f"ssm_recurrent_scan_work: n_groups = {n_groups} must be >= 1 and divide "
            f"d_inner = {d_inner} (OPTIMA asserts the same: a group owns whole channels)."
        )
    if variant == "mamba2":
        if n_heads < 1 or n_heads % n_groups != 0:
            raise ValueError(
                f"ssm_recurrent_scan_work: mamba2 needs n_heads >= 1 divisible by "
                f"n_groups = {n_groups} (got n_heads = {n_heads})."
            )
        gate_muls = tokens * n_heads * d_state
        gate_name = "input_gate"
    else:
        if n_heads:
            raise ValueError(
                "ssm_recurrent_scan_work: mamba1 selective scan has no head structure; "
                f"leave n_heads unset (got {n_heads})."
            )
        gate_muls = tokens * d_inner
        gate_name = "input_gate"
    state = tokens * d_inner * d_state
    return VectorWork(
        law=f"ssm_recurrent_scan[{variant}]",
        validated=LAW_OPTIMA_M3 if variant == "mamba2" else LAW_OPTIMA_M3_SUBSET,
        terms=(
            ("state_decay", state, 0.0),
            (gate_name, gate_muls, 0.0),
            ("outer_write", state, 0.0),
            ("state_combine", 0.0, state),
            ("readout_mul", state, 0.0),
            ("readout_reduce", 0.0, state),
        ),
        state_bytes=tokens * 2.0 * d_inner * d_state * float(act_bytes),
        detail=(
            ("tokens", tokens),
            ("state_elements", float(d_inner * d_state)),
            ("channels_per_group", float(d_inner // n_groups)),
            ("heads_per_group", float(n_heads // n_groups) if n_groups and n_heads else 0.0),
        ),
    )


def ssd_chunked_scan_work(
    tokens: float,
    *,
    d_inner: int,
    d_state: int,
    n_groups: int,
    n_heads: int,
    chunk_size: int,
    act_bytes: float = 1.0,
) -> VectorWork:
    """Mamba-2 SSD in its CHUNKED form (P2.6 1) — UNVALIDATED, no reference.

    Honesty first: OPTIMA has NO chunked reference. Its Mamba model prices
    prefill and decode with the same per-token recurrent stage time (RECON
    section 1), so there is nothing to port here and nothing to check against.
    Every term below is derived from the SSD chunk decomposition itself.

    Per chunk of ``Q = chunk_size`` tokens, per group (``L = d_inner/G``,
    ``N = d_state``, ``g_h = n_heads/G``):

    ==========================  ==================================  ==========
    term                        muls                                adds
    ==========================  ==================================  ==========
    ``decay_cumprod``           ``g_h * Q``                         --
    ``intra_scores``            ``Q*Q*N``   (C_chunk B_chunk^T)     ``Q*Q*N``
    ``intra_mask``              ``g_h*Q*Q`` (causal decay mask)     --
    ``intra_output``            ``Q*Q*L``   (G X_chunk)             ``Q*Q*L``
    ``chunk_state_scale``       ``Q*L``     (X * chunk decay)       --
    ``chunk_state``             ``Q*N*L``   (B_chunk^T Xbar)        ``Q*N*L``
    ``state_passing``           ``L*N``     (S = a S + S_chunk)     ``L*N``
    ``inter_output``            ``Q*N*L``   (C_chunk S_prev)        ``Q*N*L``
    ``combine_outputs``         --                                  ``Q*L``
    ==========================  ==================================  ==========

    The number of chunks is ``ceil(tokens / Q)`` and the TAIL CHUNK IS CHARGED
    FULL — a disclosed relaxation (it over-charges by at most one chunk), not
    a silent rounding.

    Why chunking is worth modelling at all: the state is read and written ONCE
    PER CHUNK, not once per token, so ``state_bytes`` falls by ``Q``. That is
    the entire point of SSD and it is the term a per-token law cannot express.

    CAUSAL STANCE (stated because the delta-rule law takes the OTHER one, and an
    unexplained 2x between two blocks on one chiplet reads as an accident). This
    law charges the intra-chunk terms DENSE: the full ``Q*Q`` for
    ``intra_scores`` and ``intra_output``, plus an explicit ``intra_mask`` term
    for applying the causal decay mask afterwards. That is what a vector engine
    actually retires for SSD — the reference implementation forms the whole
    ``Q x Q`` score block and masks it, because masking inside the product costs
    more than the half it saves. :func:`delta_rule_work`'s chunked form charges
    the STRICT LOWER TRIANGLE instead, because the UT transform it is built on
    never materializes the upper half at all. Both are modelling choices, both
    are LAW_UNVALIDATED, and the asymmetry is deliberate.
    """
    tokens = float(tokens)
    d_inner = int(d_inner)
    d_state = int(d_state)
    n_groups = int(n_groups)
    n_heads = int(n_heads)
    q = int(chunk_size)
    if q < 2:
        raise ValueError(
            f"ssd_chunked_scan_work: chunk_size = {q} is not chunked; call "
            "ssm_recurrent_scan_work, which is the unchunked law (and the one with "
            "an OPTIMA reference)."
        )
    if n_groups < 1 or d_inner % n_groups != 0:
        raise ValueError(
            f"ssd_chunked_scan_work: n_groups = {n_groups} must be >= 1 and divide "
            f"d_inner = {d_inner}."
        )
    if n_heads < 1 or n_heads % n_groups != 0:
        raise ValueError(
            f"ssd_chunked_scan_work: n_heads = {n_heads} must be >= 1 and divisible by "
            f"n_groups = {n_groups}."
        )
    groups = float(n_groups)
    chunks = math.ceil(tokens / q) if tokens > 0 else 0
    scale = float(chunks) * groups
    ell = float(d_inner // n_groups)
    n = float(d_state)
    gh = float(n_heads // n_groups)
    qf = float(q)
    return VectorWork(
        law="ssd_chunked_scan",
        validated=LAW_UNVALIDATED,
        terms=(
            ("decay_cumprod", scale * gh * qf, 0.0),
            ("intra_scores", scale * qf * qf * n, scale * qf * qf * n),
            ("intra_mask", scale * gh * qf * qf, 0.0),
            ("intra_output", scale * qf * qf * ell, scale * qf * qf * ell),
            ("chunk_state_scale", scale * qf * ell, 0.0),
            ("chunk_state", scale * qf * n * ell, scale * qf * n * ell),
            ("state_passing", scale * ell * n, scale * ell * n),
            ("inter_output", scale * qf * n * ell, scale * qf * n * ell),
            ("combine_outputs", 0.0, scale * qf * ell),
        ),
        state_bytes=float(chunks) * 2.0 * d_inner * d_state * float(act_bytes),
        detail=(
            ("tokens", tokens),
            ("chunks", float(chunks)),
            ("tokens_charged", float(chunks) * qf),
            ("chunk_size", qf),
            ("state_elements", float(d_inner * d_state)),
        ),
    )


# --- Delta rule (gated DeltaNet / RWKV-7 class) and RG-LRU -----------------


def _delta_state_dims(
    num_key_heads: int, key_head_dim: int, num_value_heads: int, value_head_dim: int
) -> Tuple[int, int, int]:
    """(key heads, d_k, effective d_v per key head) for a delta-rule mixer.

    The fast-weight state is ``k_heads x d_k x d_v``. When value heads
    outnumber key heads (Qwen3.5: 16 key x 128 against 32 value x 128) each
    key head carries every value channel it serves, so the effective value
    width per key head is ``value_dim / num_key_heads`` and the total state is
    exactly ``key_head_dim * value_dim`` elements.
    """
    num_key_heads = int(num_key_heads)
    num_value_heads = int(num_value_heads)
    if num_key_heads < 1 or num_value_heads < 1:
        raise ValueError("delta rule: num_key_heads and num_value_heads must be >= 1")
    if num_value_heads % num_key_heads != 0:
        raise ValueError(
            f"delta rule: num_key_heads = {num_key_heads} must divide num_value_heads "
            f"= {num_value_heads} (a key head carries whole value heads)."
        )
    value_dim = num_value_heads * int(value_head_dim)
    return (num_key_heads, int(key_head_dim), value_dim // num_key_heads)


def delta_rule_work(
    tokens: float,
    *,
    num_key_heads: int,
    key_head_dim: int,
    num_value_heads: int,
    value_head_dim: int,
    chunk_size: int = 1,
    output_gate: bool = True,
    decay_gate: bool = True,
    act_bytes: float = 1.0,
) -> VectorWork:
    """Gated delta rule / DeltaNet / RWKV-7-class fast-weight update (P2.6 3).

    **UNVALIDATED. NO NUMERIC REFERENCE EXISTS ANYWHERE.** OPTIMA models no
    linear-attention recurrence of any kind, so unlike the SSM scan there is
    nothing to check these counts against. They are derived from the delta
    rule itself and every term is named so a reader can dispute one term
    instead of the whole law. The only hard guarantee is physics: the priced
    result can never exceed ``lanes * clock`` ops/s (see
    :meth:`CimDeviceModel.price_vector_work`), which is the DESIGN2 section-8
    erratum precedent applied before the fact rather than after it.

    State: ``k_heads x d_k x d_v`` (see :func:`_delta_state_dims`).

    RECURRENT form (``chunk_size <= 1``), per token, per key head — the
    gated delta rule ``S <- diag(a) S (I - b k k^T) + b v k^T`` executed as
    read / correct / decay / write, plus the output read:

    ============================  ============  ============
    term                          muls          adds
    ============================  ============  ============
    ``read_old_value``  S^T k     ``d_k*d_v``   ``d_k*d_v``
    ``delta_correct``   b(v-vold) ``d_v``       ``d_v``
    ``state_decay``     S *= a    ``d_k*d_v``   --      (only with decay_gate)
    ``rank1_write``     S += k(x)u ``d_k*d_v``  ``d_k*d_v``
    ``output_read``     S^T q     ``d_k*d_v``   ``d_k*d_v``
    ``output_gate``     o *= g    ``d_v``       --      (only with output_gate)
    ============================  ============  ============

    CHUNKED form (``chunk_size >= 2``), per chunk of ``Q`` tokens, per key
    head — the standard UT-transform decomposition. ``T`` is the ``Q x Q``
    unit-lower-triangular matrix whose inverse turns the sequential removals
    into two matrix products:

    ==========================  ======================================
    term                        muls (adds equal unless noted)
    ==========================  ======================================
    ``decay_cumprod``           ``Q*d_k``            (no adds)
    ``gate_scale``              ``2*Q*d_k``          (no adds)
    ``kk_scores``               ``Q(Q-1)/2 * d_k``
    ``ut_transform``            ``Q(Q-1)(Q-2)/6``    (unit-lower-tri inverse)
    ``w_pseudo_value``          ``Q(Q+1)/2 * d_v``
    ``u_pseudo_key``            ``Q(Q+1)/2 * d_k``
    ``state_read``              ``Q*d_k*d_v``
    ``state_write``             ``(Q+1)*d_k*d_v``    adds ``(Q+1)*d_k*d_v``
    ``output_inter``            ``Q*d_k*d_v``
    ``output_intra``            ``Q(Q-1)/2*(d_k+d_v)``
    ``output_gate``             ``Q*d_v``            (no adds, if gated)
    ==========================  ======================================

    The tail chunk is charged FULL (disclosed, same rule as the SSD law), and
    the state moves once per chunk instead of once per token.

    CAUSAL STANCE (stated because the SSD law takes the OTHER one). The chunked
    terms above are TRIANGLE ONLY — ``Q(Q-1)/2`` for ``kk_scores`` and
    ``output_intra`` — because the UT-transform decomposition works on a
    unit-lower-triangular matrix and never materializes the upper half.
    :func:`ssd_chunked_scan_work` charges the dense ``Q*Q`` and then an explicit
    causal-mask term, because that is what an SSD vector kernel retires. So the
    two laws differ by roughly 2x on their causal terms BY CHOICE, not by
    accident; both are LAW_UNVALIDATED and neither is wrong against a reference,
    because no reference exists for either.
    """
    tokens = float(tokens)
    heads, d_k, d_v = _delta_state_dims(
        num_key_heads, key_head_dim, num_value_heads, value_head_dim
    )
    if d_k < 1 or d_v < 1:
        raise ValueError("delta rule: key_head_dim and value_head_dim must be >= 1")
    q = int(chunk_size)
    state_elements = float(heads * d_k * d_v)
    if q <= 1:
        per = float(heads * d_k * d_v)
        terms = [
            ("read_old_value", tokens * per, tokens * per),
            ("delta_correct", tokens * heads * d_v, tokens * heads * d_v),
        ]
        if decay_gate:
            terms.append(("state_decay", tokens * per, 0.0))
        terms.append(("rank1_write", tokens * per, tokens * per))
        terms.append(("output_read", tokens * per, tokens * per))
        if output_gate:
            terms.append(("output_gate", tokens * heads * d_v, 0.0))
        return VectorWork(
            law="delta_rule_recurrent",
            validated=LAW_UNVALIDATED,
            terms=tuple(terms),
            state_bytes=tokens * 2.0 * state_elements * float(act_bytes),
            detail=(
                ("tokens", tokens),
                ("key_heads", float(heads)),
                ("d_k", float(d_k)),
                ("d_v_effective", float(d_v)),
                ("state_elements", state_elements),
            ),
        )
    chunks = math.ceil(tokens / q) if tokens > 0 else 0
    scale = float(chunks) * float(heads)
    qf = float(q)
    tri = qf * (qf - 1.0) / 2.0            # strictly lower triangle of Q x Q
    tri_incl = qf * (qf + 1.0) / 2.0       # lower triangle including diagonal
    ut = qf * (qf - 1.0) * (qf - 2.0) / 6.0
    kv = float(d_k * d_v)
    terms = [
        ("decay_cumprod", scale * qf * d_k, 0.0),
        ("gate_scale", scale * 2.0 * qf * d_k, 0.0),
        ("kk_scores", scale * tri * d_k, scale * tri * d_k),
        ("ut_transform", scale * ut, scale * ut),
        ("w_pseudo_value", scale * tri_incl * d_v, scale * tri_incl * d_v),
        ("u_pseudo_key", scale * tri_incl * d_k, scale * tri_incl * d_k),
        ("state_read", scale * qf * kv, scale * qf * kv),
        ("state_write", scale * (qf + 1.0) * kv, scale * (qf + 1.0) * kv),
        ("output_inter", scale * qf * kv, scale * qf * kv),
        ("output_intra", scale * tri * (d_k + d_v), scale * tri * (d_k + d_v)),
    ]
    if not decay_gate:
        terms = [term for term in terms if term[0] not in ("decay_cumprod", "gate_scale")]
    if output_gate:
        terms.append(("output_gate", scale * qf * d_v, 0.0))
    return VectorWork(
        law="delta_rule_chunked",
        validated=LAW_UNVALIDATED,
        terms=tuple(terms),
        state_bytes=float(chunks) * 2.0 * state_elements * float(act_bytes),
        detail=(
            ("tokens", tokens),
            ("chunks", float(chunks)),
            ("tokens_charged", float(chunks) * qf),
            ("chunk_size", qf),
            ("key_heads", float(heads)),
            ("d_k", float(d_k)),
            ("d_v_effective", float(d_v)),
            ("state_elements", state_elements),
        ),
    )


def rg_lru_work(tokens: float, *, width: int, act_bytes: float = 1.0) -> VectorWork:
    """RecurrentGemma / Griffin RG-LRU: the STRICT SUBSET of the delta rule.

    **UNVALIDATED**, same standing as :func:`delta_rule_work`.

    The state is DIAGONAL: ``d_k = 1``, one scalar per channel, so the delta
    rule's outer-product write and its read-and-remove step both collapse.
    What survives, per token per channel:

    ==================  ====  ====
    term                muls  adds
    ==================  ====  ====
    ``recurrence_gate``  2    --   (a = exp(c * log sigmoid(r)); the
                                    transcendentals are charged one lane
                                    operation each, which is the honest
                                    floor and is disclosed as such)
    ``input_gate``       1    --   (i (*) x)
    ``state_decay``      1    --   (h *= a)
    ``input_scale``      2    --   (sqrt(1 - a^2) * (i (*) x))
    ``state_combine``   --     1
    ==================  ====  ====

    "Strict subset" is a checkable claim, not a slogan: at the same state size
    this law charges strictly fewer operations than
    ``delta_rule_work`` at ``d_k = 1``, and a test asserts it.
    """
    tokens = float(tokens)
    width = int(width)
    if width < 1:
        raise ValueError("rg_lru_work: width must be >= 1")
    per = tokens * width
    return VectorWork(
        law="rg_lru",
        validated=LAW_UNVALIDATED,
        terms=(
            ("recurrence_gate", 2.0 * per, 0.0),
            ("input_gate", per, 0.0),
            ("state_decay", per, 0.0),
            ("input_scale", 2.0 * per, 0.0),
            ("state_combine", 0.0, per),
        ),
        state_bytes=tokens * 2.0 * width * float(act_bytes),
        detail=(("tokens", tokens), ("width", float(width)), ("state_elements", float(width))),
    )


# --- Short depthwise conv (ADJ-3: the per-macro pool, absorbed) ------------


def short_conv_ops_per_result(kernel_size: int) -> int:
    """Scalar ops one causal depthwise-conv output costs: k muls + (k-1) adds.

    Depthwise means one tap set per channel, so the work is ``k * channels``
    multiplies and ``(k-1) * channels`` adds per token — there is no channel
    mixing to amortize. ADJ-3 puts this on the PER-MACRO DIGITAL POOL, so it
    is not a timed chiplet op: it is absorbed into the pool sizing derivation
    (D12) by :meth:`CimDeviceModel.digital_pool_sizing`.
    """
    kernel_size = int(kernel_size)
    if kernel_size < 1:
        raise ValueError("short_conv_ops_per_result: kernel_size must be >= 1")
    return 2 * kernel_size - 1



# ---------------------------------------------------------------------------
# THE MEASURED SYNTHESIS LIBRARY (D32) and DERIVED ENGINE SIZING (D31)
#
# D32: digital area and power are no longer DECLARED placeholders. They are
# COMPOSITIONS of blocks that were measured by synthesis — a unit count times
# that block's own `area_um2` / `power_W` — read from a library file checked in
# under `configs/hardware-config/digital_components_<tech>.yaml`, whose
# provenance header names the OPTIMA synthesis run every number came from.
#
# THE PATTERN IS OPTIMA'S, THE TIMING IS OURS. OPTIMA's `HWCollection` is a
# {block: count} bag with `get_total_area_mm2` / `get_total_power_W`, and this
# module borrows exactly that (:class:`EngineComposition`). What it does NOT
# borrow is how OPTIMA gets the counts: `create_s3_scan_collection` SIZES the
# engine to a microcycle budget and `get_execution_cycles` then returns that
# budget while ignoring its own arguments, so digital load can only cost area
# there, never time (AUDIT finding 6). Here the counts are DERIVED FROM THE
# MEASURED ANALOG M-PASS TIME (ADJ-9)
# by :meth:`CimDeviceModel.derive_engine_sizing` and the resulting engine is
# then TIMED by :meth:`CimDeviceModel.price_vector_work` like any other — the
# derivation is the exact inverse of the pricing law, so a run cannot buy an
# engine that its own laws would not use.
#
# NO MARGINS (D28). OPTIMA multiplies every collection by an `overhead`
# fraction (0.0 for the adder and the activations, 0.1 for LayerNorm and the
# scan, 0.15 for the S4 mul, 0.2 for softmax and the Mamba softplus/exp pair,
# and 0.4 for the depthwise conv) to cover glue and routing. That is a pad
# nobody measured, so it is NOT carried; :data:`OPTIMA_COLLECTION_OVERHEADS`
# records what was dropped so the omission is a number rather than a silence.
# ---------------------------------------------------------------------------


#: OPTIMA's per-collection `overhead` fractions, recorded and NOT applied (D28).
#: Read from perf_model/hardware/collections.py, 2026-08-24.
OPTIMA_COLLECTION_OVERHEADS = {
    "Softmax_Stage2": 0.2,
    "Adder_Residual": 0.0,
    "LayerNorm_Residual": 0.1,
    "Mamba_S3_SSM_Scan": 0.1,
    "Mamba_S4_TwiceMul": 0.15,
    # The largest pad OPTIMA carries, and the one this repo's per-macro pool
    # composition mirrors (compose_macro_pool's conv-tap rows, ADJ-3). It was
    # missing from this record, which made the stated range 0.0-0.2 wrong.
    "Mamba_DepthwiseConv_k": 0.4,
    "Mamba_SoftplusAdd": 0.2,
    "Mamba_ExpMul": 0.2,
}

#: Directory the checked-in libraries live in (repo-relative, never OPTIMA).
SYNTHESIS_LIBRARY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "hardware-config"
)

#: Provenance labels for one composed row (D21: one accounting, one label).
PROVENANCE_MEASURED = "measured-synthesis"
"""The per-unit area/power came from a synthesis report, verbatim."""
PROVENANCE_DERIVED_COUNT = "derived-count"
"""The unit COUNT was derived by a law in this module (the beat, or a width)."""
PROVENANCE_DECLARED_COUNT = "declared-count"
"""The unit COUNT came from a card knob a human wrote down."""


class SynthesisLibraryError(ValueError):
    """The library cannot answer, and no number is invented in its place.

    Raised when a library file is missing, when a card names a technology that
    is not checked in, or — the important one — when a composition asks for a
    BLOCK THE LIBRARY DOES NOT NAME. D32 makes measured blocks the source of
    digital area and power; a block nobody synthesised has no area, and a
    zero-fill or an interpolation would be an invented number (ADJ-4).
    """


@dataclass(frozen=True)
class SynthesisBlock:
    """One measured block: what synthesis reported, and where it reported it."""

    name: str
    area_um2: float
    power_w: float
    frequency_ghz: float
    pipeline_depth: int
    provenance: str
    description: str = ""
    rows: int = 0
    cols: int = 0

    @property
    def area_mm2(self) -> float:
        return float(self.area_um2) / 1.0e6


@dataclass(frozen=True)
class SynthesisLibrary:
    """A checked-in measured-block library, loaded once and cached by path.

    ``block(name)`` is the whole read interface, and it REFUSES BY NAME.
    """

    technology: str
    source: str
    source_reports: str
    path: str
    blocks: "OrderedDict[str, SynthesisBlock]"

    # -- loading ---------------------------------------------------------

    @staticmethod
    def path_for(technology: str) -> str:
        """Repo path of a named technology's library."""
        tech = str(technology).strip()
        if os.path.sep in tech or tech.endswith((".yaml", ".yml")):
            return tech
        return os.path.join(SYNTHESIS_LIBRARY_DIR, f"digital_components_{tech}.yaml")

    @classmethod
    def load(cls, technology: str = "22nm") -> "SynthesisLibrary":
        """Load (and cache) the library for a technology name or a file path."""
        path = cls.path_for(technology)
        key = os.path.abspath(path)
        cached = _SYNTHESIS_LIBRARY_CACHE.get(key)
        if cached is None:
            cached = cls._read(path)
            _SYNTHESIS_LIBRARY_CACHE[key] = cached
        return cached

    @classmethod
    def _read(cls, path: str) -> "SynthesisLibrary":
        if not os.path.isfile(path):
            available = sorted(
                name[len("digital_components_"):-len(".yaml")]
                for name in (
                    os.listdir(SYNTHESIS_LIBRARY_DIR)
                    if os.path.isdir(SYNTHESIS_LIBRARY_DIR)
                    else []
                )
                if name.startswith("digital_components_") and name.endswith(".yaml")
            )
            raise SynthesisLibraryError(
                f"no measured synthesis library at {path!r}. D32 composes digital area "
                "and power from blocks a synthesis run measured, so a technology that is "
                "not checked in has no numbers at all — it is not scaled from another "
                f"node. Checked in here: {available or 'none'}."
            )
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        raw = data.get("blocks")
        if not isinstance(raw, dict) or not raw:
            raise SynthesisLibraryError(
                f"{path}: the library declares no 'blocks:' mapping. The file is the "
                "provenance header plus the measured blocks; without the blocks there is "
                "nothing to compose."
            )
        blocks: "OrderedDict[str, SynthesisBlock]" = OrderedDict()
        for name, entry in raw.items():
            if not isinstance(entry, dict):
                raise SynthesisLibraryError(f"{path}: block {name!r} is not a mapping.")
            missing = [
                key for key in ("area_um2", "power_W", "provenance") if key not in entry
            ]
            if missing:
                raise SynthesisLibraryError(
                    f"{path}: block {name!r} declares no {', '.join(missing)}. Every "
                    "block carries its measured area, its measured power and the "
                    "provenance line that says which synthesis run measured them."
                )
            blocks[str(name)] = SynthesisBlock(
                name=str(name),
                area_um2=float(entry["area_um2"]),
                power_w=float(entry["power_W"]),
                frequency_ghz=float(entry.get("frequency_GHz", 0.0) or 0.0),
                pipeline_depth=int(entry.get("pipeline_depth", 1) or 1),
                provenance=str(entry["provenance"]),
                description=str(entry.get("description", "")),
                rows=int(entry.get("rows", 0) or 0),
                cols=int(entry.get("cols", 0) or 0),
            )
        return cls(
            technology=str(data.get("technology", "")),
            source=str(data.get("source", "")),
            source_reports=str(data.get("source_reports", "")),
            path=path,
            blocks=blocks,
        )

    # -- reading ---------------------------------------------------------

    def block(self, name: str) -> SynthesisBlock:
        """The named measured block, or a refusal that names it (D32/ADJ-4)."""
        found = self.blocks.get(str(name))
        if found is None:
            raise SynthesisLibraryError(
                f"the {self.technology} synthesis library names no block {str(name)!r}. "
                "D32 prices digital silicon from MEASURED blocks only: there is no "
                "default block, no substitute and no zero-fill, because a block nobody "
                "synthesised has no area and no power to report. The library names: "
                f"{', '.join(self.blocks)}. Source: {self.source_reports or self.source}."
            )
        return found

    def area_mm2(self, name: str) -> float:
        return self.block(name).area_mm2

    def power_w(self, name: str) -> float:
        return float(self.block(name).power_w)

    def provenance_line(self) -> str:
        return (
            f"synthesis library {self.technology}: {len(self.blocks)} measured blocks "
            f"from {self.source_reports or self.source}, checked in at {self.path} "
            "(D32; nothing is read from the source project at runtime)."
        )


#: Path -> loaded library. A library file is immutable data; one read is enough.
_SYNTHESIS_LIBRARY_CACHE: "OrderedDict[str, SynthesisLibrary]" = OrderedDict()


@dataclass(frozen=True)
class BlockCount:
    """``count`` copies of one measured block, and what they cost."""

    block: str
    count: int
    unit_area_mm2: float
    unit_power_w: float
    role: str
    count_provenance: str
    unit_provenance: str = PROVENANCE_MEASURED

    @property
    def area_mm2(self) -> float:
        return self.count * self.unit_area_mm2

    @property
    def power_w(self) -> float:
        return self.count * self.unit_power_w


@dataclass(frozen=True)
class EngineComposition:
    """One engine, composed of measured blocks — the D32 accounting unit.

    ``area_mm2`` and ``power_w`` are ``sum(count * unit)`` over :attr:`units`
    and nothing else: no overhead fraction, no margin, no rounding (D28). Every
    row carries TWO provenance labels, because they answer different questions —
    ``unit_provenance`` says where the per-unit silicon number came from
    (always ``measured-synthesis`` here) and ``count_provenance`` says where the
    COUNT came from (``derived-count`` when a law in this module produced it
    from the beat or a width, ``declared-count`` when a card knob did).
    """

    name: str
    technology: str
    units: Tuple[BlockCount, ...]
    basis: str
    disclosures: Tuple[str, ...] = ()

    @property
    def area_mm2(self) -> float:
        return math.fsum(unit.area_mm2 for unit in self.units)

    @property
    def power_w(self) -> float:
        return math.fsum(unit.power_w for unit in self.units)

    @property
    def blocks(self) -> "OrderedDict[str, int]":
        out: "OrderedDict[str, int]" = OrderedDict()
        for unit in self.units:
            out[unit.block] = out.get(unit.block, 0) + int(unit.count)
        return out

    def summary(self) -> "OrderedDict[str, object]":
        return OrderedDict(
            (
                ("engine", self.name),
                ("technology", self.technology),
                ("area_mm2", self.area_mm2),
                ("power_W", self.power_w),
                ("basis", self.basis),
                (
                    "units",
                    [
                        OrderedDict(
                            (
                                ("block", unit.block),
                                ("role", unit.role),
                                ("count", int(unit.count)),
                                ("unit_area_mm2", unit.unit_area_mm2),
                                ("unit_power_W", unit.unit_power_w),
                                ("area_mm2", unit.area_mm2),
                                ("power_W", unit.power_w),
                                ("count_provenance", unit.count_provenance),
                                ("unit_provenance", unit.unit_provenance),
                            )
                        )
                        for unit in self.units
                    ],
                ),
                ("disclosures", list(self.disclosures)),
            )
        )

    def report(self) -> str:
        lines = [
            f"[FWS-CIM] {self.name} (D32, {self.technology} measured synthesis): "
            f"{self.area_mm2:.6g} mm2, {self.power_w:.6g} W",
            f"  basis  {self.basis}",
        ]
        for unit in self.units:
            lines.append(
                f"  {unit.block:<32} x {unit.count:<8d} {unit.area_mm2:>12.6g} mm2 "
                f"{unit.power_w:>10.6g} W  [{unit.count_provenance}/{unit.unit_provenance}]"
                f"  {unit.role}"
            )
        for note in self.disclosures:
            lines.append(f"  [NOTE] {note}")
        return "\n".join(lines)


def _units(
    library: SynthesisLibrary,
    rows: Sequence[Tuple[str, int, str]],
    *,
    count_provenance: str,
) -> Tuple[BlockCount, ...]:
    """Build BlockCount rows, refusing any block the library does not name."""
    out: List[BlockCount] = []
    for block_name, count, role in rows:
        count = int(count)
        if count <= 0:
            continue
        block = library.block(block_name)
        out.append(
            BlockCount(
                block=block.name,
                count=count,
                unit_area_mm2=block.area_mm2,
                unit_power_w=float(block.power_w),
                role=role,
                count_provenance=count_provenance,
                unit_provenance=block.provenance,
            )
        )
    return tuple(out)


#: One scan/vector LANE, spelled in measured blocks. See
#: :func:`compose_vector_engine` for why a lane holds both an FP_MULT and an
#: FP_ADD.
VECTOR_LANE_BLOCKS = (
    ("FP_MULT", 1, "the lane's multiply path"),
    ("FP_ADD", 1, "the lane's add path"),
    ("M_REG", 1, "the lane's operand/accumulate register"),
)


def compose_vector_engine(
    library: SynthesisLibrary,
    lanes: int,
    *,
    count_provenance: str = PROVENANCE_DERIVED_COUNT,
) -> EngineComposition:
    """The scan/vector engine of the shared digital chiplet, in measured blocks.

    ONE LANE HOLDS BOTH ARITHMETIC UNITS, and that is a consequence of the law
    rather than a pad. :meth:`CimDeviceModel.price_vector_work` charges one
    cycle per lane for ONE SCALAR OPERATION, "a multiply OR an add", and the
    laws that feed it (`ssm_recurrent_scan_work`, `ssd_chunked_scan_work`,
    `delta_rule_work`, `rg_lru_work`) hand it a single ``ops = muls + adds``
    total with no schedule attached. A lane that may be handed either kind of
    operation on any cycle therefore contains both units. Splitting the lanes
    by the mul/add mix would be cheaper silicon, but it would only be honest
    with a per-cycle SCHEDULE that the beat-level sizing does not have, and a
    lane that could not retire the op it was handed would break the peak the
    pricing law is pinned to. The stance is stated here rather than padded into
    a number (D28), and it is the ONE stance: no other composition of a vector
    lane exists in this module.

    OPTIMA's `create_s3_scan_collection` instead emits separate `FP_MULT` and
    `FP_ADD` counts, because it derives them from ONE hard-coded scan schedule.
    Our laws are not one schedule, so that split is not available to us; the
    difference is named here (D21, one accounting per metric).
    """
    lanes = int(lanes)
    if lanes < 1:
        raise ValueError("compose_vector_engine: lanes must be >= 1")
    return EngineComposition(
        name="scan/vector engine",
        technology=library.technology,
        units=_units(
            library,
            [(block, count * lanes, role) for block, count, role in VECTOR_LANE_BLOCKS],
            count_provenance=count_provenance,
        ),
        basis=(
            f"{lanes} lane(s) x (FP_MULT + FP_ADD + M_REG): one lane retires one scalar "
            "operation of EITHER kind per vector cycle, which is the peak "
            "CimDeviceModel.price_vector_work is bound to, so the lane holds both "
            "arithmetic units"
        ),
        disclosures=(
            "vector lane composition: a lane holds an FP_MULT and an FP_ADD because the "
            "pricing law lets it retire either on any cycle and the work laws report one "
            "ops total, not a schedule. A mix-split engine would be smaller and is NOT "
            "modelled, because nothing in this repo declares the per-cycle mix.",
            "no overhead fraction is applied (D28): OPTIMA's own scan collection carries "
            f"overhead {OPTIMA_COLLECTION_OVERHEADS['Mamba_S3_SSM_Scan']} for glue and "
            "routing; that pad is dropped and named rather than carried.",
        ),
    )


def compose_softmax_engine(
    library: SynthesisLibrary,
    lanes: int,
    replicas: int = 1,
    *,
    count_provenance: str = PROVENANCE_DECLARED_COUNT,
) -> EngineComposition:
    """The softmax pipeline, in measured blocks — OPTIMA's collection, term for term.

    The per-lane census is copied from OPTIMA's `create_softmax_collection`
    (perf_model/hardware/collections.py), which is a REFERENCE in the D21 sense:
    for a width ``w`` it instantiates ``FP_COMP: w - 1`` (the running max tree),
    ``FP_ADD: 2w - 1`` (the max subtract plus the sum tree), ``FP_MULT: 2w``,
    ``BF16_EXP: w`` and ``BF16_RECIP: 1`` (one reciprocal per row, shared by the
    whole width). The width here is this card's ``cim.fabric.softmax_lanes``,
    and the whole census is instantiated once PER REPLICA, because
    :meth:`CimDeviceModel.softmax_cycles` already spends ``softmax_lanes *
    replicas`` lanes.

    Its pipeline depth is 20 in that collection, which is the same 20 this
    repo's ``cim.fabric.softmax_pipeline_depth`` defaults to — the two numbers
    agree because they have the same origin, and a test pins that.
    """
    lanes = int(lanes)
    replicas = max(1, int(replicas))
    if lanes < 1:
        raise ValueError("compose_softmax_engine: lanes must be >= 1")
    rows = [
        ("FP_COMP", replicas * (lanes - 1), "running-max comparison tree"),
        ("FP_ADD", replicas * (2 * lanes - 1), "max subtract + the sum tree"),
        ("FP_MULT", replicas * (2 * lanes), "scale and normalise"),
        ("BF16_EXP", replicas * lanes, "the exponential itself"),
        ("BF16_RECIP", replicas * 1, "one reciprocal per row, shared by the width"),
    ]
    return EngineComposition(
        name="softmax lanes",
        technology=library.technology,
        units=_units(library, rows, count_provenance=count_provenance),
        basis=(
            f"OPTIMA create_softmax_collection at width {lanes}, instantiated "
            f"{replicas} time(s) (one per attention replica, matching "
            "CimDeviceModel.softmax_cycles which spends softmax_lanes x replicas)"
        ),
        disclosures=(
            "no overhead fraction is applied (D28): OPTIMA's softmax collection carries "
            f"overhead {OPTIMA_COLLECTION_OVERHEADS['Softmax_Stage2']}, the largest pad "
            "in its library, and it is dropped and named rather than carried.",
        ),
    )


def compose_sa_fabric(
    library: SynthesisLibrary,
    rows: int,
    cols: int,
    num_arrays: int = 1,
    replicas: int = 1,
    *,
    count_provenance: str = PROVENANCE_DERIVED_COUNT,
) -> EngineComposition:
    """The systolic-array attention fabric, in measured GEMMINI blocks.

    The library's `GEMMINI_SYS_ARRAY` is a FIXED 32x32 array (its own ``rows``
    and ``cols`` fields say so), so a declared ``rows x cols`` fabric is built
    from ``ceil(rows / 32) * ceil(cols / 32)`` of them, per array, per replica.

    THE CEIL IS THE HONEST PART. OPTIMA's `DigitalTile.rtl_arrays_needed`
    multiplies the RATIOS (``rows / 32 * cols / 32``) and keeps the fraction, so
    a 32x64 fabric costs exactly 2.0 arrays there and a 40x64 fabric costs 2.5 —
    half a synthesised block. Blocks are integers exactly as chips are (D21), so
    this composition ceils, and the difference against OPTIMA is 0 whenever the
    declared geometry is a multiple of 32 (which every shipped card is).
    """
    rows = int(rows)
    cols = int(cols)
    num_arrays = max(1, int(num_arrays))
    replicas = max(1, int(replicas))
    gemmini = library.block("GEMMINI_SYS_ARRAY")
    phys_rows = int(gemmini.rows or 0)
    phys_cols = int(gemmini.cols or 0)
    if phys_rows < 1 or phys_cols < 1:
        raise SynthesisLibraryError(
            f"{library.technology} GEMMINI_SYS_ARRAY declares no rows/cols geometry, so "
            "the number of blocks a rows x cols fabric needs cannot be counted. The "
            "block's own geometry is what makes the count a measurement rather than an "
            "assumption."
        )
    tiles = _ceil_div(rows, phys_rows) * _ceil_div(cols, phys_cols)
    blocks = tiles * num_arrays * replicas
    exact = (rows % phys_rows == 0) and (cols % phys_cols == 0)
    disclosures = [
        "no overhead fraction is applied (D28).",
        "the attention datapath's converter blocks (COMBINED_CONV, "
        "COMBINED_CONV_SEQUENTIAL, INT8_BUFF_DIV_SYS_ARRAY_PENALTY) are NOT composed: "
        "all three are measured at 0 area and 0 power in both checked-in libraries, and "
        "OPTIMA's own source note records its counts for them (rows, cols, 2*rows*cols) "
        "as unexplained placeholders. They are named here and left out, so no invented "
        "count enters and no number changes.",
        "TRANSPOSER: ONE per attention replica (one K-matrix preparation path per "
        "replica). OPTIMA instantiates 9 per replica and its own source comment says "
        "'the 9 is unexplained'; an uncalibrated count is not copied (ADJ-4). At "
        f"{library.technology} the difference is {8 * replicas} x "
        f"{library.block('TRANSPOSER').area_mm2:.6g} mm2.",
    ]
    if not exact:
        disclosures.append(
            f"the declared fabric is {rows} x {cols}, which is not a whole multiple of "
            f"the measured {phys_rows} x {phys_cols} block: {tiles} blocks are "
            f"instantiated per array and the remainder columns/rows are IDLE SILICON "
            "that this composition still pays for, exactly as the machine would."
        )
    return EngineComposition(
        name="SA attention fabric",
        technology=library.technology,
        units=_units(
            library,
            [
                (
                    "GEMMINI_SYS_ARRAY",
                    blocks,
                    f"{phys_rows}x{phys_cols} systolic blocks tiling {rows}x{cols} "
                    f"x {num_arrays} array(s) x {replicas} replica(s)",
                ),
                ("TRANSPOSER", replicas, "K-matrix preparation, one per replica"),
            ],
            count_provenance=count_provenance,
        ),
        basis=(
            f"ceil({rows}/{phys_rows}) * ceil({cols}/{phys_cols}) = {tiles} measured "
            f"{phys_rows}x{phys_cols} blocks per array, x {num_arrays} arrays "
            f"x {replicas} replicas = {blocks}, plus {replicas} TRANSPOSER"
        ),
        disclosures=tuple(disclosures),
    )


def compose_macro_pool(
    library: SynthesisLibrary,
    sizing: "DigitalPoolSizing",
    *,
    count_provenance: str = PROVENANCE_DERIVED_COUNT,
) -> EngineComposition:
    """One macro's digital pool, in measured blocks (D12 sizing x D32 pricing).

    The counts are :meth:`CimDeviceModel.digital_pool_sizing`'s own derived
    widths, unchanged: ``adders`` shift-add adders, ``conv_lanes`` depthwise-conv
    lanes, and one holding register per lane.

    This CLOSES the named gap P7.2 left open. `price_packing_accumulation`
    discloses that "the accumulator's holding REGISTER is not priced separately:
    no card declares a register area". The library declares one — `M_REG` — so
    the register is now a measured block with a counted instance per lane
    instead of a gap.
    """
    lanes = max(0, int(sizing.lanes))
    conv_lanes = max(0, int(sizing.conv_lanes))
    rows = [
        ("FP_ADD", int(sizing.adders), "shift-add tree adders (bit-slice reduction)"),
        ("FP_MULT", conv_lanes, "short depthwise conv taps (ADJ-3)"),
        ("FP_ADD", conv_lanes, "short depthwise conv accumulate (ADJ-3)"),
        ("M_REG", lanes + conv_lanes, "one holding register per pool lane"),
    ]
    return EngineComposition(
        name="per-macro digital pool",
        technology=library.technology,
        units=_units(library, rows, count_provenance=count_provenance),
        basis=(
            f"CimDeviceModel.digital_pool_sizing: {sizing.adders} adders + "
            f"{conv_lanes} conv lane(s) + {lanes + conv_lanes} register(s), the widths "
            "D12 derives from the macro's own result rate"
        ),
        disclosures=(
            "the pool's holding register is now a MEASURED block (M_REG), which retires "
            "the P7.2 disclosure that called it a named gap; the register count is one "
            "per lane, not one per adder, because the K-inner walk keeps one partial "
            "live per lane at a time.",
            "no overhead fraction is applied (D28).",
        ),
    )


# --- Derived engine sizing (D31): the beat sets the width -------------------


#: The lane count :meth:`CimDeviceModel.price_vector_work` reports while the
#: D31 PROBE pass is running. During the probe a vector op costs 0 s by
#: construction, so this number times nothing; it exists only so the work laws
#: can be evaluated before any width has been derived. Large enough that
#: ``ceil(ops / lanes) == 1`` for every work count this repo can produce.
_PROBE_LANES = 1 << 62


class EngineSizingError(ValueError):
    """The beat cannot be held at ANY engine width, and no margin hides it.

    D31 derives the engine from the beat. When the pipeline fill alone already
    costs more cycles than the beat has, no lane count fixes it: adding lanes
    shortens the streaming term and never the fill. That is a real, reportable
    infeasibility (the beat is too short for the declared clock and pipeline
    depth), so it is refused by name rather than clamped.
    """


@dataclass(frozen=True)
class EngineDemand:
    """What ONE stage asks of its vector engine in ONE beat.

    ``calls`` is the number of SEPARATE priced vector-op calls the stage runs in
    a beat, and ``ops`` their scalar-operation totals. They are kept apart
    because :meth:`CimDeviceModel.price_vector_work` charges every call its own
    pipeline fill: ``depth + ceil(ops/lanes) - 1`` cycles each. Summing the ops
    into one number and sizing on that would under-count the fill by
    ``(calls - 1) * (depth - 1)`` cycles, which is a real cost of the machine.

    ``analog_time_s`` is D31-v2's SIZING TARGET (ADJ-9): the wall time this
    stage's own analog macros spend passing weights in one beat, measured off
    the probe timeline. The engine is derived UP until it fits inside that
    number, so the ANALOG m-pass is the term that sets the stage's time and the
    digital side never is. A stage with no analog work leaves it at 0.0, which
    is the one case the criterion cannot reach and which the derivation then
    names rather than smooths.
    """

    stage: int
    ops: Tuple[float, ...]
    analog_time_s: float = 0.0

    @property
    def calls(self) -> int:
        return len(self.ops)

    @property
    def total_ops(self) -> float:
        return math.fsum(self.ops)


#: Why one stage's sizing target is what it is (D31-v2 / ADJ-9).
TARGET_ANALOG_STAGE = "analog_stage_time"
TARGET_NO_ANALOG_WORK = "no_analog_work_in_stage"
TARGET_ANALOG_BELOW_FILL = "analog_stage_time_below_engine_fill"


@dataclass(frozen=True)
class StageEngineSizing:
    """One stage's derived width, its sizing TARGET, and the cycles it spends."""

    stage: int
    calls: int
    #: The per-call scalar-op counts, kept so the cycle identity stays
    #: CHECKABLE: ``vector_cycles_at(ops, lanes, depth) == used_cycles``. A row
    #: that carried only the total could not reconstruct its own fill, because
    #: every call pays one.
    ops: Tuple[float, ...]
    total_ops: float
    lanes: int
    budget_cycles: int
    fill_cycles: int
    stream_cycles: int
    used_cycles: int
    time_s: float
    slack_s: float
    #: The stage's own measured ANALOG m-pass time in one beat (0.0 = none).
    analog_time_s: float = 0.0
    #: The seconds the width was actually derived against — the analog stage
    #: time where ADJ-9's criterion is reachable, the analog BEAT where it is
    #: not. Never both: one accounting per metric (D21).
    target_s: float = 0.0
    #: Which of the two it is, by name (``TARGET_*`` above).
    target_kind: str = TARGET_ANALOG_STAGE

    @property
    def analog_bound(self) -> bool:
        """Does the ANALOG m-pass set this stage's time rather than the engine?"""
        return self.analog_time_s > 0.0 and self.time_s <= self.analog_time_s

    @property
    def duty_at_target(self) -> float:
        """Engine time as a share of the target it was sized against."""
        return (self.time_s / self.target_s) if self.target_s > 0 else 0.0


@dataclass(frozen=True)
class DerivedEngineSizing:
    """The engine D31-v2 derives to the ANALOG FLOOR, with its whole basis.

    ``vector_lanes`` is the width the WORST stage needs; the chiplet card is one
    card, so one width is provisioned and the per-stage rows show which stage
    set it and how much slack the others run with. ``binding_stage`` is that
    worst stage — reporting it is the point of the exercise (D28: idle silicon
    must be visible, and so must the stage that paid for it).

    ADJ-9 (D31-v2) moved the SIZING TARGET. It used to be the analog BEAT (the
    slowest stage's time), which left every faster stage's engine free to be
    that stage's own binding term. It is now each stage's OWN analog m-pass
    time, so the analog side binds every stage it can bind — and the stages
    where it cannot are counted in :attr:`unreachable_stages` and named in the
    disclosures rather than quietly folded into the answer.
    """

    analog_beat_s: float
    clock_hz: float
    pipeline_depth: int
    vector_lanes: int
    binding_stage: int
    per_stage: Tuple[StageEngineSizing, ...]
    composition: Optional[EngineComposition] = None
    basis: str = ""
    disclosures: Tuple[str, ...] = ()
    #: ADJ-9's reachability evidence, attached by the evaluator that measured
    #: it: which stage takes longest in the probe and what it spends the time
    #: on, itemized by op block. It rides the SIZING rather than a module
    #: global because a report is written long after the derivation ran, and a
    #: global would hand one run's evidence to another run's report (D21).
    beat_setting_stage: Optional[Mapping[str, object]] = None

    @property
    def binding_row(self) -> Optional[StageEngineSizing]:
        for row in self.per_stage:
            if row.stage == self.binding_stage:
                return row
        return None

    @property
    def utilization(self) -> float:
        """Binding stage's engine duty inside its OWN sizing target (ADJ-9).

        1.0 means the engine exactly fills the analog m-pass time it was sized
        against; anything below is the integer-lane remainder, never a margin
        (D28 forbids one).
        """
        row = self.binding_row
        return row.duty_at_target if row is not None else 0.0

    @property
    def analog_bound_stages(self) -> Tuple[int, ...]:
        """Stages where the ANALOG m-pass really is the longer term."""
        return tuple(int(row.stage) for row in self.per_stage if row.analog_bound)

    @property
    def unreachable_stages(self) -> Tuple[int, ...]:
        """Stages ADJ-9's criterion cannot reach, whatever the width."""
        return tuple(
            int(row.stage)
            for row in self.per_stage
            if row.target_kind != TARGET_ANALOG_STAGE
        )

    def summary(self) -> "OrderedDict[str, object]":
        return OrderedDict(
            (
                ("analog_beat_s", self.analog_beat_s),
                ("vector_lanes", int(self.vector_lanes)),
                ("lane_provenance", PROVENANCE_DERIVED_COUNT),
                ("binding_stage", int(self.binding_stage)),
                ("vector_clock_hz", self.clock_hz),
                ("vector_pipeline_depth", int(self.pipeline_depth)),
                ("sizing_target", "analog_stage_time"),
                ("engine_duty_at_target", self.utilization),
                ("analog_bound_stages", list(self.analog_bound_stages)),
                ("stages_sized", len(self.per_stage)),
                ("unreachable_stages", list(self.unreachable_stages)),
                ("beat_setting_stage", dict(self.beat_setting_stage or {}) or None),
                ("basis", self.basis),
                (
                    "per_stage",
                    [
                        OrderedDict(
                            (
                                ("stage", int(row.stage)),
                                ("vector_calls", int(row.calls)),
                                ("scalar_ops", row.total_ops),
                                ("scalar_ops_per_call", list(row.ops)),
                                ("lanes", int(row.lanes)),
                                ("analog_time_s", row.analog_time_s),
                                ("target_s", row.target_s),
                                ("target_kind", row.target_kind),
                                ("analog_bound", bool(row.analog_bound)),
                                ("budget_cycles", int(row.budget_cycles)),
                                ("fill_cycles", int(row.fill_cycles)),
                                ("stream_cycles", int(row.stream_cycles)),
                                ("used_cycles", int(row.used_cycles)),
                                ("time_s", row.time_s),
                                ("slack_s", row.slack_s),
                            )
                        )
                        for row in self.per_stage
                    ],
                ),
                (
                    "composition",
                    self.composition.summary() if self.composition else None,
                ),
                ("disclosures", list(self.disclosures)),
            )
        )

    def report(self) -> str:
        lines = [
            "[FWS-CIM] derived engine sizing (D31-v2/ADJ-9 — sized to the ANALOG FLOOR, never a sweep)",
            f"  analog beat             {self.analog_beat_s:.6g} s (the slowest stage's "
            "probe time; the fallback budget only)",
            f"  vector lanes            {self.vector_lanes} (DERIVED; stage "
            f"{self.binding_stage} is binding, engine duty "
            f"{self.utilization * 100:.3f}% of ITS OWN analog m-pass time)",
            f"  analog-bound stages     {len(self.analog_bound_stages)} of "
            f"{len(self.per_stage)}"
            + (
                f" (unreachable: {list(self.unreachable_stages)})"
                if self.unreachable_stages
                else ""
            ),
            f"  vector engine           {self.clock_hz / 1e9:.4g} GHz, pipeline depth "
            f"{self.pipeline_depth}",
        ]
        for row in self.per_stage:
            lines.append(
                f"  stage {row.stage:<3d} {row.calls:>4d} call(s) {row.total_ops:>14.6g} ops"
                f"  {row.used_cycles:>10d}/{row.budget_cycles:<10d} cycles"
                f"  digital {row.time_s:.6g} s vs analog {row.analog_time_s:.6g} s"
                f"  ({'ANALOG-BOUND' if row.analog_bound else row.target_kind})"
            )
        if self.composition is not None:
            lines.append(self.composition.report())
        for note in self.disclosures:
            lines.append(f"  [NOTE] {note}")
        return "\n".join(lines)


def vector_cycles_at(ops: Sequence[float], lanes: int, depth: int) -> int:
    """Cycles a stage's vector calls cost at ``lanes`` — the pricing law, summed.

    Each call costs ``depth + ceil(ops/lanes) - 1``, exactly what
    :meth:`CimDeviceModel.price_vector_work` charges; a stage's calls share one
    engine, so within a beat they add. This function IS the predicate
    :func:`derive_vector_lanes` inverts, so the derivation and the pricing can
    never drift apart.
    """
    lanes = max(1, int(lanes))
    depth = max(1, int(depth))
    return int(
        sum(depth + _ceil_div(math.ceil(float(call)), lanes) - 1 for call in ops)
    )


def derive_vector_lanes(
    ops: Sequence[float], budget_s: float, clock_hz: float, depth: int
) -> int:
    """The SMALLEST lane count whose priced time fits ``budget_s`` (D31, D28).

    ``budget_cycles = floor(budget_s * clock)`` — floor, because a cycle the
    budget does not contain cannot be spent — and the answer is the smallest
    integer ``lanes`` with ``vector_cycles_at(ops, lanes, depth) <=
    budget_cycles``. Smallest for a GIVEN budget, so there is no margin (D28);
    integer, because a lane is silicon. Under ADJ-9 the budget handed in is the
    stage's own ANALOG m-pass time, which is what makes the answer a
    derive-UP: a smaller budget buys more lanes, and the analog floor is the
    smallest budget the machine can physically justify.

    The predicate is monotone in ``lanes`` (``ceil(x/lanes)`` never rises as
    lanes rise), so a binary search returns the exact minimum rather than a
    conservative one. Above ``lanes = max(ops)`` every call already costs its
    fill plus one streaming cycle, so that is the upper bound of the search and
    a budget that does not fit there does not fit anywhere —
    :class:`EngineSizingError`.
    """
    calls = [float(call) for call in ops if float(call) > 0]
    depth = max(1, int(depth))
    if budget_s <= 0 or clock_hz <= 0:
        raise EngineSizingError(
            f"derive_vector_lanes needs a positive time budget and clock (got "
            f"budget={budget_s!r}, clock={clock_hz!r}). D31 derives the engine FROM a "
            "measured time — under ADJ-9 the stage's own analog m-pass time — and with "
            "no such time there is nothing to derive from and no default width to fall "
            "back on."
        )
    if not calls:
        return 1
    budget = int(math.floor(float(budget_s) * float(clock_hz)))
    ceiling = max(1, int(math.ceil(max(calls))))
    if vector_cycles_at(calls, ceiling, depth) > budget:
        floor_cycles = len(calls) * depth
        raise EngineSizingError(
            f"no vector-engine width holds a {budget_s:.6g} s budget: {len(calls)} "
            f"call(s) cost at least {floor_cycles} cycles (pipeline depth {depth} each, "
            f"one streaming cycle each) and the budget contains only {budget} cycles at "
            f"{clock_hz / 1e9:.4g} GHz. Lanes shorten the STREAMING term and never the "
            "FILL, so this is not fixable by provisioning: the budget is shorter than "
            "the engine's own latency. Widen it, raise the clock, or declare a "
            "shallower pipeline (D31 refuses to clamp, and D28 forbids padding it)."
        )
    low, high = 1, ceiling
    while low < high:
        mid = (low + high) // 2
        if vector_cycles_at(calls, mid, depth) <= budget:
            high = mid
        else:
            low = mid + 1
    return int(low)


# --- ADJ-10: the FABRIC derives to the analog floor too ---------------------


def fold_group_split(
    num_arrays: int, qk_at, pv_at
) -> Tuple[int, int]:
    """Partition ``num_arrays`` into a QK group and a PV group (ADJ-10).

    ``qk_at(a)`` / ``pv_at(a)`` return the cycles that run costs on a group of
    ``a`` arrays. The partition returned MINIMISES ``qk_at(a_qk) +
    pv_at(a_pv)``, and the sum is the objective for one reason: the lowered
    DAG places attention as THREE SERIAL OPS (qk -> softmax -> pv, P3's
    placement, disclosed as ``attention_op_folding``), so the sum is the time
    the timeline measures on the stage, and it is the quantity ADJ-10's
    criterion is taken on. Choosing a partition against one number and judging
    it against another would be two accountings of one metric (D21).

    THE CLOSED FORM READS THE SAME PARTITION DIFFERENTLY, and that is the
    already-disclosed ADJ-8 divergence rather than a second law:
    :meth:`CimDeviceModel.attention_call_timing` reports
    ``max(qk, pv) + fill_drain`` on exactly the groups chosen here, which is
    the smaller of the two readings. Where the derivation lands on both
    shipped machines — both groups at or above the fold count — the two
    objectives agree anyway, because each run is at its own floor and no
    partition can improve either.

    Ties break toward the more BALANCED split (smallest ``max`` of the pair),
    then toward the smaller ``a_qk``, so the answer is deterministic and does
    not depend on iteration order. ``num_arrays < 2`` returns (1, 1): a
    one-array fabric cannot run the pair concurrently at all, which is the
    case :class:`CimDeviceModel` already warns about at construction and which
    this function does not silently repair.
    """
    num_arrays = int(num_arrays)
    if num_arrays < 2:
        return 1, 1
    best: Optional[Tuple[int, int, int, int]] = None
    for a_qk in range(1, num_arrays):
        a_pv = num_arrays - a_qk
        qk = int(qk_at(a_qk))
        pv = int(pv_at(a_pv))
        key = (qk + pv, max(qk, pv), a_qk)
        if best is None or key < best[:3]:
            best = (key[0], key[1], key[2], a_pv)
    assert best is not None
    return int(best[2]), int(best[3])


#: Why one stage's FABRIC width is what it is (ADJ-10). The first three mirror
#: :data:`TARGET_ANALOG_STAGE` and friends; the fourth is ADJ-10's own and has
#: no ADJ-9 twin.
TARGET_FABRIC_ANALOG_STAGE = "analog_stage_time"
TARGET_FABRIC_NO_ANALOG_WORK = "no_analog_work_in_stage"
TARGET_FABRIC_SATURATED = "fold_concurrency_saturated_below_analog_time"


@dataclass(frozen=True)
class AttentionCallDemand:
    """ONE attention call of one stage, in the dims its cycles depend on.

    Kept as CALL DIMS rather than as a duration because ADJ-10 has to re-price
    the call at a candidate width, and a duration measured at the declared
    width cannot be re-priced. Every field is what
    :meth:`CimDeviceModel.attention_call_timing` was handed, so
    :meth:`CimDeviceModel.attention_cycles_at` reconstructs the same numbers
    the run will later be priced with — the derivation IS the pricing law,
    inverted, exactly as ADJ-9's is.
    """

    m: int
    k: int
    n: int
    folds: int
    softmax_tokens: int
    heads_chip: int
    #: Cycles this stage pays ONCE for the array's fill/drain, charged on the
    #: QK op by the lowering. It is a constant of the call and no width moves
    #: it, so it is carried explicitly rather than folded into a cycle count.
    fill_drain_cycles: int = 0

    @property
    def saturation_arrays(self) -> int:
        """The width past which no further array shortens THIS call.

        ``2 * folds``: each group saturates at one fold per array, and a
        group wider than that owns arrays with nothing to carry. Not a
        margin and not a cap — it is where the derivative goes to zero.
        """
        return 2 * max(1, int(self.folds))

    @property
    def saturation_softmax_width(self) -> int:
        """The softmax width past which no further lane shortens THIS call."""
        return max(1, int(self.softmax_tokens) * int(self.heads_chip))


@dataclass(frozen=True)
class FabricDemand:
    """What ONE stage asks of its ATTENTION fabric in ONE beat (ADJ-10).

    The twin of :class:`EngineDemand`, and deliberately the same shape: a
    stage, its calls, and ``analog_time_s`` — the wall time that stage's own
    analog macros spend passing weights in the beat, measured as the UNION of
    their busy intervals off the probe timeline. One target, one measurement,
    two engines.
    """

    stage: int
    calls: Tuple[AttentionCallDemand, ...]
    analog_time_s: float = 0.0

    @property
    def saturation_arrays(self) -> int:
        return max((call.saturation_arrays for call in self.calls), default=2)

    @property
    def saturation_softmax_width(self) -> int:
        return max(
            (call.saturation_softmax_width for call in self.calls), default=1
        )


@dataclass(frozen=True)
class StageFabricSizing:
    """One stage's derived fabric width, its target, and the cycles it spends."""

    stage: int
    calls: int
    num_arrays: int
    qk_arrays: int
    pv_arrays: int
    softmax_width: int
    qk_cycles: int
    pv_cycles: int
    softmax_cycles: int
    fill_drain_cycles: int
    used_cycles: int
    budget_cycles: int
    time_s: float
    slack_s: float
    #: The stage's own measured ANALOG m-pass time in one beat (0.0 = none).
    analog_time_s: float = 0.0
    target_s: float = 0.0
    target_kind: str = TARGET_FABRIC_ANALOG_STAGE
    #: The stage's attention time at FULL fold concurrency — the floor the
    #: declared rows x cols geometry leaves behind. Equal to ``time_s``
    #: whenever the derivation saturated.
    floor_time_s: float = 0.0

    @property
    def analog_bound(self) -> bool:
        """Does the ANALOG m-pass set this stage's time rather than the fabric?"""
        return self.analog_time_s > 0.0 and self.time_s <= self.analog_time_s

    @property
    def duty_at_target(self) -> float:
        return (self.time_s / self.target_s) if self.target_s > 0 else 0.0


class FabricSizingError(ValueError):
    """A fabric width was asked for where no attention call exists to size it.

    ADJ-10 derives the fabric FROM MEASURED DEMAND. With no attention call in
    the beat there is no demand, no target and no width — and a number here
    would be an invention, not a derivation (ADJ-4).
    """


@dataclass(frozen=True)
class DerivedFabricSizing:
    """The attention fabric ADJ-10 derives to the ANALOG FLOOR, with its basis.

    ``num_arrays`` is the width the WORST stage needs, in INTEGER COPIES of the
    measured ``GEMMINI_SYS_ARRAY`` block; ``rows`` and ``cols`` per array are
    NEVER derived — they stay as declared and as measured, because inventing
    array geometry is refused (ADJ-4/D32) and only the COUNT is a composition
    of measured blocks. ``softmax_lanes`` is the same exercise on the softmax
    pipeline's own measured-lane composition.

    WHERE ADJ-10 DIFFERS FROM ADJ-9, AND WHY. ADJ-9's scan derivation falls
    back to the analog BEAT for a stage whose criterion is unreachable. That is
    right there: an unreachable scan stage is one with NO analog work, or one
    whose analog time is under the engine's own pipeline fill, and in neither
    case is there a floor to walk to. The fabric has one. Its concurrency
    SATURATES at ``2 * folds`` — past that, another copy of the measured block
    carries no fold and buys no time — so where the analog m-pass cannot be
    reached this derivation goes to SATURATION and names the residue, rather
    than falling back to a larger budget that would buy a NARROWER fabric than
    the machine can use. Neither rule pads and neither clamps: ADJ-9 stops at
    the smallest width that meets its target, ADJ-10 stops at the smallest
    width that meets its target OR at the smallest width past which no width
    helps, whichever comes first.
    """

    analog_beat_s: float
    clock_hz: float
    rows: int
    cols: int
    replicas: int
    num_arrays: int
    softmax_lanes: int
    declared_num_arrays: int
    declared_softmax_lanes: int
    binding_stage: int
    per_stage: Tuple[StageFabricSizing, ...]
    composition: Optional[EngineComposition] = None
    softmax_composition: Optional[EngineComposition] = None
    basis: str = ""
    disclosures: Tuple[str, ...] = ()

    @property
    def binding_row(self) -> Optional[StageFabricSizing]:
        for row in self.per_stage:
            if row.stage == self.binding_stage:
                return row
        return None

    @property
    def target_ratio(self) -> float:
        """Binding stage's attention time DIVIDED BY its analog target.

        Deliberately NOT called a duty or a utilization, and deliberately not
        capped at 1: below 1 the analog m-pass is the longer term and the
        remainder is the integer-copy remainder (never a margin, D28); ABOVE 1
        the fabric saturated before it reached the target and the number says
        by how much. A capped 'utilization' would hide exactly the case ADJ-10
        exists to expose.
        """
        row = self.binding_row
        return row.duty_at_target if row is not None else 0.0

    @property
    def analog_bound_stages(self) -> Tuple[int, ...]:
        return tuple(int(row.stage) for row in self.per_stage if row.analog_bound)

    @property
    def saturated_stages(self) -> Tuple[int, ...]:
        """Stages the analog floor could not be reached on, at any width."""
        return tuple(
            int(row.stage)
            for row in self.per_stage
            if row.target_kind == TARGET_FABRIC_SATURATED
        )

    @property
    def unreachable_stages(self) -> Tuple[int, ...]:
        return tuple(
            int(row.stage)
            for row in self.per_stage
            if row.target_kind != TARGET_FABRIC_ANALOG_STAGE
        )

    def summary(self) -> "OrderedDict[str, object]":
        return OrderedDict(
            (
                ("analog_beat_s", self.analog_beat_s),
                ("num_arrays", int(self.num_arrays)),
                ("num_arrays_declared", int(self.declared_num_arrays)),
                ("softmax_lanes", int(self.softmax_lanes)),
                ("softmax_lanes_declared", int(self.declared_softmax_lanes)),
                ("array_provenance", PROVENANCE_DERIVED_COUNT),
                ("array_rows", int(self.rows)),
                ("array_cols", int(self.cols)),
                ("array_geometry_provenance", PROVENANCE_DECLARED_COUNT),
                ("replicas", int(self.replicas)),
                ("binding_stage", int(self.binding_stage)),
                ("fabric_clock_hz", self.clock_hz),
                ("sizing_target", "analog_stage_time"),
                ("attention_time_over_analog_target", self.target_ratio),
                ("analog_bound_stages", list(self.analog_bound_stages)),
                ("stages_sized", len(self.per_stage)),
                ("saturated_stages", list(self.saturated_stages)),
                ("unreachable_stages", list(self.unreachable_stages)),
                ("basis", self.basis),
                (
                    "per_stage",
                    [
                        OrderedDict(
                            (
                                ("stage", int(row.stage)),
                                ("attention_calls", int(row.calls)),
                                ("num_arrays", int(row.num_arrays)),
                                ("qk_arrays", int(row.qk_arrays)),
                                ("pv_arrays", int(row.pv_arrays)),
                                ("softmax_width", int(row.softmax_width)),
                                ("qk_cycles", int(row.qk_cycles)),
                                ("pv_cycles", int(row.pv_cycles)),
                                ("softmax_cycles", int(row.softmax_cycles)),
                                ("fill_drain_cycles", int(row.fill_drain_cycles)),
                                ("used_cycles", int(row.used_cycles)),
                                ("budget_cycles", int(row.budget_cycles)),
                                ("analog_time_s", row.analog_time_s),
                                ("target_s", row.target_s),
                                ("target_kind", row.target_kind),
                                ("analog_bound", bool(row.analog_bound)),
                                ("time_s", row.time_s),
                                ("floor_time_s", row.floor_time_s),
                                ("slack_s", row.slack_s),
                            )
                        )
                        for row in self.per_stage
                    ],
                ),
                (
                    "composition",
                    self.composition.summary() if self.composition else None,
                ),
                (
                    "softmax_composition",
                    self.softmax_composition.summary()
                    if self.softmax_composition
                    else None,
                ),
                ("disclosures", list(self.disclosures)),
            )
        )

    def report(self) -> str:
        lines = [
            "[FWS-CIM] derived attention fabric (ADJ-10 — every composable digital "
            "engine derives to the ANALOG FLOOR)",
            f"  analog beat             {self.analog_beat_s:.6g} s",
            f"  num_arrays              {self.num_arrays} (DERIVED, integer copies of "
            f"the measured {self.rows}x{self.cols} array; declared "
            f"{self.declared_num_arrays})",
            f"  softmax_lanes           {self.softmax_lanes} (DERIVED; declared "
            f"{self.declared_softmax_lanes})",
            f"  analog-bound stages     {len(self.analog_bound_stages)} of "
            f"{len(self.per_stage)}"
            + (
                f" (saturated: {list(self.saturated_stages)})"
                if self.saturated_stages
                else ""
            ),
        ]
        for row in self.per_stage:
            lines.append(
                f"  stage {row.stage:<3d} {row.calls:>3d} call(s)  qk {row.qk_cycles:>9d} "
                f"+ pv {row.pv_cycles:>9d} + sm {row.softmax_cycles:>6d} cycles"
                f"  digital {row.time_s:.6g} s vs analog {row.analog_time_s:.6g} s"
                f"  ({'ANALOG-BOUND' if row.analog_bound else row.target_kind})"
            )
        if self.composition is not None:
            lines.append(self.composition.report())
        if self.softmax_composition is not None:
            lines.append(self.softmax_composition.report())
        for note in self.disclosures:
            lines.append(f"  [NOTE] {note}")
        return "\n".join(lines)


# --- MLA (D6): the pricing seam and the KV-replication area consequence ----


@dataclass(frozen=True)
class MLAReplication:
    """What MLA's shared KV latent costs in REPLICATED weights (D6, P1 finding 2).

    MLA is TP-hostile: the KV latent is one shared rank, not a per-head slice,
    so the tensors that read or write it cannot be split across a tp shard and
    are stored on EVERY shard. Under FWS weights are cells, so this lands
    directly on area — our weakest metric, which is exactly why D6 says show
    how ugly it is.

    Every figure is PER LAYER unless ``layers`` is passed. ``extra_*`` is what
    replication COSTS relative to a single shard: ``(tp - 1)`` copies.
    """

    tp: int
    layers: int
    latent_bytes_per_shard: float
    up_projection_bytes_per_shard: float
    replicated_bytes_per_shard: float
    extra_replicated_bytes: float
    replicated_arrays_per_shard: int
    extra_replicated_arrays: int
    replicated_area_mm2_per_shard: float
    extra_replicated_area_mm2: float
    matrices: Tuple[Tuple[str, int, int, int], ...]   # (name, K, N, arrays)


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
        #: D31: the engine sizing DERIVED from this run's beat, installed by
        #: :meth:`install_derived_engine` once the beat is known. None means
        #: "not derived yet" — never "zero lanes".
        self._derived_engine: Optional[DerivedEngineSizing] = None
        #: D31 pass A: while probing, vector work is COUNTED and costs no time,
        #: so the measured beat is the one the ANALOG stages set. See
        #: :meth:`engine_probe`.
        self._engine_probe = False
        self._probe_ops: List[float] = []
        #: ADJ-10: the attention fabric DERIVED from this run's per-stage
        #: analog m-pass times, installed by :meth:`install_derived_fabric`.
        #: None means "not derived yet" — never "no arrays".
        self._derived_fabric: Optional[DerivedFabricSizing] = None
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

    # -- ADJ-10: the fabric's DERIVED width (declared stays a legal override) --

    @property
    def fabric_num_arrays(self) -> int:
        """Arrays of the attention fabric — DERIVED (ADJ-10), or declared.

        TWO SOURCES, IN THIS ORDER, and the report says which:

        1. the sizing :meth:`install_derived_fabric` put here once the
           per-stage analog m-pass times were measured — the ADJ-10 answer and
           the default;
        2. ``cim.fabric.num_arrays`` as DECLARED. ADJ-10 retires it as a design
           input the way D31 retired ``vector_lanes``, so a declared value that
           the derivation did not produce is an OVERRIDE riding a disclosure
           (:meth:`fabric_sizing_disclosures`), and as a SWEEP AXIS it is
           refused by name in tools/fws_qif_dse.py.

        There is no third case and no refusal: unlike a scan lane count, an
        array count always has a declared value to fall back on, because the
        SA law cannot be evaluated without one and every shipped card carries
        it. What ADJ-10 changes is which of the two a mapped run uses.
        """
        card = self.digital_card
        if card.has_fabric_override:
            return int(card.fabric_num_arrays_effective)
        if self._derived_fabric is not None:
            return int(self._derived_fabric.num_arrays)
        return int(self.fabric.num_arrays)

    @property
    def fabric_softmax_lanes(self) -> int:
        """Softmax lanes PER REPLICA — DERIVED (ADJ-10), pinned, or declared."""
        card = self.digital_card
        if card.has_fabric_override:
            return int(card.fabric_softmax_lanes_effective)
        if self._derived_fabric is not None:
            return int(self._derived_fabric.softmax_lanes)
        return int(self.fabric.softmax_lanes)

    @property
    def softmax_width(self) -> int:
        """The softmax law's own width: lanes x replicas (ADJ-10 or declared)."""
        return max(1, self.fabric_softmax_lanes * int(self.fabric.replicas))

    @property
    def fabric_provenance(self) -> str:
        """Where this run's array count came from: derived or declared."""
        if self.digital_card.has_fabric_override:
            return PROVENANCE_DECLARED_COUNT
        return (
            PROVENANCE_DERIVED_COUNT
            if self._derived_fabric is not None
            else PROVENANCE_DECLARED_COUNT
        )

    def fabric_sizing_disclosures(self) -> Tuple[str, ...]:
        """What the attention fabric's width rests on (ADJ-10, D21).

        A card that PINS the fabric says so here, under its own name and with
        the decision it overrides quoted, exactly as a declared ``vector_lanes``
        rides :meth:`vector_engine_disclosures`. It does NOT print what the
        derivation would have returned: a pinned card skips the derivation
        entirely, so no such number exists to compare with (the P7.9 correction
        applies here for the same reason).
        """
        card = self.digital_card
        if not card.has_fabric_override:
            return ()
        return (
            f"cim.cards.{card.name}.fabric_num_arrays = "
            f"{int(card.fabric_num_arrays_effective)} and .fabric_softmax_lanes = "
            f"{int(card.fabric_softmax_lanes_effective)} are DECLARED, so they are an "
            "OVERRIDE: ADJ-10 retires both as design inputs on a mapped run, because "
            "each is a count of MEASURED synthesis blocks and each is derived up until "
            "every stage's attention time fits that stage's own analog m-pass. This "
            "machine pins them instead, and the run is priced on the pinned width. The "
            "derivation did not run, so this line cannot say what it would have "
            "returned. As a SWEEP AXIS the same knob is refused by name "
            "(tools/fws_qif_dse.py REFUSED_AXES); what is legal here is pinning ONE "
            "machine, with this disclosure attached.",
        )

    def install_derived_fabric(
        self, sizing: Optional["DerivedFabricSizing"]
    ) -> None:
        """Install (or clear) the fabric ADJ-10 derived from this run's stages."""
        self._derived_fabric = sizing

    @property
    def derived_fabric(self) -> Optional["DerivedFabricSizing"]:
        return self._derived_fabric

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
        pass-1 form is unchanged; decode: B * shared_heads). The width is
        :attr:`softmax_width` — ``softmax_lanes * replicas`` with the lane
        count DERIVED under ADJ-10 when a derivation is installed and DECLARED
        otherwise, so this law and the derivation cannot drift apart.
        """
        return (
            int(self.fabric.softmax_pipeline_depth)
            + _ceil_div(int(tokens_q) * int(heads_chip), self.softmax_width)
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
        folds = max(1, h_rep * s_fold)
        # ADJ-10: the arrays are a POOL of concurrent folds, partitioned per
        # call. At num_arrays = 2 the split is (1, 1) and the two expressions
        # below are the pass-1 law bit for bit.
        a_qk, a_pv = fold_group_split(
            self.fabric_num_arrays,
            lambda arrays: self.sa_cycles(m, n, int(k) * _ceil_div(folds, arrays)),
            lambda arrays: self.sa_cycles(m, k, int(n) * _ceil_div(folds, arrays)),
        )
        qk_per_array = _ceil_div(folds, a_qk)
        pv_per_array = _ceil_div(folds, a_pv)
        qk = self.sa_cycles(m, n, int(k) * qk_per_array)
        pv = self.sa_cycles(m, k, int(n) * pv_per_array)
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
            folds=folds,
            qk_arrays=a_qk,
            pv_arrays=a_pv,
            qk_folds_per_array=qk_per_array,
            pv_folds_per_array=pv_per_array,
            softmax_width=self.softmax_width,
            call_m=int(m),
            call_k=int(k),
            call_n=int(n),
            softmax_tokens=sm_tokens,
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
        i_shared = p.shared_expert_intermediate
        ffn1 = self.arrays(p.hidden_dim, p.ffn1_fold * i_moe)
        ffn2 = self.arrays(i_moe, p.hidden_dim)
        ffn1_shared = self.arrays(p.hidden_dim, p.ffn1_fold * i_shared)
        ffn2_shared = self.arrays(i_shared, p.hidden_dim)
        return OrderedDict(
            (
                ("qkv", dense["qkv"]),
                ("o_proj", dense["o_proj"]),
                ("router", self.router_arrays()),
                ("ffn1_routed", p.num_experts * ffn1),
                ("ffn2_routed", p.num_experts * ffn2),
                ("ffn1_shared", p.n_shared_experts * ffn1_shared),
                ("ffn2_shared", p.n_shared_experts * ffn2_shared),
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

    def system_area_mm2(self) -> float:
        """Analog macro area plus the shared digital chiplet's area (D32)."""
        return self.total_area_mm2() + self.shared_digital_area_mm2()

    # ------------------------------------------------------------------
    # D32: digital area and power composed from the MEASURED library
    # ------------------------------------------------------------------

    @property
    def synthesis_technology(self) -> Optional[str]:
        """The measured library this run's digital cards price against, if any."""
        card = self.digital_card
        tech = str(getattr(card, "synthesis_library", "") or "").strip()
        return tech or None

    def synthesis_library(self) -> SynthesisLibrary:
        """This run's measured block library, or a refusal that names the knob."""
        tech = self.synthesis_technology
        if not tech:
            raise SynthesisLibraryError(
                f"shared digital chiplet card '{self.digital_card.name}' names no "
                "synthesis library: set cim.cards.<card>.synthesis_library to a "
                "technology that is checked in under configs/hardware-config "
                "(digital_components_<tech>.yaml). D32 composes digital area and power "
                "from MEASURED blocks; without a library the card can only report the "
                "declared area_mm2 placeholder it was given, and this model will not "
                "pretend the two are the same number."
            )
        return SynthesisLibrary.load(tech)

    def has_synthesis_library(self) -> bool:
        return self.synthesis_technology is not None

    def resolved_vector_lanes(self) -> Optional[int]:
        """The lane count if this run HAS one, else None — never a refusal.

        Composition asks a different question from pricing. Pricing a scan op
        without an engine is a gap and must refuse (:attr:`vector_lanes`).
        Composing the chiplet of a run that prices NO scan op at all is not a
        gap: the run demands no scan engine, so none is provisioned and none is
        composed. D31 derives the width from demand, and zero demand derives
        zero silicon.
        """
        if self._engine_probe:
            # Pass A's sentinel width is not silicon and must never be composed
            # or reported; while probing there is no engine yet, by definition.
            return None
        try:
            return int(self.vector_lanes)
        except EngineCapabilityError:
            return None

    def vector_engine_composition(
        self, lanes: Optional[int] = None
    ) -> EngineComposition:
        """The scan/vector engine of ONE shared digital chiplet, in blocks (D32)."""
        lanes = self.vector_lanes if lanes is None else int(lanes)
        return compose_vector_engine(
            self.synthesis_library(),
            lanes,
            count_provenance=(
                PROVENANCE_DECLARED_COUNT
                if self.digital_card.has_vector_engine
                else PROVENANCE_DERIVED_COUNT
            ),
        )

    def softmax_engine_composition(self) -> EngineComposition:
        """The softmax pipeline of ONE shared digital chiplet, in blocks (D32).

        ADJ-10: the WIDTH is the derived lane count when a derivation is
        installed, so the composed silicon is the silicon the softmax law was
        actually priced on. The per-lane census is unchanged.
        """
        return compose_softmax_engine(
            self.synthesis_library(),
            self.fabric_softmax_lanes,
            int(self.fabric.replicas),
            count_provenance=self.fabric_provenance,
        )

    def sa_fabric_composition(self) -> EngineComposition:
        """The attention systolic fabric of ONE shared digital chiplet (D32).

        ADJ-10: ``num_arrays`` is DERIVED (integer copies of the measured
        32x32 block); ``rows`` and ``cols`` stay declared, because array
        geometry is never invented (ADJ-4).
        """
        return compose_sa_fabric(
            self.synthesis_library(),
            int(self.fabric.rows),
            int(self.fabric.cols),
            self.fabric_num_arrays,
            int(self.fabric.replicas),
            count_provenance=PROVENANCE_DERIVED_COUNT,
        )

    def macro_pool_composition(
        self, sizing: Optional[DigitalPoolSizing] = None
    ) -> EngineComposition:
        """ONE analog macro's digital pool, in measured blocks (D12 x D32)."""
        sizing = self.digital_pool_sizing() if sizing is None else sizing
        return compose_macro_pool(self.synthesis_library(), sizing)

    def shared_digital_compositions(self) -> Tuple[EngineComposition, ...]:
        """Every engine ONE shared digital chiplet is made of (D32).

        The chiplet is the SA attention fabric plus its softmax pipeline plus
        the scan/vector engine — the three units this repo's laws time. Nothing
        else is composed, and the gap is named by
        :meth:`shared_digital_area_disclosures` rather than absorbed into a pad.
        """
        out = [self.sa_fabric_composition(), self.softmax_engine_composition()]
        lanes = self.resolved_vector_lanes()
        if lanes is not None:
            out.append(self.vector_engine_composition(lanes))
        return tuple(out)

    def shared_digital_area_mm2(self) -> float:
        """Shared digital chiplet area (mm2), COMPOSED when a library is named.

        D32 replaces the declared placeholder: when the card names a synthesis
        library, this is the sum of the measured block areas of the three
        engines the chiplet's laws time. A card that names none keeps returning
        its declared ``area_mm2`` exactly as before, so nothing moves under a
        config that did not ask for the library.
        """
        if not self.has_synthesis_library():
            return float(self.digital_card.area_mm2)
        return math.fsum(
            composition.area_mm2 for composition in self.shared_digital_compositions()
        )

    def shared_digital_power_w(self) -> float:
        """Shared digital chiplet power (W), composed from measured blocks (D32).

        There is no declared fallback: before D32 no card carried a power knob
        at all, so a card with no library has NO power number and says so with
        a refusal rather than a zero.
        """
        return math.fsum(
            composition.power_w for composition in self.shared_digital_compositions()
        )

    def shared_digital_area_disclosures(self) -> Tuple[str, ...]:
        """What the composed chiplet area does and does not include (D21/D32)."""
        if not self.has_synthesis_library():
            return (
                f"shared digital chiplet area is the DECLARED "
                f"{float(self.digital_card.area_mm2):.6g} mm2 placeholder: card "
                f"'{self.digital_card.name}' names no synthesis_library, so D32's "
                "measured composition did not run and no power figure exists for it.",
            )
        library = self.synthesis_library()
        notes = [library.provenance_line()]
        notes.extend(self.fabric_sizing_disclosures())
        for composition in self.shared_digital_compositions():
            notes.extend(composition.disclosures)
        if self.resolved_vector_lanes() is None:
            notes.append(
                "NO scan/vector engine is composed into this chiplet: the run prices no "
                "SSM scan, delta rule or RG-LRU op, so it demands none and D31 derives "
                "none. The chiplet area and power below are the SA fabric and the "
                "softmax pipeline only, which is a statement about this WORKLOAD, not "
                "about the card."
            )
        notes.append(
            "the composed chiplet is the SA fabric + softmax pipeline + scan/vector "
            "engine, which are the three units this repo declares laws for. Its "
            "activation SRAM, its interconnect and its control are NOT composed — this "
            "repo declares no law that says how many of anything they need — so the "
            "composed area is a LOWER BOUND on the chiplet, named here rather than "
            "padded (D28)."
        )
        notes.append(
            "TWO MEASURED BLOCKS THE LIBRARY HOLDS AND NO COMPOSITION USES: M_SOFTPLUS "
            "and ELASTIC_BUFFER_256_MXFP4_ELEMS. The scan lane composes no "
            "transcendental unit because the scan work law counts no transcendental op "
            "(ssm_recurrent_scan_work prices exp(dt A) as a plain multiply), so the "
            "engine SIZED is exactly the engine PRICED — and the composed scan area is "
            "therefore a FLOOR for a model that really runs exp/softplus. No elastic "
            "buffering between stages is modelled either. Both are absences of a LAW, "
            "not measured zeros, unlike COMBINED_CONV, COMBINED_CONV_SEQUENTIAL and "
            "INT8_BUFF_DIV_SYS_ARRAY_PENALTY, which the source measures at zero."
        )
        if float(self.digital_card.area_mm2) > 0:
            notes.append(
                f"card '{self.digital_card.name}' also declares area_mm2 = "
                f"{float(self.digital_card.area_mm2):.6g} mm2. The COMPOSED area wins "
                "(D32) and the declared number is not added to it: one accounting per "
                "metric (D21)."
            )
        seen: List[str] = []
        for note in notes:
            if note not in seen:
                seen.append(note)
        return tuple(seen)

    def report_digital_composition(self) -> str:
        """The D32 line: every engine of the shared chiplet, block by block."""
        if not self.has_synthesis_library():
            return (
                "[FWS-CIM] digital area (D32): card "
                f"'{self.digital_card.name}' names no synthesis_library, so the "
                f"{float(self.digital_card.area_mm2):.6g} mm2 it reports is the DECLARED "
                "placeholder and no power figure exists."
            )
        lines = [
            "[FWS-CIM] shared digital chiplet, composed from measured synthesis (D32): "
            f"{self.shared_digital_area_mm2():.6g} mm2, "
            f"{self.shared_digital_power_w():.6g} W"
        ]
        for composition in self.shared_digital_compositions():
            lines.append(composition.report())
        for note in self.shared_digital_area_disclosures():
            lines.append(f"  [NOTE] {note}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # ADJ-10: derive the ATTENTION FABRIC to the ANALOG FLOOR
    # ------------------------------------------------------------------

    def attention_cycles_at(
        self,
        call: "AttentionCallDemand",
        num_arrays: int,
        softmax_width: int,
    ) -> Tuple[int, int, int, int, int]:
        """One attention call's cycles at a candidate width — THE PRICING LAW.

        Returns ``(qk, pv, softmax, a_qk, a_pv)`` computed with exactly the
        expressions :meth:`attention_call_timing` uses, so a width this method
        accepts is a width the run's own laws will reproduce. The DAG places
        the three ops in SERIES (``attention_op_folding``), so the stage cost
        the derivation compares against a target is
        ``qk + fill_drain + softmax + pv``.
        """
        folds = max(1, int(call.folds))
        a_qk, a_pv = fold_group_split(
            num_arrays,
            lambda arrays: self.sa_cycles(
                call.m, call.n, int(call.k) * _ceil_div(folds, arrays)
            ),
            lambda arrays: self.sa_cycles(
                call.m, call.k, int(call.n) * _ceil_div(folds, arrays)
            ),
        )
        qk = self.sa_cycles(call.m, call.n, int(call.k) * _ceil_div(folds, a_qk))
        pv = self.sa_cycles(call.m, call.k, int(call.n) * _ceil_div(folds, a_pv))
        softmax = (
            int(self.fabric.softmax_pipeline_depth)
            + _ceil_div(
                int(call.softmax_tokens) * int(call.heads_chip),
                max(1, int(softmax_width)),
            )
            - 1
        )
        return int(qk), int(pv), int(softmax), int(a_qk), int(a_pv)

    def _stage_attention_cycles(
        self,
        calls: Sequence["AttentionCallDemand"],
        num_arrays: int,
        softmax_width: int,
    ) -> Tuple[int, int, int, int, int, int]:
        """A stage's whole attention bill at a candidate width.

        ``(total, qk, pv, softmax, a_qk, a_pv)``. The calls of one stage run on
        ONE chiplet (the lowering assigns a chip's fabric ops to
        ``engines[chip_id % len(engines)]``) and decode runs a stage's layers
        in sequence, so they SUM. A stage plan that spread one stage over
        several chips would have its attention on several chiplets and this sum
        would then be an upper bound; that case is named here rather than
        assumed away.

        ``a_qk``/``a_pv`` are the LAST call's partition, and they are a report
        field only — the cycle totals above are each call's own. A stage's
        calls are its attention LAYERS, which share head count, head dim and
        context inside one beat, so the partition is the same for all of them
        on every machine this repo ships; a stage whose layers really differed
        would still be priced call by call and only this one reported field
        would speak for the last of them.
        """
        total = qk_t = pv_t = sm_t = 0
        a_qk = a_pv = 1
        for call in calls:
            qk, pv, softmax, a_qk, a_pv = self.attention_cycles_at(
                call, num_arrays, softmax_width
            )
            total += qk + pv + softmax + int(call.fill_drain_cycles)
            qk_t += qk
            pv_t += pv
            sm_t += softmax
        return total, qk_t, pv_t, sm_t, a_qk, a_pv

    def derive_fabric_sizing(
        self,
        analog_beat_s: float,
        demand: Sequence["FabricDemand"],
        *,
        compose: Optional[bool] = None,
    ) -> "DerivedFabricSizing":
        """Size the ATTENTION FABRIC up until the ANALOG m-pass binds (ADJ-10).

        THE CRITERION, EXACTLY. For every stage: *the stage's attention time
        (qk + fill/drain + softmax + pv, summed over the stage's calls) <= that
        stage's OWN measured ANALOG m-pass time*. Identical in shape to ADJ-9's
        scan criterion and taken against the identical measurement — the union
        of the busy intervals of the stage's analog macros in the probe beat —
        so the two derivations are one accounting asked of two engines (D21).

        WHAT IS DERIVED AND WHAT IS NOT. ``num_arrays`` is derived, in INTEGER
        COPIES of the measured ``GEMMINI_SYS_ARRAY`` block, and so is
        ``softmax_lanes`` in copies of the softmax pipeline's measured per-lane
        census. ``rows`` and ``cols`` per array are NOT derived and never will
        be: the block's geometry is a measurement, and a fabric of some other
        shape would be an invented number (ADJ-4). This is the whole difference
        between "compose more of what was measured" and "invent silicon".

        THE LADDER, AND WHY IT IS EXACT RATHER THAN GREEDY. Both widths
        SATURATE — arrays at ``2 * folds`` (one fold per array in each group)
        and softmax at ``softmax_tokens * heads_chip`` (one element per lane) —
        so the candidate lattice is finite and small, and the search enumerates
        it. The order is LEXICOGRAPHIC, arrays first: a ``GEMMINI_SYS_ARRAY``
        is the largest block in the library by three orders of magnitude and
        the two systolic runs are the dominant terms, so buying softmax lanes
        to avoid buying arrays would be buying the wrong silicon. Within that
        order both answers are the SMALLEST width that meets the target, so
        there is no margin (D28).

        WHERE THE ANALOG FLOOR IS NOT REACHABLE, IT SATURATES AND SAYS SO. Fold
        concurrency runs out: past ``2 * folds`` another copy of the measured
        block carries no fold, and what is left is
        ``sa_cycles(m, n, k) + sa_cycles(m, k, n)`` — the DECLARED
        ``rows x cols`` geometry's own floor, made of the ``ceil(n / cols)``
        column passes and the ``k + rows + cols - 2`` pipeline term. Those
        stages are counted in :attr:`DerivedFabricSizing.saturated_stages`,
        named by ``target_kind``, and given the SATURATION width — the smallest
        width past which no width helps. They are not clamped to a narrower
        fabric and no margin is added to a wider one.

        ONE CARD, ONE FABRIC: the provisioned width is the maximum over the
        stages, every stage is then RE-PRICED on it, and the slack the
        non-binding stages run with is printed rather than smoothed (D28).
        """
        clock = self.f_fabric_hz
        beat_s = float(analog_beat_s)
        ordered = sorted(demand, key=lambda item: int(item.stage))
        ordered = [entry for entry in ordered if entry.calls]
        if not ordered:
            raise FabricSizingError(
                "derive_fabric_sizing was given no per-stage attention demand. ADJ-10 "
                "sizes the fabric from the attention a stage runs in a beat; with no "
                "call there is nothing to size and a width would be an invention."
            )
        rows: List[StageFabricSizing] = []
        for entry in ordered:
            analog_s = max(0.0, float(entry.analog_time_s or 0.0))
            target_s, target_kind = analog_s, TARGET_FABRIC_ANALOG_STAGE
            if analog_s <= 0:
                target_s, target_kind = beat_s, TARGET_FABRIC_NO_ANALOG_WORK
            sat_arrays = max(2, int(entry.saturation_arrays))
            sat_width = max(1, int(entry.saturation_softmax_width))
            budget = int(math.floor(max(0.0, target_s) * clock))
            floor_cycles = self._stage_attention_cycles(
                entry.calls, sat_arrays, sat_width
            )[0]
            if floor_cycles > budget:
                arrays, width = sat_arrays, sat_width
                target_kind = TARGET_FABRIC_SATURATED
            else:
                arrays = next(
                    candidate
                    for candidate in range(2, sat_arrays + 1)
                    if self._stage_attention_cycles(
                        entry.calls, candidate, sat_width
                    )[0]
                    <= budget
                )
                width = next(
                    candidate
                    for candidate in range(1, sat_width + 1)
                    if self._stage_attention_cycles(entry.calls, arrays, candidate)[0]
                    <= budget
                )
            total, qk, pv, softmax, a_qk, a_pv = self._stage_attention_cycles(
                entry.calls, arrays, width
            )
            fill = sum(int(call.fill_drain_cycles) for call in entry.calls)
            rows.append(
                StageFabricSizing(
                    stage=int(entry.stage),
                    calls=len(entry.calls),
                    num_arrays=int(arrays),
                    qk_arrays=int(a_qk),
                    pv_arrays=int(a_pv),
                    softmax_width=int(width),
                    qk_cycles=int(qk),
                    pv_cycles=int(pv),
                    softmax_cycles=int(softmax),
                    fill_drain_cycles=int(fill),
                    used_cycles=int(total),
                    budget_cycles=budget,
                    time_s=total / clock,
                    slack_s=target_s - total / clock,
                    analog_time_s=analog_s,
                    target_s=target_s,
                    target_kind=target_kind,
                    floor_time_s=floor_cycles / clock,
                )
            )
        replicas = max(1, int(self.fabric.replicas))
        provisioned = max(int(row.num_arrays) for row in rows)
        width = max(int(row.softmax_width) for row in rows)
        lanes = _ceil_div(width, replicas)
        binding = max(
            rows, key=lambda row: (int(row.num_arrays), int(row.used_cycles))
        ).stage
        # Re-price every stage on the PROVISIONED fabric: one card carries one
        # width, so a stage that asked for less actually runs on the wider
        # fabric and finishes early. The slack printed is the slack of the
        # machine that gets built.
        repriced: List[StageFabricSizing] = []
        for row, entry in zip(rows, ordered):
            total, qk, pv, softmax, a_qk, a_pv = self._stage_attention_cycles(
                entry.calls, provisioned, lanes * replicas
            )
            repriced.append(
                StageFabricSizing(
                    stage=row.stage,
                    calls=row.calls,
                    num_arrays=provisioned,
                    qk_arrays=int(a_qk),
                    pv_arrays=int(a_pv),
                    softmax_width=lanes * replicas,
                    qk_cycles=int(qk),
                    pv_cycles=int(pv),
                    softmax_cycles=int(softmax),
                    fill_drain_cycles=row.fill_drain_cycles,
                    used_cycles=int(total),
                    budget_cycles=row.budget_cycles,
                    time_s=total / clock,
                    slack_s=row.target_s - total / clock,
                    analog_time_s=row.analog_time_s,
                    target_s=row.target_s,
                    target_kind=row.target_kind,
                    floor_time_s=row.floor_time_s,
                )
            )
        rows = repriced
        composition = softmax_composition = None
        want = self.has_synthesis_library() if compose is None else bool(compose)
        if want:
            composition = compose_sa_fabric(
                self.synthesis_library(),
                int(self.fabric.rows),
                int(self.fabric.cols),
                provisioned,
                replicas,
                count_provenance=PROVENANCE_DERIVED_COUNT,
            )
            softmax_composition = compose_softmax_engine(
                self.synthesis_library(),
                lanes,
                replicas,
                count_provenance=PROVENANCE_DERIVED_COUNT,
            )
        analog_bound = [row for row in rows if row.analog_bound]
        saturated = [row for row in rows if row.target_kind == TARGET_FABRIC_SATURATED]
        declared_arrays = int(self.fabric.num_arrays)
        declared_lanes = int(self.fabric.softmax_lanes)
        disclosures = [
            "ADJ-10: cim.fabric.num_arrays and cim.fabric.softmax_lanes are no longer "
            "declared design points on a mapped run. Both are DERIVED here as the "
            "smallest integer counts of MEASURED blocks for which every stage's "
            "attention time fits that stage's own ANALOG m-pass time, and both are "
            f"reported ({declared_arrays} -> {provisioned} array(s), "
            f"{declared_lanes} -> {lanes} softmax lane(s) per replica). Only the COUNT "
            f"is derived: the array stays the measured {int(self.fabric.rows)} x "
            f"{int(self.fabric.cols)} geometry, because inventing array geometry is "
            "refused (ADJ-4).",
            "FOLD SEMANTICS (the law ADJ-10 extends). The arrays are partitioned into a "
            "QK group and a PV group and the call's folds — heads_per_replica x streams "
            "— are dealt across the group, so an array in a group of a carries "
            "ceil(folds / a) of them and sees a contraction dim of that many folds "
            "instead of all of them. At num_arrays = 2 the split is (1, 1) and the law "
            "is the pass-1 law bit for bit, which is why no OPTIMA parity number moves.",
            f"ANALOG-BOUND BY CONSTRUCTION: {len(analog_bound)} of {len(rows)} stage(s) "
            "run with the analog m-pass longer than the fabric's own attention time"
            + (
                ". FOLD CONCURRENCY SATURATES on stage(s) "
                + ", ".join(
                    f"{int(row.stage)} (floor {row.floor_time_s:.6g} s against a "
                    f"{row.analog_time_s:.6g} s analog m-pass)"
                    for row in saturated
                )
                + ": past 2 x folds arrays another copy of the measured block carries no "
                "fold and buys no time, so the residue is the DECLARED rows x cols "
                "geometry — ceil(n / cols) column passes and the k + rows + cols - 2 "
                "pipeline term — which integer copies cannot shorten. Those stages take "
                "the SATURATION width and the residue is NAMED, never clamped and never "
                "padded."
                if saturated
                else ", and there is no stage the criterion could not reach."
            ),
            "WHY SATURATION RATHER THAN ADJ-9'S BEAT FALLBACK. ADJ-9 hands an "
            "unreachable scan stage the analog BEAT as its budget, which is right there: "
            "its unreachable cases are a stage with no analog work at all and a stage "
            "under the engine's own pipeline fill, and neither has a floor to walk to. "
            "The fabric has one. Falling back to the wider beat budget here would derive "
            "a NARROWER fabric than the machine can use and leave measured throughput on "
            "the table for silicon that buys time, which is the trade ADJ-9's own "
            "rationale rejects.",
            "the derivation inverts the pricing law exactly — same sa_cycles, same "
            "fold split, same softmax law, same serialization of the three lowered ops "
            "— so the fabric sized here is the fabric the run is then priced on. No "
            "margin is added (D28).",
            "one card, one fabric: the provisioned width is the maximum over the "
            "stages, so every non-binding stage runs with the slack printed beside it. "
            "That slack is idle silicon and D28 requires it to be visible.",
            "THE P3 SLOT CENSUS IS NOT THIS NUMBER. fws_mapping enumerates "
            "num_arrays x replicas engine SLOTS per shared chiplet from the DECLARED "
            f"geometry ({declared_arrays} x {replicas}) when it PLACES, which is before "
            "any beat has been measured. Those slots hold no tile, carry no device and "
            "price no time — the chiplet is ONE device with one queue — so the "
            "derivation does not move them. The silicon, the power and the timing all "
            "use the derived width; the placement record is a placement record, and the "
            "two are reconciled here rather than left for a reader to notice (D21).",
        ]
        return DerivedFabricSizing(
            analog_beat_s=beat_s,
            clock_hz=clock,
            rows=int(self.fabric.rows),
            cols=int(self.fabric.cols),
            replicas=replicas,
            num_arrays=provisioned,
            softmax_lanes=lanes,
            declared_num_arrays=declared_arrays,
            declared_softmax_lanes=declared_lanes,
            binding_stage=int(binding),
            per_stage=tuple(rows),
            composition=composition,
            softmax_composition=softmax_composition,
            basis=(
                "smallest integer array count (then softmax lane count) whose priced "
                "attention time — qk + fill/drain + softmax + pv, the three ops the "
                "lowering serializes — fits the stage's own measured ANALOG m-pass "
                f"time (ADJ-10); binding stage {int(binding)}, "
                f"{int(self.fabric.rows)}x{int(self.fabric.cols)} arrays at "
                f"{clock / 1e9:.4g} GHz over {len(rows)} stage(s), "
                f"{len(analog_bound)} of them analog-bound"
                + (
                    f"; fold concurrency saturated on {len(saturated)}"
                    if saturated
                    else ""
                )
            ),
            disclosures=tuple(disclosures),
        )

    # ------------------------------------------------------------------
    # D31-v2 (ADJ-9): derive the engine width to the ANALOG FLOOR
    # ------------------------------------------------------------------

    def derive_engine_sizing(
        self,
        analog_beat_s: float,
        demand: Sequence[EngineDemand],
        *,
        compose: Optional[bool] = None,
    ) -> DerivedEngineSizing:
        """Size the engine UP until the ANALOG m-pass binds, and report it (ADJ-9).

        ``analog_beat_s`` is the beat the probe pass measured with every vector
        op counted and untimed. Under D31-v2 it is no longer the sizing target:
        it is the FALLBACK budget for a stage whose own analog time cannot
        serve as one, and the number the report prints beside the derivation so
        a reader can see both.

        ``demand`` is one :class:`EngineDemand` per pipeline stage: the
        scalar-op count of each vector call the stage runs in ONE beat, plus
        ``analog_time_s`` — the wall time that stage's own analog macros spend
        passing weights in that beat.

        WHAT THE CRITERION IS, EXACTLY (ADJ-9, D31-v2). For every stage:
        *digital per-stage time <= that stage's ANALOG m-pass time*. The width
        is the smallest integer lane count that satisfies it — smallest for
        that target, so there is no margin (D28), and integer because a lane is
        silicon. Because the target is the analog time rather than the beat,
        the answer is a DERIVE-UP: the engine grows until the analog side is
        the longer of the two terms in every stage it can be, which is what
        makes the analog floor the thing that sets the machine's pace.

        WHAT IT REPLACED, AND WHY. The old criterion was *digital per-beat <=
        the analog BEAT*. The beat is the SLOWEST stage's time, so a stage that
        was faster than the beat got an engine sized against somebody else's
        stage and became its own binding term — the machine was left
        digital-bound by a minimal derivation. ADJ-9's rationale: against the
        Invariant-W analog floor, digital lanes are nearly free (the audit
        measured +27% tokens/s for +0.78% silicon on Granite), so a derivation
        that leaves throughput on the table to save lanes is the wrong trade.

        WHERE THE CRITERION IS NOT REACHABLE, IT SAYS SO. Two cases, both named
        in the per-stage rows (``target_kind``) and in the disclosures:

        * a stage with NO analog work in the beat has no analog time to be
          bound by, at any width;
        * a stage whose analog time is shorter than the engine's own pipeline
          FILL (``calls * depth`` cycles) cannot be held by any width either,
          because lanes shorten the streaming term and never the fill.

        Both fall back to the analog BEAT, which is the widest budget this
        derivation ever uses, and both are counted in
        :attr:`DerivedEngineSizing.unreachable_stages`. Nothing is clamped and
        nothing is padded.

        THE DERIVATION IS THE PRICING LAW, INVERTED. For a candidate width the
        cost of a stage's beat is exactly what :meth:`price_vector_work` would
        charge — ``sum over calls of (depth + ceil(ops/lanes) - 1)`` — so an
        engine this method sizes is one the run's own laws will actually use.

        ONE CARD, ONE WIDTH. The chiplet is a single card, so the provisioned
        width is the maximum over the stages and the per-stage rows report the
        slack the others run with — which is exactly the idle silicon D28 says
        must stay visible.
        """
        clock = self.vector_clock_hz
        depth = self.vector_pipeline_depth
        beat_s = float(analog_beat_s)
        ordered = sorted(demand, key=lambda item: int(item.stage))
        rows: List[StageEngineSizing] = []
        for entry in ordered:
            calls = [float(call) for call in entry.ops if float(call) > 0]
            analog_s = max(0.0, float(getattr(entry, "analog_time_s", 0.0) or 0.0))
            # ADJ-9: the stage's OWN analog m-pass time is the target. Where it
            # cannot be one, the analog beat is, and the row says which.
            lanes = None
            target_s, target_kind = analog_s, TARGET_ANALOG_STAGE
            if analog_s <= 0:
                target_s, target_kind = beat_s, TARGET_NO_ANALOG_WORK
            else:
                try:
                    lanes = derive_vector_lanes(calls, analog_s, clock, depth)
                except EngineSizingError:
                    # No width holds this stage's analog m-pass, at any lane
                    # count: the beat is the fallback and the row says which.
                    target_s, target_kind = beat_s, TARGET_ANALOG_BELOW_FILL
            if lanes is None:
                lanes = derive_vector_lanes(calls, target_s, clock, depth)
            budget = int(math.floor(target_s * clock))
            used = vector_cycles_at(calls, lanes, depth)
            rows.append(
                StageEngineSizing(
                    stage=int(entry.stage),
                    calls=len(calls),
                    ops=tuple(calls),
                    total_ops=math.fsum(calls),
                    lanes=int(lanes),
                    budget_cycles=budget,
                    fill_cycles=len(calls) * (depth - 1),
                    stream_cycles=used - len(calls) * (depth - 1),
                    used_cycles=used,
                    time_s=used / clock,
                    slack_s=target_s - used / clock,
                    analog_time_s=analog_s,
                    target_s=target_s,
                    target_kind=target_kind,
                )
            )
        if not rows:
            raise EngineSizingError(
                "derive_engine_sizing was given no per-stage demand. D31 sizes the "
                "engine from the work a stage does in a beat; with no demand there is "
                "nothing to size and a width would be an invention, not a derivation."
            )
        provisioned = max(int(row.lanes) for row in rows)
        binding = max(rows, key=lambda row: (int(row.lanes), row.total_ops)).stage
        # Re-price every stage at the PROVISIONED width. One card carries one
        # width, so a stage whose own derivation asked for fewer lanes actually
        # runs on the wider engine and finishes early: its reported slack is the
        # slack of the machine that gets built, not of the one it asked for.
        repriced: List[StageEngineSizing] = []
        for row, entry in zip(rows, ordered):
            calls = [float(call) for call in entry.ops if float(call) > 0]
            used = vector_cycles_at(calls, provisioned, depth)
            fill = len(calls) * (depth - 1)
            repriced.append(
                StageEngineSizing(
                    stage=row.stage,
                    calls=row.calls,
                    ops=tuple(calls),
                    total_ops=row.total_ops,
                    lanes=provisioned,
                    budget_cycles=row.budget_cycles,
                    fill_cycles=fill,
                    stream_cycles=used - fill,
                    used_cycles=used,
                    time_s=used / clock,
                    slack_s=row.target_s - used / clock,
                    analog_time_s=row.analog_time_s,
                    target_s=row.target_s,
                    target_kind=row.target_kind,
                )
            )
        rows = repriced
        composition = None
        want_composition = (
            self.has_synthesis_library() if compose is None else bool(compose)
        )
        if want_composition:
            composition = compose_vector_engine(
                self.synthesis_library(),
                provisioned,
                count_provenance=PROVENANCE_DERIVED_COUNT,
            )
        analog_bound = [row for row in rows if row.analog_bound]
        unreachable = [row for row in rows if row.target_kind != TARGET_ANALOG_STAGE]
        disclosures = [
            "D31-v2 (ADJ-9): the scan/vector engine is not a swept axis and not a "
            "declared knob, and it is no longer sized to the analog BEAT. Its width is "
            "the smallest integer for which EVERY stage's digital per-stage time fits "
            "that stage's OWN analog m-pass time, so the analog side is the binding "
            "term by construction wherever it can be. The width is reported.",
            f"ANALOG-BOUND BY CONSTRUCTION: {len(analog_bound)} of {len(rows)} stage(s) "
            "run with the analog m-pass longer than the derived engine's own per-stage "
            "time"
            + (
                ". The criterion is UNREACHABLE on stage(s) "
                + ", ".join(
                    f"{int(row.stage)} ({row.target_kind})" for row in unreachable
                )
                + ", which fall back to the analog beat "
                f"({beat_s:.6g} s) — a stage with no analog work has no analog time to "
                "be bound by, and a stage whose analog time is shorter than the "
                "engine's own pipeline fill cannot be held at any width because lanes "
                "shorten the streaming term and never the fill. Neither is clamped."
                if unreachable
                else ", and there is no stage the criterion could not reach."
            ),
            "one card, one width: the provisioned width is the maximum over the stages, "
            "so every non-binding stage runs with the slack printed beside it. That "
            "slack is idle silicon and D28 requires it to be visible, not smoothed.",
            "the derivation inverts CimDeviceModel.price_vector_work exactly (same "
            "pipeline depth, same ceil, same per-call fill), so the engine it sizes is "
            "the engine the run then prices. No margin is added (D28).",
            (
                "THE CONDITION ON THAT INVERSION: price_vector_work returns "
                "max(arithmetic time, state time), and only the ARITHMETIC term is "
                "inverted here. state time = ceil(state_bytes / state_bytes_per_cycle) "
                "/ clock and lanes do not shorten it, so a card that declares "
                "state_bytes_per_cycle could have a state term the derived width cannot "
                "hold — no lane count fixes it, and this method would not refuse it the "
                "way it refuses a beat shorter than the pipeline fill. "
                + (
                    f"This card declares state_bytes_per_cycle = "
                    f"{float(self.digital_card.state_bytes_per_cycle):.6g}, so the state "
                    "term is LIVE and the inversion above is exact only for the "
                    "arithmetic half."
                    if float(self.digital_card.state_bytes_per_cycle) > 0
                    else "This card declares no state_bytes_per_cycle, so the state term "
                    "is 0 and the inversion is exact."
                )
            ),
            "the criterion is digital-per-stage <= that stage's own ANALOG m-pass time, "
            "so the analog side sets each stage's pace and the engine never does. It "
            "does NOT make the MEASURED beat equal the analog time and cannot: a D29 "
            "stage holds one stream at a time, so its analog passes and its scan run in "
            "SERIES and the beat is their sum at any width. What ADJ-9 buys is that the "
            "sum is dominated by the analog half. A stage's measured time can still be "
            "set by a term THIS derivation does not size. ADJ-10 closed the two that "
            "used to be declared — the attention fabric's array count and the softmax "
            "pipeline's lane count now derive against this same per-stage target — so "
            "what is left is the array GEOMETRY (rows x cols, the shape the systolic "
            "law was validated at) and the terms that are not engines at all. Which one "
            "actually binds the machine that gets BUILT is measured and named under its "
            "own name, evaluation.digital_silicon.binding_term, and no lane count moves "
            "it.",
        ]
        return DerivedEngineSizing(
            analog_beat_s=beat_s,
            clock_hz=clock,
            pipeline_depth=depth,
            vector_lanes=provisioned,
            binding_stage=int(binding),
            per_stage=tuple(rows),
            composition=composition,
            basis=(
                "smallest integer lanes with sum(depth + ceil(ops/lanes) - 1) <= "
                "floor(target x clock) for EVERY stage, where the target is the "
                "stage's own measured ANALOG m-pass time (ADJ-9/D31-v2); binding stage "
                f"{int(binding)} at {int(next(int(r.budget_cycles) for r in rows if r.stage == binding))} "
                f"cycles of budget, {clock / 1e9:.4g} GHz, depth {depth}, over "
                f"{len(rows)} stage(s), {len(analog_bound)} of them analog-bound"
                + (
                    f"; {len(unreachable)} fell back to the {beat_s:.6g} s analog beat"
                    if unreachable
                    else ""
                )
            ),
            disclosures=tuple(disclosures),
        )

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
        """(owner, site) pairs the user wrote in `cim.allocation.assignments`.

        The row range is a PLACEHOLDER, not a claim: a config entry declares a
        macro and a set of mux slots and says nothing about rows. Only
        :meth:`_check_sites` consumes these pairs and it reads column sets
        alone. See :class:`TileSite` for the pinned row convention and for how
        a user allocation actually reaches a placed tile.
        """
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
    # Dense packing: Invariant W in code (P7.2, D27)
    # ------------------------------------------------------------------

    def bank_columns(self, column_sets_per_tile: Optional[int] = None) -> int:
        """Stored weight columns in one allocatable BANK: ``cols_adc * bank``."""
        bank = self._packing_bank(column_sets_per_tile)
        return int(self.card.stored_columns_per_set) * bank

    def _packing_bank(self, column_sets_per_tile: Optional[int]) -> int:
        """The bank size the packer allocates in, validated against the card."""
        mux = int(self.card.column_sets_per_macro)
        bank = (
            self.column_sets_per_tile
            if column_sets_per_tile is None
            else int(column_sets_per_tile)
        )
        if bank < 1 or mux % bank != 0:
            raise ValueError(
                f"column_sets_per_tile = {bank} must be >= 1 and divide the card's mux "
                f"= {mux}: the mux slot is the smallest allocatable unit (ADJ-4)."
            )
        return bank

    def dense_blocks(
        self,
        k: int,
        n: int,
        *,
        column_sets_per_tile: Optional[int] = None,
        walk: str = CANONICAL_WALK,
    ) -> Tuple[Tuple[int, int, int, int], ...]:
        """One tensor's (K, N) blocks in the CANONICAL walk order.

        Returns half-open ``(k_start, k_end, n_start, n_end)`` quadruples. A
        block is one bank's worth of weights: ``rows`` tall and
        ``cols_adc * bank`` wide, clipped to the tensor at the far edge — the
        clip is the dimension-mismatch remainder Invariant W reports.

        K-INNER (D26): the K blocks of one output block come out CONSECUTIVELY,
        so ``dense_pack`` lands them in one macro whenever they fit and their
        partial sums never cross a link.
        """
        if walk != CANONICAL_WALK:
            refuse_dead_fold("non_canonical_walk", context=f"dense_blocks(walk={walk!r})")
        k = int(k)
        n = int(n)
        if k < 1 or n < 1:
            raise ValueError(
                f"a weight matrix needs K >= 1 and N >= 1 (got K x N = {k} x {n}): a "
                "tensor with an empty dimension holds no weights and cannot be packed."
            )
        rows = int(self.card.params.rows)
        width = self.bank_columns(column_sets_per_tile)
        blocks = []
        for cb in range(_ceil_div(n, width)):
            for rb in range(_ceil_div(k, rows)):
                blocks.append(
                    (
                        rb * rows,
                        min(k, (rb + 1) * rows),
                        cb * width,
                        min(n, (cb + 1) * width),
                    )
                )
        return tuple(blocks)

    def dense_pack(
        self,
        requests: Sequence["PackRequest"],
        *,
        first_macro_id: int = 0,
        column_sets_per_tile: Optional[int] = None,
        walk: str = CANONICAL_WALK,
    ) -> DensePacking:
        """Pack every tensor's blocks into a CONTIGUOUS bank stream (D27).

        Invariant W in one sentence: bank ``i`` of the stream goes to macro
        ``first_macro_id + i // banks_per_macro``, slot ``i % banks_per_macro``,
        with no gap and no per-tensor rounding — a tensor that ends mid-macro is
        followed by the next tensor's first block in the very next bank, so a
        bank idles only when the stream itself runs out.

        The order is the caller's: requests are packed as given, which keeps a
        layer's tensors adjacent (and therefore on one chip) because that is the
        order a mapper offers them in. The packer reorders nothing — a packer
        that shuffled tensors to save a remainder would be choosing the mapping,
        which is the optimizer's job and not this law's.

        Refusals (D26): a non-canonical walk, and a slicing card — slicing x
        folding is on the DOA register, and a card that slices places its slice
        groups through :meth:`enumerate_tiles`.
        """
        if walk != CANONICAL_WALK:
            refuse_dead_fold("non_canonical_walk", context=f"dense_pack(walk={walk!r})")
        if self.n_slices > 1:
            refuse_dead_fold(
                "slicing_x_folding",
                context=(
                    f"dense_pack on a card declaring {self.n_slices} bit slices "
                    f"(bits_per_cell {self.card.bits_per_cell} < weight_bits "
                    f"{self.card.weight_bits})"
                ),
            )
        card = self.card
        rows = int(card.params.rows)
        bank = self._packing_bank(column_sets_per_tile)
        mux = int(card.column_sets_per_macro)
        width = int(card.stored_columns_per_set) * bank
        banks_per_macro = mux // bank
        cells_per_bank = rows * width
        cells_per_macro = rows * int(card.stored_columns_per_set) * mux

        tiles: List[Tile] = []
        stacks: List[KStack] = []
        fills: "OrderedDict[int, List[object]]" = OrderedDict()
        real_cells = 0
        index = 0
        for request in requests:
            blocks = self.dense_blocks(
                request.k, request.n, column_sets_per_tile=bank, walk=walk
            )
            by_column: "OrderedDict[int, List[int]]" = OrderedDict()
            for (k_start, k_end, n_start, n_end) in blocks:
                macro_id = first_macro_id + index // banks_per_macro
                slot = index % banks_per_macro
                index += 1
                tiles.append(
                    Tile(
                        owner=request.owner,
                        k_start=k_start,
                        k_end=k_end,
                        n_start=n_start,
                        n_end=n_end,
                        slice_index=0,
                        site=TileSite(
                            macro_id=macro_id,
                            row_start=k_start,
                            row_end=k_end,
                            column_sets=tuple(range(slot * bank, (slot + 1) * bank)),
                        ),
                    )
                )
                cells = (k_end - k_start) * (n_end - n_start)
                real_cells += cells
                entry = fills.setdefault(macro_id, [0, 0, OrderedDict()])
                entry[0] += 1
                entry[1] += cells
                entry[2].setdefault(request.owner, None)
                by_column.setdefault(n_start, []).append(macro_id)
            for n_start, macro_ids in by_column.items():
                n_end = min(int(request.n), n_start + width)
                stacks.append(
                    KStack(
                        owner=request.owner,
                        n_start=n_start,
                        n_end=n_end,
                        depth=len(macro_ids),
                        macro_ids=tuple(macro_ids),
                        sink_macro_id=macro_ids[0],
                    )
                )
        banks_used = index
        macro_count = _ceil_div(banks_used, banks_per_macro)
        block_cells = banks_used * cells_per_bank
        committed_cells = macro_count * cells_per_macro
        macro_fills = tuple(
            MacroFill(
                macro_id=macro_id,
                banks_used=int(entry[0]),
                banks_total=banks_per_macro,
                real_cells=int(entry[1]),
                committed_cells=cells_per_macro,
                owners=tuple(entry[2]),
            )
            for macro_id, entry in fills.items()
        )
        disclosures: List[str] = []
        if bank == mux:
            disclosures.append(
                f"the card admits WHOLE-MACRO allocation only (bank_depth = {mux} mux "
                "slots), so the dense packer allocates whole macros: every tensor still "
                "starts on a fresh macro and this packing is the dedicated placement, "
                "block for block. The reported waste is then the card's granularity "
                "speaking, not the packer's — a card declaring a smaller bank_depth is "
                "what lets Invariant W actually fill the remainder."
            )
        if macro_count and banks_used % banks_per_macro:
            disclosures.append(
                f"the last macro holds {banks_used % banks_per_macro} of "
                f"{banks_per_macro} banks: the stream ran out, so those banks hold no "
                "weights. They are counted in tail_cells, never rounded away."
            )
        return DensePacking(
            tiles=tuple(tiles),
            macro_fills=macro_fills,
            k_stacks=tuple(stacks),
            walk=walk,
            first_macro_id=int(first_macro_id),
            macro_count=macro_count,
            banks_used=banks_used,
            banks_per_macro=banks_per_macro,
            column_sets_per_bank=bank,
            cells_per_bank=cells_per_bank,
            cells_per_macro=cells_per_macro,
            real_cells=real_cells,
            block_cells=block_cells,
            committed_cells=committed_cells,
            remainder_cells=block_cells - real_cells,
            tail_cells=committed_cells - block_cells,
            cell_floor_macros=_ceil_div(real_cells, cells_per_macro),
            disclosures=tuple(disclosures),
        )

    # ------------------------------------------------------------------
    # Accumulator pricing for K stacks (P7.2)
    # ------------------------------------------------------------------

    def price_accumulation(
        self,
        stack: KStack,
        m_tokens: float,
        *,
        act_bytes: float = 0.0,
        pool: Optional[DigitalPoolSizing] = None,
    ) -> AccumulatorCost:
        """One K stack's accumulation cost BEYOND the analog pass walk.

        See :class:`AccumulatorCost` for the law. The short form: the walk fires
        the stack's ``d`` banks back to back, add ``j`` hides under pass
        ``j + 1``, and what is left over is the final drain plus any stall the
        pool cannot keep up with. A SPREAD stack is charged nothing here — the
        row-block partial-sum law already prices it end to end (D21).
        """
        pool = self.digital_pool_sizing() if pool is None else pool
        depth = int(stack.depth)
        width = int(stack.width)
        results = float(m_tokens) * width
        if depth <= 1:
            return AccumulatorCost(
                stack=stack,
                depth=depth,
                width=width,
                lanes=int(pool.lanes),
                partial_adds=0.0,
                hidden_adds=0.0,
                drain_cycles=0,
                stall_s=0.0,
                time_s=0.0,
                energy_pj=0.0,
                transport_partials=0,
                transport_bytes=0.0,
                priced_here=True,
                law="none: one K block reaches the output, so nothing accumulates",
            )
        if not stack.local:
            partials = len(dict.fromkeys(stack.macro_ids)) - 1
            return AccumulatorCost(
                stack=stack,
                depth=depth,
                width=width,
                lanes=int(pool.lanes),
                partial_adds=0.0,
                hidden_adds=0.0,
                drain_cycles=0,
                stall_s=0.0,
                time_s=0.0,
                energy_pj=0.0,
                transport_partials=partials,
                transport_bytes=results * partials * float(act_bytes),
                priced_here=False,
                law=(
                    "row_block_partial_sum: P3 emits the partial transfer and the pool "
                    "rowsum, P4 prices them with price_reduction (the chained-transport "
                    "sibling, D11)"
                ),
                disclosures=(
                    f"K stack {stack.owner.label} n[{stack.n_start}:{stack.n_end}] spans "
                    f"{len(set(stack.macro_ids))} macros, so its partials cross a link "
                    "and the EXISTING row-block law prices them. This accumulator law "
                    "charges it zero on purpose: two laws for one shape would be two "
                    "accountings of one metric (D21). transport_bytes is reported for "
                    "reconciliation, not to be added on top.",
                ),
            )
        lanes = max(1, int(pool.lanes))
        drain_cycles = _ceil_div(math.ceil(results), lanes)
        drain_s = drain_cycles / float(pool.pool_clock_hz)
        # One ADC pass of this stack: the time the NEXT partial takes to arrive,
        # which is what an intermediate add can hide under.
        pass_s = self.analog_op_time(m_tokens, 1)
        stall_s = max(0.0, drain_s - pass_s)
        hidden = max(0, depth - 2)
        disclosures: List[str] = []
        if float(self.card.pool_energy_per_add_pj) <= 0:
            disclosures.append(
                "accumulator energy is 0 pJ: cim.cards.<card>.pool_energy_per_add_pj is "
                "not declared, so the term reports zero rather than an invented "
                "per-add figure. The accumulator TIME is fully priced."
            )
        if stall_s > 0:
            disclosures.append(
                f"the pool drains one partial in {drain_s:.3e} s against an ADC pass of "
                f"{pass_s:.3e} s, so the accumulator STALLS the pass walk. P2.5 sizes "
                "the pool to consume the macro's peak result rate, so a stall means the "
                "sizing was overridden, not that the law disagrees with itself."
            )
        return AccumulatorCost(
            stack=stack,
            depth=depth,
            width=width,
            lanes=lanes,
            partial_adds=results * (depth - 1),
            hidden_adds=results * hidden,
            drain_cycles=drain_cycles,
            stall_s=stall_s,
            time_s=drain_s + hidden * stall_s,
            energy_pj=results * (depth - 1) * float(self.card.pool_energy_per_add_pj),
            transport_partials=0,
            transport_bytes=0.0,
            priced_here=True,
            law=(
                "in-macro K accumulation: (depth - 1) adds per output, all but the last "
                "hidden under the following ADC pass; the tail is the final drain"
            ),
            disclosures=tuple(disclosures),
        )

    def price_packing_accumulation(
        self,
        packing: DensePacking,
        m_tokens: float,
        *,
        act_bytes: float = 0.0,
        pool: Optional[DigitalPoolSizing] = None,
    ) -> PackingAccumulatorCost:
        """Every K stack of a packing, charged the way the machine pays.

        Time is the WORST MACRO's accumulator tail (macros fire concurrently,
        stacks sharing a macro serialize); area counts ONE accumulator per macro
        that hosts a local stack, because the K-inner walk keeps exactly one
        live at a time. See :class:`PackingAccumulatorCost`.
        """
        pool = self.digital_pool_sizing() if pool is None else pool
        costs = tuple(
            self.price_accumulation(stack, m_tokens, act_bytes=act_bytes, pool=pool)
            for stack in packing.k_stacks
            if stack.depth > 1
        )
        per_macro: "OrderedDict[int, float]" = OrderedDict()
        for cost in costs:
            if cost.priced_here and cost.time_s > 0:
                macro = cost.stack.sink_macro_id
                per_macro[macro] = per_macro.get(macro, 0.0) + cost.time_s
        hosts = {
            cost.stack.sink_macro_id
            for cost in costs
            if cost.priced_here and cost.depth > 1
        }
        adders = min(
            int(packing.cells_per_bank // max(1, int(self.card.params.rows))),
            max(1, int(pool.lanes)),
        )
        area_knob = float(self.card.pool_area_mm2_per_adder)
        disclosures: List[str] = []
        for cost in costs:
            for note in cost.disclosures:
                if note not in disclosures:
                    disclosures.append(note)
        if hosts and area_knob <= 0:
            disclosures.append(
                "accumulator area is 0 mm2: cim.cards.<card>.pool_area_mm2_per_adder is "
                "not declared, exactly as area_mm2_per_array: 0 disables area reporting. "
                "The accumulator COUNT and its adder width are reported regardless."
            )
        if hosts:
            disclosures.append(
                "the accumulator's holding REGISTER is not priced separately: no card "
                "declares a register area or energy, so the adder knob prices the adder "
                "and the register is a named gap rather than an invented number."
            )
        return PackingAccumulatorCost(
            stacks=costs,
            time_s=max(per_macro.values()) if per_macro else 0.0,
            energy_pj=sum(cost.energy_pj for cost in costs),
            area_mm2=len(hosts) * adders * area_knob,
            accumulators=len(hosts),
            adders_per_accumulator=adders if hosts else 0,
            transport_bytes=sum(cost.transport_bytes for cost in costs),
            local_stacks=sum(1 for cost in costs if cost.priced_here),
            spread_stacks=sum(1 for cost in costs if not cost.priced_here),
            disclosures=tuple(disclosures),
        )

    def report_dense_packing(self, packing: DensePacking) -> str:
        """The Invariant W line: floor, reached, and waste% itemized (D27)."""
        summary = packing.summary()
        lines = [
            "[FWS-CIM] dense packing (Invariant W, D27) — waste is a metric, not a rounding",
            f"  walk                    {packing.walk} (canonical; D26 refuses the rest)",
            f"  banks                   {packing.banks_used} used"
            f" ({packing.banks_per_macro} per macro, {packing.column_sets_per_bank}"
            f" mux slot(s) each, {packing.cells_per_bank} cells)",
            f"  macros                  {packing.macro_count}"
            f"  (global cell floor {packing.cell_floor_macros},"
            f" delta {packing.floor_delta_macros:+d})",
            f"  cells                   {packing.real_cells} real /"
            f" {packing.committed_cells} committed",
            f"  waste                   {packing.waste_pct:.3f}%"
            f"  = remainder {packing.remainder_cells} + tail {packing.tail_cells} cells",
            f"  K stacks                {summary['k_stacks']}"
            f" ({summary['local_k_stacks']} in-macro, {summary['spread_k_stacks']} spread)",
        ]
        for note in packing.disclosures:
            lines.append(f"  [NOTE] {note}")
        return "\n".join(lines)

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
        self,
        n_slices: Optional[int] = None,
        *,
        conv_kernel: int = 0,
        conv_channels: int = 0,
    ) -> DigitalPoolSizing:
        """Derive the per-macro pool from the card and its resident tiles.

        Sized never to block: enough lanes to consume the macro's peak result
        rate. The number is REPORTED, never a constraint (D12). Crossing the
        ADJ-4 disclosure share prints a note and changes nothing.

        SHORT DEPTHWISE CONV (ADJ-3, P2.6 4). Mamba-2's ``d_conv = 4``, LFM2's
        ``L = 3`` and the Qwen3.5 linear block's ``k = 4`` run HERE, on the
        per-macro pool, not on the shared digital chiplet — so the conv is not
        a timed op, it is absorbed into this derivation exactly as the
        shift-add trees are. Its work is ``k`` muls and ``k - 1`` adds per
        emitted channel per token (depthwise: one tap set per channel, no
        channel mixing to amortize), so:

            conv_column_duty = min(conv_channels, cols_adc * mux) / (cols_adc * mux)
            conv_ops_per_s   = result_rate * conv_column_duty * (2k - 1)
            conv_lanes       = ceil(conv_ops_per_s / pool_clock)

        The duty is what makes ``conv_channels`` load-bearing rather than
        decorative: a macro whose stored columns are only partly conv channels
        does proportionally less conv work, and a block wider than one macro
        saturates at 1.0 (the "every emitted result is a conv channel" worst
        case). ``conv_kernel = 0`` leaves every conv figure at 0 and the
        sizing bit-identical to P2.5.
        """
        card = self.card
        n_s = self.n_slices if n_slices is None else max(1, int(n_slices))
        pool_clock_hz = (
            float(card.pool_clock_ghz) * 1e9 if card.pool_clock_ghz > 0 else self.f_fabric_hz
        )
        rate = self.macro_result_rate_per_s()
        lanes = max(1, math.ceil(rate / pool_clock_hz))
        adders = lanes * (n_s - 1)
        conv_kernel = int(conv_kernel)
        conv_channels = int(conv_channels)
        conv_ops_per_result = 0
        conv_duty = 0.0
        conv_ops_per_s = 0.0
        conv_lanes = 0
        if conv_kernel > 0:
            if conv_channels < 1:
                raise ValueError(
                    "digital_pool_sizing: conv_kernel was declared without conv_channels. "
                    "A depthwise conv's work is k x channels; the channel count is not a "
                    "number this model may invent (ADJ-4)."
                )
            conv_ops_per_result = short_conv_ops_per_result(conv_kernel)
            stored_columns = int(card.stored_columns_per_set) * int(card.column_sets_per_macro)
            conv_duty = min(conv_channels, stored_columns) / stored_columns
            conv_ops_per_s = rate * conv_duty * conv_ops_per_result
            conv_lanes = max(1, math.ceil(conv_ops_per_s / pool_clock_hz))
        conv_area = conv_lanes * float(card.pool_area_mm2_per_adder)
        area = (adders + conv_lanes) * float(card.pool_area_mm2_per_adder)
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
            conv_kernel=conv_kernel,
            conv_channels=conv_channels,
            conv_ops_per_result=conv_ops_per_result,
            conv_column_duty=conv_duty,
            conv_ops_per_s=conv_ops_per_s,
            conv_lanes=conv_lanes,
            conv_area_mm2=conv_area,
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
        if sizing.conv_kernel > 0:
            line += (
                f" Absorbed short depthwise conv (ADJ-3): k = {sizing.conv_kernel} over "
                f"{sizing.conv_channels} channels costs {sizing.conv_ops_per_result} ops per "
                f"result at column duty {sizing.conv_column_duty:.4g}, adding "
                f"{sizing.conv_lanes} conv lanes ({sizing.conv_ops_per_s:.6g} ops/s) and "
                f"{sizing.conv_area_mm2:.6g} mm2."
            )
        if self.has_synthesis_library() and float(self.card.pool_area_mm2_per_adder) <= 0:
            pool = self.macro_pool_composition(sizing)
            line += (
                f" The area above is 0 because the card declares no "
                f"pool_area_mm2_per_adder; the pool's COMPOSED area (D32, "
                f"{pool.technology} measured blocks) is {pool.area_mm2:.6g} mm2 per "
                f"macro and {pool.power_w:.6g} W, itemized block by block in "
                "evaluation.digital_silicon.per_macro_pool. Two accountings, two names "
                "(D21): the declared knob is not silently replaced here."
            )
        digital_area = self.shared_digital_area_mm2()
        if digital_area > 0:
            source = (
                f"COMPOSES (D32, {self.synthesis_technology} measured blocks)"
                if self.has_synthesis_library()
                else "declares"
            )
            line += (
                f" Shared digital chiplet card '{self.digital_card.name}' {source} "
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

    # ------------------------------------------------------------------
    # Digital op laws (P2.6): the chiplet's vector/scan engine
    # ------------------------------------------------------------------

    @property
    def vector_lanes(self) -> int:
        """Scan/vector lanes of the shared digital chiplet — DERIVED (D31).

        THREE SOURCES, IN THIS ORDER, and every one of them says which it is:

        1. an EXPLICIT ``cim.cards.<card>.vector_lanes``. D31 retires the lane
           count as a design input, so a declared value is an OVERRIDE and it
           rides a disclosure (:meth:`vector_engine_disclosures`) naming what
           the beat would have derived instead when a derivation exists.
        2. the sizing :meth:`install_derived_engine` put here once the beat was
           known — the default, and the D31 answer.
        3. neither: :class:`EngineCapabilityError`, still. A lane count cannot
           be invented from the systolic array (a matmul engine) or from
           ``softmax_lanes`` (a softmax pipeline), and it cannot be derived
           without a beat, so an unmapped call with no override is refused
           rather than answered (ADJ-4).
        """
        card = self.digital_card
        if card.has_vector_engine:
            return int(card.vector_lanes)
        if self._engine_probe:
            return _PROBE_LANES
        if self._derived_engine is not None:
            return int(self._derived_engine.vector_lanes)
        raise EngineCapabilityError(
            f"shared digital chiplet card '{card.name}' has no vector engine: none was "
            "DERIVED for this run and the card declares no override. D31 sizes the "
            "scan/vector engine FROM THE PIPELINE BEAT — the evaluator measures the beat "
            "the analog stages set, then calls "
            "CimDeviceModel.derive_engine_sizing/install_derived_engine — so a call "
            "outside a mapped run has no beat to derive from. Set "
            "cim.cards.<card>.vector_lanes to override it explicitly (the value rides a "
            "disclosure, D31), or price this op inside a mapped run. There is no "
            "default: the systolic array is a matmul engine and softmax_lanes is a "
            "softmax pipeline, so deriving scan lanes from either would invent silicon "
            "(ADJ-4). Every SSM scan, delta-rule and RG-LRU law needs it."
        )

    @property
    def vector_lanes_provenance(self) -> str:
        """Where this run's lane count came from: declared, derived, or absent."""
        if self.digital_card.has_vector_engine:
            return PROVENANCE_DECLARED_COUNT
        if self._derived_engine is not None:
            return PROVENANCE_DERIVED_COUNT
        return "undetermined"

    # -- D31 pass A: probe the beat the ANALOG stages set ----------------

    #: The width the probe pass pretends to have. It is never composed, never
    #: reported and never priced: while probing, every vector op costs 0 s (see
    #: :meth:`price_vector_work`), and this value only exists so the work laws
    #: can run at all. It is not a default and not a fallback.
    ENGINE_PROBE_LANES = _PROBE_LANES

    @property
    def engine_probing(self) -> bool:
        return bool(self._engine_probe)

    def begin_engine_probe(self) -> None:
        """Enter pass A: vector work is COUNTED and costs no time (D31).

        The point of the probe is that the beat it produces is the beat the
        ANALOG stages set — the input D31 sizes the engine against. Timing the
        vector ops during pass A would make the beat depend on a width nobody
        has derived yet, which is the circularity this pass exists to break.
        """
        self._engine_probe = True
        self._probe_ops = []

    def end_engine_probe(self) -> Tuple[float, ...]:
        """Leave pass A and return every vector-op scalar-op count it saw."""
        self._engine_probe = False
        ops = tuple(self._probe_ops)
        self._probe_ops = []
        return ops

    def install_derived_engine(self, sizing: Optional["DerivedEngineSizing"]) -> None:
        """Install (or clear) the width D31 derived from this run's beat."""
        self._derived_engine = sizing

    @property
    def derived_engine(self) -> Optional["DerivedEngineSizing"]:
        return self._derived_engine

    @property
    def vector_clock_hz(self) -> float:
        """Vector-engine clock: the card's own, else the chiplet's clock."""
        return float(self.digital_card.vector_clock_ghz_effective) * 1e9

    @property
    def vector_pipeline_depth(self) -> int:
        """Vector-engine fill depth: the card's own, else the softmax depth."""
        return int(self.digital_card.vector_pipeline_depth_effective)

    def vector_engine_disclosures(self) -> Tuple[str, ...]:
        """Every default and relaxation the vector engine is running under (D21).

        Returned with each priced op so a report can print them; a knob that
        was INHERITED rather than declared says so, and an undeclared state
        bandwidth says that it bounds nothing.
        """
        card = self.digital_card
        notes = []
        if card.has_vector_engine:
            derived = self._derived_engine
            line = (
                f"vector_lanes = {int(card.vector_lanes)} is DECLARED on card "
                f"'{card.name}' and is therefore an OVERRIDE: D31 retires the scan/vector "
                "engine as a design input, because its width is derived from the pipeline "
                "beat and reported, never chosen or swept. The run honours the "
                "declaration; this line is the disclosure it rides on."
            )
            if derived is not None:
                # Only reachable when a caller derived a sizing by hand and
                # installed it beside a declared width: evaluate_fws's own path
                # skips the probe entirely for an override card, so a shipped
                # run never has a derivation to compare the declaration with.
                # P7.9: the config headers and the status entries used to claim
                # this sentence always rides an override. It does not, and they
                # no longer say so.
                line += (
                    f" The measured analog stage times would have derived "
                    f"{int(derived.vector_lanes)} lane(s) (ADJ-9)."
                )
            notes.append(line)
        elif self._derived_engine is not None:
            sizing = self._derived_engine
            row = sizing.binding_row
            target = row.target_s if row is not None else 0.0
            notes.append(
                f"vector_lanes = {int(sizing.vector_lanes)} is DERIVED (D31-v2, ADJ-9) "
                "to the ANALOG FLOOR: the smallest width for which every stage's "
                "digital per-stage time fits that stage's own measured ANALOG m-pass "
                f"time, with no margin (D28). Stage {int(sizing.binding_stage)} is "
                f"binding at a {target:.6g} s analog m-pass and the engine runs at "
                f"{sizing.utilization * 100:.3f}% duty inside it; "
                f"{len(sizing.analog_bound_stages)} of {len(sizing.per_stage)} stage(s) "
                "are analog-bound by construction."
            )
        if card.vector_clock_ghz <= 0:
            notes.append(
                f"vector_clock_ghz undeclared: inherited cim.fabric.clock_ghz = "
                f"{card.fabric.clock_ghz} GHz (one chiplet declares one clock)."
            )
        if card.vector_pipeline_depth <= 0:
            notes.append(
                f"vector_pipeline_depth undeclared: inherited "
                f"cim.fabric.softmax_pipeline_depth = {card.fabric.softmax_pipeline_depth}, "
                "the only elementwise-pipeline depth this chiplet declares."
            )
        if float(card.state_bytes_per_cycle) <= 0:
            notes.append(
                "state_bytes_per_cycle undeclared: recurrent-state traffic is REPORTED "
                "and bounds nothing. A scan whose state does not fit the engine's "
                "registers would be state-bound; this model cannot say so until a card "
                "declares the number (disclosed relaxation, not a silent zero)."
            )
        return tuple(notes)

    def price_vector_work(
        self, work: VectorWork, *, extra_disclosures: Tuple[str, ...] = ()
    ) -> DigitalOpCost:
        """Time declared work on the declared engine — the P2.6 stance in one method.

        ``arith_cycles = pipeline_depth + ceil(ops / lanes) - 1`` (the pipelined
        shape this module already uses for softmax lanes and the shift-add
        tree: one result per lane per cycle after fill).

        PHYSICS INVARIANT, true by construction and pinned by a test: since
        ``arith_cycles >= ceil(ops/lanes)``, the returned ``ops_per_s`` can
        never exceed ``lanes * clock``. No law in this module can outrun the
        silicon its card declares — the DESIGN2 section-8 erratum is the
        precedent for making that a checked property rather than a hope.
        """
        clock = self.vector_clock_hz
        depth = self.vector_pipeline_depth
        ops = work.ops
        if self._engine_probe and not self.digital_card.has_vector_engine:
            # D31 pass A. The work is RECORDED and costs nothing, so the beat
            # this pass measures is the one the analog stages set — the input
            # the sizing is derived from. Nothing here reaches a report: the
            # probe's pricing is thrown away and pass B prices the real engine.
            self._probe_ops.append(float(ops))
            return DigitalOpCost(
                law=work.law,
                validated=work.validated,
                work=work,
                lanes=0,
                clock_hz=clock,
                pipeline_depth=depth,
                arith_cycles=0,
                arith_time_s=0.0,
                state_time_s=0.0,
                time_s=0.0,
                ops_per_s=0.0,
                disclosures=(
                    "D31 engine PROBE pass: this vector op is counted, not timed, so "
                    "the measured beat is the one the analog stages set. The number "
                    "is discarded — the run is priced again on the derived engine.",
                ),
            )
        lanes = self.vector_lanes
        arith_cycles = depth + _ceil_div(math.ceil(ops), lanes) - 1
        arith_time = arith_cycles / clock
        state_bpc = float(self.digital_card.state_bytes_per_cycle)
        if state_bpc > 0:
            # Ceil on CYCLES, never on the declared bandwidth: rounding a
            # fractional bytes/cycle figure up would hand the engine silicon
            # the card never declared.
            state_time = math.ceil(work.state_bytes / state_bpc) / clock
        else:
            state_time = 0.0
        time_s = max(arith_time, state_time)
        return DigitalOpCost(
            law=work.law,
            validated=work.validated,
            work=work,
            lanes=lanes,
            clock_hz=clock,
            pipeline_depth=depth,
            arith_cycles=arith_cycles,
            arith_time_s=arith_time,
            state_time_s=state_time,
            time_s=time_s,
            ops_per_s=(ops / time_s) if time_s > 0 else 0.0,
            disclosures=self.vector_engine_disclosures() + tuple(extra_disclosures),
        )

    # --- SSM scan (P2.6 1, 2) ------------------------------------------

    def ssm_scan_work(
        self,
        tokens: float,
        *,
        d_inner: int,
        d_state: int,
        n_groups: int = 1,
        n_heads: int = 0,
        variant: str = "mamba2",
        chunk_size: int = 1,
        act_bytes: float = 1.0,
    ) -> VectorWork:
        """Pick the scan law: chunked SSD, or the per-token recurrence.

        The recurrence wins whenever chunking cannot apply — Mamba-1 (which
        has no chunked form), ``chunk_size <= 1``, or a single token (a DECODE
        step is always recurrent, which is also OPTIMA's stance: it prices
        prefill and decode with the same per-token stage time).
        """
        if variant == "mamba1" or int(chunk_size) <= 1 or float(tokens) <= 1:
            return ssm_recurrent_scan_work(
                tokens,
                d_inner=d_inner,
                d_state=d_state,
                n_groups=n_groups,
                n_heads=n_heads,
                variant=variant,
                act_bytes=act_bytes,
            )
        return ssd_chunked_scan_work(
            tokens,
            d_inner=d_inner,
            d_state=d_state,
            n_groups=n_groups,
            n_heads=n_heads,
            chunk_size=chunk_size,
            act_bytes=act_bytes,
        )

    def price_ssm_scan(self, tokens: float, **kwargs) -> DigitalOpCost:
        """Price one SSM scan call on the chiplet (D13: act x act goes here)."""
        return self.price_vector_work(self.ssm_scan_work(tokens, **kwargs))

    def price_ssm_block(
        self,
        ssm,
        hidden_dim: int,
        tokens: float,
        *,
        act_bytes: float = 1.0,
        chunked: bool = True,
    ) -> DigitalOpCost:
        """Price a parsed ``model_param.ssm`` block's scan (config -> law seam).

        Reads a :class:`config.SSMBlockConfig`: ``variant``, ``d_state``,
        ``n_groups``, ``n_heads``, ``chunk_size`` and the resolved ``d_inner``
        (explicit, else ``expand * hidden_dim``). This is the only place a
        model config touches the scan laws, so P3/P4 get one seam, not five.
        """
        variant = str(getattr(ssm, "variant", "mamba2"))
        d_inner = ssm.resolve_d_inner(int(hidden_dim))
        n_groups = int(getattr(ssm, "n_groups", None) or 1)
        n_heads = int(getattr(ssm, "n_heads", None) or 0)
        chunk_size = int(getattr(ssm, "chunk_size", None) or 1) if chunked else 1
        if variant == "mamba1":
            n_heads = 0
        return self.price_ssm_scan(
            tokens,
            d_inner=d_inner,
            d_state=int(ssm.d_state),
            n_groups=n_groups,
            n_heads=n_heads,
            variant=variant,
            chunk_size=chunk_size,
            act_bytes=act_bytes,
        )

    # --- Delta rule / RG-LRU (P2.6 3) -----------------------------------

    def price_delta_rule(self, tokens: float, **kwargs) -> DigitalOpCost:
        """Price one gated-delta-rule call. UNVALIDATED law — see the docstring
        of :func:`delta_rule_work`; the cost carries ``validated ==
        LAW_UNVALIDATED`` so a report can never present it as a checked number.
        """
        return self.price_vector_work(delta_rule_work(tokens, **kwargs))

    def price_linear_attention_block(
        self,
        linear_attention,
        tokens: float,
        *,
        chunk_size: int = 1,
        act_bytes: float = 1.0,
    ) -> DigitalOpCost:
        """Price a parsed ``model_param.linear_attention`` block (config -> law seam)."""
        return self.price_delta_rule(
            tokens,
            num_key_heads=int(linear_attention.num_key_heads),
            key_head_dim=int(linear_attention.key_head_dim),
            num_value_heads=int(linear_attention.num_value_heads),
            value_head_dim=int(linear_attention.value_head_dim),
            chunk_size=chunk_size,
            output_gate=bool(getattr(linear_attention, "output_gate", True)),
            decay_gate=bool(getattr(linear_attention, "decay_gate", True)),
            act_bytes=act_bytes,
        )

    def price_rg_lru(self, tokens: float, *, width: int, act_bytes: float = 1.0) -> DigitalOpCost:
        """Price one RG-LRU (Griffin / RecurrentGemma) call. UNVALIDATED."""
        return self.price_vector_work(rg_lru_work(tokens, width=width, act_bytes=act_bytes))

    # --- Sliding-window attention (P2.6 5) ------------------------------

    @staticmethod
    def window_context(context: int, window: Optional[int]) -> int:
        """Context a sliding-window attention op actually sees: min(context, window).

        ``window`` None or <= 0 means the layer is global and the full context
        stands — the ``local_global_interval = 1`` case P1 calls inert.
        """
        context = int(context)
        if window is None or int(window) <= 0:
            return context
        return min(context, int(window))

    def sliding_window_decode_timing(
        self,
        context: int,
        window: Optional[int],
        batch_size: Optional[int] = None,
        kv_heads: Optional[int] = None,
        tp: int = 1,
    ) -> AttentionTiming:
        """SWA decode: the existing folded SA law at n = min(context, window).

        A THIN WRAPPER on purpose (P2.6 5). Sliding-window attention is not a
        new law — it is the same act x act attention on the same shared digital
        chiplet (D13) reading a shorter context. The only new content is where
        the context comes from, so the only new code is
        :meth:`window_context`. Anything more would be a second accounting of
        one metric.
        """
        return self.decode_attention_timing(
            self.window_context(context, window),
            batch_size=batch_size,
            kv_heads=kv_heads,
            tp=tp,
        )

    def sliding_window_prefill_timing(
        self,
        seq_len: Optional[int] = None,
        window: Optional[int] = None,
        tp: int = 1,
        streams: int = 1,
    ) -> AttentionTiming:
        """SWA prefill: the folded prefill law at n = min(S, window).

        RELAXATION, disclosed: the folded-K law prices ONE score block of
        ``m = S * shared_heads`` query rows against ``n`` context columns, so
        capping ``n`` at the window charges the band as a rectangle rather
        than a diagonal band of width ``window``. It is an UPPER bound on the
        banded work and a lower bound on the full-context work; it is the same
        shape the pass-1 prefill law already uses, so it introduces no second
        accounting.
        """
        p = self.params
        s = int(p.seq_len if seq_len is None else seq_len)
        return self.attention_call_timing(
            m=s * p.shared_heads,
            k=p.head_dim,
            n=self.window_context(s, window),
            tp=tp,
            streams=streams,
        )

    # --- MLA (P2.6 6, D6) ------------------------------------------------

    def mla_attention_timing(
        self,
        context: int,
        *,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        batch_size: int = 1,
        tp: int = 1,
        query_rows: Optional[int] = None,
        streams: Optional[int] = None,
    ) -> AttentionTiming:
        """Price MLA attention through the SA law at the MLA CALL DIMS (D6).

        The absorbed MLA form is what a decode step actually runs: the query
        is projected into the LATENT space, so the score call contracts over
        ``kv_lora_rank + qk_rope_head_dim`` (the compressed KV plus the shared
        decoupled RoPE key) against the context, and the output call reads the
        latent back at width ``kv_lora_rank``. Both are act x act, so both run
        on the shared digital chiplet (D13) under the SAME folded-K law as
        MHA/GQA — no second attention accounting exists.

        ``kv_heads = 1`` IS the modelling statement, not a shortcut: the KV
        latent is ONE shared rank. ``heads_chip`` therefore stays 1 at every
        tp, so tp buys no attention-side reduction at all. That is the D6
        "show exactly how ugly" finding, expressed as a number the report can
        print rather than a sentence in a plan.

        ``query_rows`` defaults to ``num_heads`` (one decode step); a prefill
        call passes ``S * num_heads``.
        """
        latent_k = int(kv_lora_rank) + int(qk_rope_head_dim)
        m = int(num_heads) if query_rows is None else int(query_rows)
        b = int(batch_size)
        return self.attention_call_timing(
            m=m,
            k=latent_k,
            n=int(context),
            kv_heads=1,
            tp=tp,
            softmax_tokens=b * m,
            streams=b if streams is None else int(streams),
        )

    def mla_attention_timing_from_config(
        self, attention, context: int, *, batch_size: int = 1, tp: int = 1, seq_len: int = 1
    ) -> AttentionTiming:
        """MLA timing from a parsed ``model_param.attention`` block (config -> law seam)."""
        if str(getattr(attention, "attention_type", "")).lower() != "mla":
            raise ValueError(
                "mla_attention_timing_from_config: attention_type must be 'mla' "
                f"(got {getattr(attention, 'attention_type', None)!r})"
            )
        return self.mla_attention_timing(
            context,
            num_heads=int(attention.num_heads),
            kv_lora_rank=int(attention.kv_lora_rank),
            qk_rope_head_dim=int(attention.qk_rope_head_dim),
            batch_size=batch_size,
            tp=tp,
            query_rows=int(seq_len) * int(attention.num_heads),
        )

    def mla_kv_replication(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int = 0,
        tp: int = 1,
        layers: int = 1,
        weight_bytes: float = 1.0,
    ) -> MLAReplication:
        """The MLA KV-replication AREA consequence, computed (D6, P1 finding 2).

        Which tensors are replicated, and why exactly those: every matrix
        whose contraction dimension IS the shared latent rank reads or writes
        the WHOLE latent, and the latent is not head-sliced, so a tp shard
        cannot hold a slice of it. Those tensors sit on every shard:

        * ``W_DKV`` — ``hidden x (kv_lora_rank + qk_rope_head_dim)``, the KV
          down-projection that produces the latent.
        * ``W_DQ`` — ``hidden x q_lora_rank``, the query down-projection, when
          the model uses one.
        * ``W_UK`` — ``kv_lora_rank x (num_heads * qk_nope_head_dim)``.
        * ``W_UV`` — ``kv_lora_rank x (num_heads * v_head_dim)``.

        The up-projections are the case P1 finding 2 names, and the stance is
        kept here: they read the full latent on every shard, so the shard
        stores them whole.

        AREA is computed in OUR area law, not converted from bytes: weights
        are cells, so the figure is ``arrays(K, N) * area_mm2_per_array``
        summed over those matrices. Bytes are reported beside it because P4
        wants both, and they are two named metrics with two definitions — not
        two accountings of one (D21). ``extra_*`` is what tp actually PAYS:
        ``(tp - 1)`` further copies of the same tensors.
        """
        tp = max(1, int(tp))
        layers = max(1, int(layers))
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        kv_lora_rank = int(kv_lora_rank)
        latent_in = kv_lora_rank + int(qk_rope_head_dim)
        shapes = [("W_DKV", hidden_dim, latent_in)]
        if int(q_lora_rank) > 0:
            shapes.append(("W_DQ", hidden_dim, int(q_lora_rank)))
        shapes.append(("W_UK", kv_lora_rank, num_heads * int(qk_nope_head_dim)))
        shapes.append(("W_UV", kv_lora_rank, num_heads * int(v_head_dim)))
        matrices = tuple(
            (name, k, n, self.arrays(k, n)) for name, k, n in shapes
        )
        latent_names = ("W_DKV", "W_DQ")
        latent_bytes = float(
            sum(k * n for name, k, n, _ in matrices if name in latent_names)
        ) * float(weight_bytes) * layers
        up_bytes = float(
            sum(k * n for name, k, n, _ in matrices if name not in latent_names)
        ) * float(weight_bytes) * layers
        arrays_per_shard = sum(a for _, _, _, a in matrices) * layers
        area_per_shard = arrays_per_shard * float(self.analog.area_mm2_per_array)
        replicated_bytes = latent_bytes + up_bytes
        return MLAReplication(
            tp=tp,
            layers=layers,
            latent_bytes_per_shard=latent_bytes,
            up_projection_bytes_per_shard=up_bytes,
            replicated_bytes_per_shard=replicated_bytes,
            extra_replicated_bytes=(tp - 1) * replicated_bytes,
            replicated_arrays_per_shard=arrays_per_shard,
            extra_replicated_arrays=(tp - 1) * arrays_per_shard,
            replicated_area_mm2_per_shard=area_per_shard,
            extra_replicated_area_mm2=(tp - 1) * area_per_shard,
            matrices=matrices,
        )

    def mla_kv_replication_from_config(
        self, attention, hidden_dim: int, *, tp: int = 1, layers: int = 1, weight_bytes: float = 1.0
    ) -> MLAReplication:
        """MLA replication figures from a parsed ``model_param.attention`` block."""
        if str(getattr(attention, "attention_type", "")).lower() != "mla":
            raise ValueError(
                "mla_kv_replication_from_config: attention_type must be 'mla' "
                f"(got {getattr(attention, 'attention_type', None)!r})"
            )
        return self.mla_kv_replication(
            hidden_dim=hidden_dim,
            num_heads=int(attention.num_heads),
            kv_lora_rank=int(attention.kv_lora_rank),
            qk_nope_head_dim=int(attention.qk_nope_head_dim),
            qk_rope_head_dim=int(attention.qk_rope_head_dim),
            v_head_dim=int(attention.v_head_dim),
            q_lora_rank=int(attention.q_lora_rank or 0),
            tp=tp,
            layers=layers,
            weight_bytes=weight_bytes,
        )

    def report_mla_replication(self, replication: MLAReplication) -> str:
        """The one-line D6 disclosure P4 prints: area paid to KV replication."""
        names = ", ".join(name for name, _, _, _ in replication.matrices)
        return (
            f"[FWS-CIM] MLA KV replication (D6): tp = {replication.tp} stores {names} on "
            f"EVERY shard because the KV latent is one shared rank, not a per-head slice. "
            f"Per shard: {replication.replicated_arrays_per_shard} arrays = "
            f"{replication.replicated_area_mm2_per_shard:.6g} mm2 "
            f"({replication.replicated_bytes_per_shard:.6g} weight bytes, of which "
            f"{replication.up_projection_bytes_per_shard:.6g} are the up-projections). "
            f"Area PAID to replication: {replication.extra_replicated_area_mm2:.6g} mm2 "
            f"({replication.extra_replicated_arrays} arrays) over "
            f"{replication.layers} layer(s)."
        )
